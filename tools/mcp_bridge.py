"""
Bridge between the agent's ToolRegistry and external MCP servers.

Given a command to launch a MCP server (e.g. `python -m mcp_server.server),
this:
    1. spawns the server as a subprocess and completes the MCP handshake,
    2. calls the standard `list_tools` to discover what the server offers,
    3. translates each MCP too's JSON-schema into our Tool format, and 
    4. registers an async handler that invokes the tool over MCP.

None of this is specific to our demo server -- any standards-compliant MCP
server works the same way. That is the whole point of MCP.

Lifecycle: call `await connect()` once at startup and `await aclose()` at 
shutdown. The session and subprocess stay alive in between via an 
AsyncExitStack.
"""

from __future__ import annotations

import shutil
from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from tools.registry import Tool, ToolRegistry

class MCPBridge:
    """Connects to one MCP server and registers its tools into a ToolRegistry"""
    def __init__(
            self,
            registry:ToolRegistry,
            command:str,
            args: list[str] | None = None,
            *,
            name:str="mcp"
    )->None:
        self._registry=registry
        self._command = command
        self._args = args or []
        self._name = name # prefix for tool names, avoids cross-server clashes
        self._stack = AsyncExitStack()
        self._session: ClientSession | None = None
        self._connected = False
    # ------------------------------------------------------------------------lifecycle
    
    async def connect(self)-> list[str]:
        """Spawn the server, handshake, and register its tools.
        
        Returns the list of registered (prefixed) tool names.
        """
        if self._connected:
            raise RuntimeError("MCPBridge is already connected.")
        
        # `command` must be resolvable on PATH; given a clear error if not.
        if shutil.which(self._command) is None:
            raise FileNotFoundError(
                f"MCP server command {self._command!r} was not found on PATH."
            )
        
        server_params = StdioServerParameters(
            command = self._command,
            args=self._args,
        )

        # Enter the stdio transport, then the client session, keeping both
        # open for the bridge's lifetime via the exit stack.
        read, write = await self._stack.enter_async_context(
            stdio_client(server_params)
        )
        session = await self._stack.enter_async_context(
            ClientSession(read,write)
        )
        await session.initialize()
        self._session = session
        self._connected = True

        return await self._register_tools()
    
    async def aclose(self) -> None:
        """Shut down the session and terminate the server subprocess."""
        await self._stack.aclose()
        self._session = None
        self._connected = False
    
    async def _register_tools(self)-> list[str]:
        """Discover tools via MCP and add them to the registry."""
        assert self._session is not None
        listed = await self._session.list_tools()

        registered: list[str] = []
        for mcp_tool in listed.tools:
            local_name = f"{self._name}__{mcp_tool.name}"
            self._registry.register(
                Tool(
                    name=local_name,
                    description=mcp_tool.description or f"MCP tool {mcp_tool.name}",
                    parameters=_normalize_schema(mcp_tool.inputSchema),
                    handler=self._make_handler(mcp_tool.name),

                )
            )
            registered.append(local_name)
        return registered
    
    async def _make_handler(self,remote_name:str):
        """Build an async handler that calls one remote MCP tool by name."""
        
        async def handler(args: dict[str,Any])->str:
            if self._session is None:
                return "ERROR: MCP session is not connected."
            result = await self._session.call_tool(remote_name, args)
            return _extract_text(result)
        
        return handler
    
# -----------------------------------------------------------------------------#helpers

def _normalize_schema(input_schema: dict[str, Any] | None)-> dict[str,Any]:
    """Turn an MCP inputSchema into a valid OpenAI function-parameters object.
    MCP inputSchema is already JSON Schema, so this is mostly a pass-through.
    We only guarantee a usable object schema when the server omits one.
    """
    if not input_schema:
        return {"type":"object", "properties":{}}
    schema=dict(input_schema)
    schema.setdefault("type","object")
    schema.setdefault("properties",{})
    return schema

def _extract_text(result: Any)->str:
    """Pull text out of a MCP CallToolResult's content blocks."""
    # The SDK exposes `.content`: a list of content blocks. Text blocks have
    # `.type == "text"` and a `.text` attribute. We join all the text blocks.
    content = getattr(result, "content", None)
    if not content:
        return ""
    
    parts: list[str]=[]
    for block in content:
        text = getattr(block, "text", None)
        if text is not None:
            parts.append(text)
        else:
            # Non - text block (image, resource, ref, ...) - describe it.
            parts.append(f"[non-text content: {getattr(block, 'type', 'unknown')}]")

    text = "\n".join(parts)

    # Surface tool-side errors clearly to the LLM.
    if getattr(result, "isError", False):
        return f"ERROR (from MCP tool): {text}"
    return text






