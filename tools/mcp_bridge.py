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



