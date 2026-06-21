"""
Tool registry for the agent.

Holds all tools the LLM can call, exposes them as OpenAI-format JSON schemas,
and dispatches tool calls by name. Every tool handler is async; sync functions
can be registered via `register_sync` which wraps them.

Tools results are always strings (OpenAI's tool-message format requires text).
Exceptions raised by handlers are caught and returned as error strings so the
LLM can see what went wrong and retry.
"""

from __future__ import annotations

import asyncio
import inspect
import json
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

# A handler takes a dict of JSON-decoded arguments and returns a string result.
# It is always async at the registry level (sync funcs are wrapped on register)

ToolHandler = Callable[[dict[str,Any]], Awaitable[str]]

@dataclass(frozen=True)
class Tool:
    """One callable capability exposed to the LLM."""
    name:str
    description:str
    """JSON schema describing the tool's arguments. Must be a vaild OpenAI
    `function.parameters` object: {"type":"object", "properties":...}.
    """
    parameters:dict[str,Any]
    handler:ToolHandler

    def to_openai_schema(self)->dict[str,Any]:
        """Convert to the dict shape the OpenAI `tools=[...]` array expects."""
        return{
            "type":"function",
            "function":{
                "name":self.name,
                "description": self.description,
                "parameters": self.parameters
            },
        }
    
class ToolRegistry:
    """Holds tools, emits OpenAI schemas, and dispatches calls."""

    def __init__(self)-> None:
        self._tools:dict[str,Tool]={}

    #-----------------------------------------------------------------------register
    def register(self, tool:Tool)-> None:
        """Register a tool whose handler is already async."""
        if tool.name in self._tools:
            raise ValueError(f"Tool name {tool.name} already registered.")
        self._tools[tool.name]=tool
    
    def register_sync(
            self,
            name:str,
            description:str,
            parameters: dict[str,Any],
            handler: Callable[[dict[str,Any]], Any]
    ) -> None:
        """Register a sync handler. It will be run in a thread when called.
        Use this for blocking I/O (file ops, requests) so we don't stall the 
        asyncio event loop that the agent runs under.
        """
        async def async_wrapper(args: dict[str, Any]) ->str:
            result = await asyncio.to_thread(handler, args)
            return _coerce_to_str(result)
        
        self.register(
            Tool(
                name=name,
                description=description,
                parameters=parameters,
                handler=async_wrapper,
            )
        )
    
    #-----------------------------------------------------------------------inspect

    def names(self)->list[str]:
        """List of registered tool names."""
        return list(self._tools)
    
    def openai_schemas(self) -> list[dict[str,Any]]:
        """The list to pass as `tools=` to chat.completions.create"""
        return [t.to_openai_schema() for t in self._tools.values()]
    
    #-----------------------------------------------------------------------dispatch
    async def dispatch(self, name:str, raw_arguments:str, tool_call_id: str|None=None):
        """Run the named tool with JSON-string arguments from the LLM.
        `tool_call_id` is the OpenAI generated id for this specific invocation;
        threading it through gives us correlation in logs and errors.
        Never raises: all errors are caught and returned as as string so the model
        receives them in the `tool` message and can decide how to react.
        """
        tool = self._tools.get(name)
        if tool is None:
            return _error(f"Unknown tool {name!r}. Available: {self.names()}", tool_call_id)
        try:
            args = json.loads(raw_arguments) if raw_arguments else{}
        except json.JSONDecodeError as e:
            return _error(f"Arguments were not valid JSON: {e}",tool_call_id)
        
        if not isinstance(args,dict):
            return _error(f"Arguments must be JSON objects, got {type(args).__name__}", tool_call_id)
        
        try:
            result = await tool.handler(args)
        except Exception as e:
            return _error(f"{type(e).__name__}: {e}", tool_call_id)
        return _coerce_to_str(result)

#-----------------------------------------------------------------------helpers
def _coerce_to_str(value:Any)->str:
    """OpenAI tool messages must be string. Convert anything else sensibly."""
    if isinstance(value,str):
        return value
    if value is None:
        return ""
    try:
        return json.dumps(value, ensure_ascii=False, indent=2, default=str)
    except (TypeError, ValueError):
        return str(value)
    
def _error(message:str, tool_call_id:str | None = None):
    """Standard error envelope so the LLM learns to recognise failures."""
    if tool_call_id:
        return f"ERROR [{tool_call_id}]: {message}"
    return f"ERROR: {message}"

#Sanity check to import time -- catches typos in the type alias quickly.
assert inspect.iscoroutinefunction(
    ToolRegistry.dispatch
), "dispatch must be async"