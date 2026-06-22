"""
Interactive command-line interface for SimpleAgent.

Responsibilities:
    - load environment variables from .env
    - build the tool registry (filesystem + web + MCP tools),
    - connect the MCP bridge (spawns the demo MCP server),
    - construct the Agent and run an async REPL,
    - shut the MCP subprocess down cleanly on exit.

Run with:
    uv run python main.py
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown

from agent import Agent
from tools.registry import ToolRegistry
import tools.builtin_fs as builtin_fs
import tools.builtin_web as builtin_web
from tools.mcp_bridge import MCPBridge

console = Console()

_BANNER = """\
[bold cyan]SimpleAgent[/bold cyan] - a from scratch AI agent.
Type your message and press Enter. Commands:
    [yellow]/tools[/yellow] list available tools
    [yellow]/reset[/yellow] clear the conversation
    [yellow]/exit[/yellow]  quit
"""

def _log_tool_call(name: str, args:dict[str,Any])-> None:
    """Dim one-line log so the user can watch what the agent is doing."""
    console.print(f"[dim]   -> {name}({args})[/dim]")

async def _build_registry()->tuple[ToolRegistry, MCPBridge]:
    """Assemble all tools. Returns the registry and the (connected) MCP bridge"""
    registry = ToolRegistry()

    #Built-in sync Tools.
    builtin_fs.register(registry)
    builtin_web.register(registry)

    #MCP tools, discovered from our demo server over stdio.
    bridge = MCPBridge(
        registry,
        command=sys.executable,
        args=["-m", "mcp_server.server"],
        name="mcp"
    )
    try:
        registered = await bridge.connect()
        console.print(f"[dim]Connected to MCP server, tools: {registered}[/dim]")
    except Exception as e:
        console.print(f"[yellow]Warning: MCP server unavailable ({e})."
                      f"Continuing without MCP tools.[/yellow]")

    return registry, bridge

async def _repl(agent:Agent, registry: ToolRegistry)->None:
    """The read-eval-print loop."""
    console.print(_BANNER)

    while True:
        try:
            user_input = console.input("[bold green]>[/bold green]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            break

        if not user_input:
            continue

        #Slash commands
        if user_input in ("/exit", "/quit"):
            break
        if user_input == "/reset":
            agent.reset()
            console.print("[dim]Conversation cleared.[/dim]")
            continue
        if user_input == "/tools":
            console.print("[dim]Available tools:[/dim]")
            for name in registry.names():
                console.print(f"    [cyan]{name}[/cyan]")
            continue
        #Normal message: run a full agent turn.
        try:
            reply = await agent.run_turn(user_input)
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            continue

        console.print(Markdown(reply))

async def main()->None:
    load_dotenv()   # populate environment from .env before anything reads it.

    registry, bridge = await _build_registry()

    try:
        agent = Agent(registry, tool_logger=_log_tool_call)
    except Exception as e:
        console.print(f"[red]{e}[/red]")
        await bridge.aclose()
        return
    
    try:
        await _repl(agent, registry)
    finally:
        #Always shut the MCP subprocess down, even on error or Ctrl+C
        await bridge.aclose()
        console.print("[dim]Goodbye.[/dim]")


if __name__ == "__main__":
    asyncio.run(main())
