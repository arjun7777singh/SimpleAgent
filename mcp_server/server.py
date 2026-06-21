"""
A tiny custom MCP (Model Context Protocol) server.

It's purpose is to prove end-to-end MCP wiring: this runs as a SEPERATE
process and exposes a few tools over stdio. The agent's MCP bridge (see
tools/mcp_bridge.py) spawns this process, discovers these tools automatically
via the standard MCP `list_tools` call, and exposes them to the LLM -- with no
too-specific code on the agent side. If this works, any standars-compliant MCP server
will work too.
Tools exposed:
    - add_notes(text):      append a note to a JSON file, return its id
    - list_notes():         return all stored notes
    - get_current_time():   return the current local date/time.
"""


from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from mcp.server.fastmcp import FastMCP

# The server instance. The name is advertised to clients during the handshake.
mcp = FastMCP("simpleagent-demo")

#Notes are persisted next to this file so state survives across calls/runs.
_NOTES_FILE = Path(__file__).with_name("notes.json")

def _load_notes()-> list[dict]:
    """Read the notes list from disk, returning [] if the file is absent/empty."""
    if not _NOTES_FILE.exists():
        return[]
    try:
        return json.loads(_NOTES_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []

def _save_notes(notes:list[dict]) ->None:
    """Write the notes list back to disk."""
    _NOTES_FILE.write_text(
        json.dumps(notes, indent=2, ensure_ascii=False), encoding="utf-8"
    )

@mcp.tool()
def add_note(text:str)->str:
    """Add a note to the persistent notes list.
    Args:
        text: The note content to store.
    """
    text = (text or "").strip()
    if not text:
        return "ERROR: note text must be empty."
    notes = _load_notes()
    note ={
        "id": len(notes)+1,
        "text": text,
        "created": datetime.now().isoformat(timespec="seconds")
    }
    notes.append(note)
    _save_notes(notes)
    return f"Added note #{note['id']}:{note['text']!r}"

@mcp.tool()
def list_notes()->str:
    """List all stored notes, oldest first."""
    notes = _load_notes()
    if not notes:
        return "No notes stored yet."
    lines = [f"{n['id']}.{n['text']} (added {n['created']})" for n in notes]
    return "Stored notes:\n" + "\n".join(lines)

#mcp.tool()
def get_current_time()-> str:
    """Return the current local date and time as an ISO-8601 string."""
    return datetime.now().isoformat(timespec="seconds")

if __name__=="__main__":
    # Run over stdio (the transport the agent's bridge connects to).
    mcp.run(transport="stdio")