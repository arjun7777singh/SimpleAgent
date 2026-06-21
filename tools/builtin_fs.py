"""
Sandboxed fielsystem tools for the agent.
Exposes a file/directory tools the LLM can call: list, read, write, make directory,
and delete. Every path the model supplies is confined to a single sandbox root
(AGENT_WORKSPACE_ROOT, default ./agent_workspace). Any path that resolves outside the sanbox
-- via `..`, an absolute path, or a symlink -- is rejected before any I/O happens.

This is the most security-sensitive module in the project: the LLM chosses the paths,
and its choices may be influenced by the untrusted web content. The single `_resolve()` chokepoint
below is what keeps those choices inside the jail.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

from tools.registry import ToolRegistry

#How many characters of a file we will return to the model at once.
_MAX_READ_CHARS = 8000

def _sandbox_root()-> Path:
    """The absolute, real path of the sandbox directory (created if missing)."""
    root = os.environ.get("AGENT_WORKSPACE_ROOT", "./agent_workspace")
    root_path = Path(root).expanduser().resolve()
    root_path.mkdir(parents=True, exist_ok=True)
    return root_path

def _resolve(user_path: str)-> Path:
    """Resolve a user-supplied path INSIDE the sandbox, or raise.
    Security chokepoint. Steps:
        1. Treat the path as relative to the sandbox root. A leading slash is
        stripped so '/etc/passwd' becomes '<root>/etc/passwd', not the real
        system path.
        2. Resolve to a real absolute path -- this collapses '..' segments and 
        follows any symlinks.
        3. Verify the result is the root itself or a descendant of it.
    """
    if not isinstance(user_path, str) or not user_path.strip():
        raise ValueError("'path' is required and must be a non-empty string.")
    root = _sandbox_root()

    #Strip leading slashes/backslashes so absolute-looking paths stay jailed.
    cleaned = user_path.strip().lstrip("/\\")
    candidate = (root / cleaned).resolve()

    #Containment check. is_relative_to is available on Python 3.9+.
    if candidate != root and not candidate.is_relative_to(root):
        raise ValueError(
            f"Path {user_path!r} is outside the sandbox and is not allowed."
        )
    return candidate

def _rel(path:Path)-> str:
    """Display a path relative to the sandbox root (never leak absolute paths.)"""
    try:
        return str(path.relative_to(_sandbox_root())) or "."
    except ValueError:
        return str(path)
    
#-------------------------------------------------------------------------------handlers
def _list_directory(args: dict[str,Any])->str:
    target = _resolve(args.get("path","."))
    if not target.exists():
        raise FileNotFoundError(f"Directory does not exist: {_rel(target)}")
    if not target.is_dir():
        raise NotADirectoryError(f"Not a directory: {_rel(target)}")
    
    entries = sorted(
        target.iterdir(), key = lambda p: (p.is_file(),p.name.lower())
    )
    if not entries:
        return f"{_rel(target)}/ is empty."
    
    lines = [f"Contents of {_rel(target)}/:"]
    for entry in entries:
        kind = "dir" if entry.is_dir() else "file"
        size = "" if entry.is_dir() else f" ({entry.stat().st_size} bytes)"
        lines.append(f" [{kind}] {entry.name}{size}")
    return "\n".join(lines)

def _read_file(args: dict[str,Any]) -> str:
    target = _resolve(args.get("path",""))
    if not target.exists():
        raise FileNotFoundError(f"File does not exist: {_rel(target)}")
    if not target.is_file():
        raise IsADirectoryError(f"Not a file: {_rel(target)}")
    
    text = target.read_text(encoding="utf-8", errors="replace")
    if len(text) > _MAX_READ_CHARS:
        text = text[:_MAX_READ_CHARS] + (
            f"\n\n[...truncated at {_MAX_READ_CHARS} characters...]"
        )
    return f"Contents of {_rel(target)}:\n\n{text}"

def _write_file(args:dict[str,Any])->str:
    target = _resolve(args.get("path",""))
    content = args.get("content","")
    if not isinstance(content,str):
        raise ValueError("'content' must be a string.")
    if target.exists() and target.is_dir():
        raise IsADirectoryError(f"Cannot write: {_rel(target)} is a directory.")
    
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"Wrote {len(content)} characters to {_rel(target)}"

def _create_directory(args:dict[str,Any])->str:
    target = _resolve(args.get("path",""))
    if target.exists():
        if target.is_dir():
            return f"Directory already exists: {_rel(target)}/"
        raise FileExistsError(f"A file already exists at: {_rel(target)}")
    target.mkdir(parents=True, exist_ok=True)
    return f"Created directory: {_rel(target)}"

def _delete_path(args: dict[str,Any])->str:
    target = _resolve(args.get("path",""))

    #Second safety barrier: refuse unless the model explicitly confirms.
    if args.get("confirm") is not True:
        raise ValueError(
            "Deletion requires 'confirm': true. Re-issue the call with" \
            "confirm set to true once you are sure."
        )
    #Never allow deleting the sandbox root itself.
    if target==_sandbox_root():
        raise ValueError("Refusing to delete the sandbox root directory.")
    if not target.exists():
        raise FileNotFoundError(f"Nothing to delete at: {_rel(target)}")
    
    if target.is_dir():
        shutil.rmtree(target)
        return f"Deleted directory (recursively): {_rel(target)}"
    target.unlink()
    return f"Deleted file: {_rel(target)}"

#-------------------------------------------------------------------------------schemas

_PATH_PROP={
    "type":"string",
    "description":"Path relative to the agent workspace sandbox. "
    "Absolute paths and '..' that escape the sandbox are rejected.",
}

def register(registry:ToolRegistry)->None:
    """Register all filesystem tools into the given registry."""
    registry.register_sync(
        name="list_directory",
        description="List the files and subdirectories inside a directory"
        "within the workspace. Defaults to the workspace root.",
        parameters={
            "type":"object",
            "properties":{"path": {**_PATH_PROP, "description":"Directory to list. Defaults to '.'"}},
            "required":[]
        },
        handler=_list_directory,
    )
    registry.register_sync(
        name="read_file",
        description="Read and return the text contents of a file in the workspace.",
        parameters={
            "type":"object",
            "properties":{"path":_PATH_PROP},
            "required": ["path"],
        },
        handler = _read_file,
    )
    registry.register_sync(
        name="write_file",
        description="Write text to a file in the workspace, creating parent "
        "directories as needed. Overwrites the file if it already exists.",
        parameters={
            "type":"object",
            "properties":{"path":_PATH_PROP,"content":{"type":"string","description":"The full text to write to the file."}},
            "required": ["path","content"],
        },
        handler = _write_file,
    )
    registry.register_sync(
        name="create_directory",
        description="Create a new directory (and any missing parents) in the workspace.",
        parameters={
            "type":"object",
            "properties":{"path":_PATH_PROP},
            "required": ["path"],
        },
        handler = _create_directory,
    )
    registry.register_sync(
        name="delete_path",
        description="Delete a file or directory in the workspace. Requires "
        "'confirm': ture. Deleting a directory removes everything inside it.",
        parameters={
            "type":"object",
            "properties":{
                "path": _PATH_PROP,
                "confirm": {
                    "type":"boolean",
                    "description": "Must be set to true to actually delete. A safety check.",
                },
            },
            "required":["path","confirm"],
        },
        handler=_delete_path,
    )