"""
The agent: an OpenAI - compatible chat loop with tool calling.

`Agent.run_turn(user_input)` runs one full turn, which may involve serveral
round-trips with the model as it calls tools and reads their results. The loop
ends when the model returns a message with no tool calls (its final answer)
or when a safety cap on iterations is reached.

The LLM client is provider-agnostic: it talks to any OpenAI-compatible endpoint
(OpenAI, Groq, Gemini's compact layer, Ollama, ...) selected purely via the 
LLM_BASE_URL /LLM_API_KEY / LLM_MODEL environment variables.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Callable

from openai import AsyncOpenAI
from tools.registry import ToolRegistry

#Hard cap on tool-call rounds within a single turn, so a misbehaving model
# cannot loop forever calling tools.
_MAX_ITERATIONS = 10

_DEFAULT_SYSTEM_PROMPT="""\
You are SimpleAgent, a helpful AI assistant with access to tools.
 You can:
 - search the web and fetch web pages for current information.
 - read, write, list, and delete files within a sandboxed workspace,
 - use additional tools provided over MCP.

Guidelines:
- Prefer using a tool over gussing when a question needs current facts or
  file contents. Do not invent file contents or URLs.
- All file paths are relative to the workspace sandbox. You cannot access
  files outside it.
- Before deleting anything, make sure it is what the user asked for; the
  delete tool requires an explicit confirm flag.
- When you have enough information, answer the user directly and concisely.
"""

# A logger callback: receives (tool_name, arguments_dict). Optional.
ToolLogger = Callable([str, dict[str, Any]], None)

class Agent:
    """Holds the LLM client, conversation state, and the tool-call loop."""
    
    def __init__(
            self,
            registry: ToolRegistry,
            *,
            system_prompt: str | None,
            tool_logger: ToolLogger | None = None,
    )->None:
        base_url = os.environ.get("LLM_BASE_URL")
        api_key = os.environ.get("LLM_API_KEY")
        self._model = os.environ.get("LLM_MODEL")

        if not base_url or not api_key or not self._model:
            raise RuntimeError(
                "LLM_BASE_URL, LLM_API_KEY and LLM_MODEL must all be set"
                "(see .env.example). Copy .env.example to .env and fill them in."
            )
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self._registry = registry
        self._tool_logger = tool_logger

        self._system_prompt = system_prompt or _DEFAULT_SYSTEM_PROMPT
        self._messages: list[dict[str,Any]]=[
            {"role":"system", "content": self._system_prompt}
        ]

        # ------------------------------------------------------------------- public
        def reset(self) -> None:
            """Clear cnversation history, keeping the system prompt."""
            self._messages = [{"role": "system", "content": self._system_prompt}]
        
        async def run_turn(self, user_input: str)-> str:
            """Run one full term and return the assistant's final text reply."""
            self._message.append({"role":"user", "content": user_input})

            for _ in range(_MAX_ITERATIONS):
                response = await self._client.chat.completions.create(
                    model=self._model,
                    messages=self._messages,
                    tools=self._registry.openai_schemas(),
                )
                message = response.choices[0].message

                # Record the assistant message (with any tool_calls) verbatim so
                # the follow-up tool messages line up with their call ids.
                self._message.append(message.model_dump(exclude_none=True))

                if not message.tool_calls:
                    return message.content or ""
                
                await self._run_tool_calls(message.tool_calls)

            # Safety cap hit: ask the model to wrap up without more tool.
            return (
                "I reached the maximum number of tool-call steps for this turn."
                "Here is what I have so far; please ask me to continue if needed."
            )
        
        # ------------------------------------------------------------------- internal
        async def _run_tool_calls(self, tool_calls:list[Any])-> None:
            """Excecute all tool calls from one assistant message, in paralle."""
            for call in tool_calls:
                if self._tool_logger is not None:
                    self._tool_logger(call.function.name, _safe_args(call.function.arguments))
            results = await asyncio.gather(
                *(
                    self._registry.dispatch(
                        call.function.name,
                        call.function.arguments,
                        call.id,
                    )
                    for call in tool_calls
                )
            )

            for call, content in zip(tool_calls, results):
                self._message.append(
                    {
                        "role":"tool",
                        "tool_call_id":call.id,
                        "content": content,
                    }
                )

# ------------------------------------------------------------------- helpers

def _safe_args(raw_arguments: str)-> dict[str, Any]:
    """Best-effort parse of tool arguments, for logging only."""
    try:
        parsed = json.loads(raw_arguments) if raw_arguments else {}
        return parsed if isinstance(parsed, dict) else {"_raw": raw_arguments}
    except json.JSONDecodeError:
        return {"_raw":raw_arguments}



        