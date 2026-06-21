"""
Web tools backed by Tavily (https://tavily.com)
"""

from __future__ import annotations
import os
from typing import Any
from tavily import TavilyClient

from tools.registry import ToolRegistry

#Module level cache for the client so we don't rebuild it on every call.
_client: TavilyClient | None = None

def _get_client() -> TavilyClient:
    """
    Create once and return the Tavily client. Raises a clear error if the API 
    key is missing, so the message that reaches the LLM is actionable than a 
    generic error.
    """
    global _client
    if _client is None:
        api_key = os.environ.get('TAVILY_API_KEY')
        if not api_key:
            raise RuntimeError(
                "TAVILY_API_KEY is not set. Add it to your .env file." \
                "to enable web search."
            )
        _client = TavilyClient(api_key=api_key)
    return _client

def _web_search(args: dict[str,Any])-> str:
    """Run a Tavily search and format the results from the LLM."""
    query = args.get("query")
    if not query or not isinstance(query, str):
        raise ValueError("'query' is required and must be non-empty string.")
    
    #Clamp max results into a sane range; default 5
    max_results = args.get("max_results",5)
    if not isinstance(max_results, int):
        max_results = 5
    max_results = max(1, min(max_results, 10))

    response = _get_client().search(
        query=query,
        max_results=max_results,
        search_depth="basic"
    )
    results = response.get("results",[])
    if not results:
        return f"No results found for query: {query!r}"
    
    lines: list[str] = [f"Search results for {query!r}:\n"]
    for i, r in enumerate(results, start=1):
        title=r.get("title", "(no title)")
        url = r.get("url","")
        content=(r.get("content") or "").strip()
        lines.append(f"{i}. {title}\n {url}\n {content}\n")

    #Tavily can also synthesise a direct answer for some queries
    answer = response.get("answer")
    if answer:
        lines.append(f"\nTavily summary: {answer}")

    return "\n".join(lines)

def _web_fetch(args: dict[str,Any])-> str:
    """Fetch and return the clean text content of a specific URL via Tavily Extract."""
    url = args.get("url")
    if not url or not isinstance(url, str):
        raise ValueError("'url' is required and must be non-empty string.")
    response = _get_client().extract(urls=[url])
    results = response.get("results",[])
    if not results:
        failed = response.get("failed_results",[])
        return f"Could not fetch {url!r}. {failed or ''}".strip()
    content = (results[0].get("raw_content") or "").strip()
    if not content:
        return f"Fetched {url!r} but it had no extractable text content."
    #Gaurd against dumping an enormous page into the model's context.
    max_chars = 8000
    if len(content)>max_chars:
        content = content[:max_chars] + f"\n\n[...truncated at {max_chars} characters...]"
    return f"Content of {url}:\n\n{content}"

#JSON schema describing the tool's argument (OpenAI function-parameters shape.)
_SEARCH_PARAMETERS: dict[str, Any] = {
    "type":"object",
    "properties":{
        "query":{
            "type":"string",
            "description":"The search query, phrased as you would type it into a search engine.",
        },
        "max_results":{
            "type":"integer",
            "description":"How many results to return (1-10). Default to 5",
            "minimum":1,
            "maximum":10
        },
    },
    "required":["query"],
}

_FETCH_PARAMETERS: dict[str, Any] = {
    "type":"object",
    "properties":{
        "url":{
            "type":"string",
            "description":"The full URL of the page to fetch and read (must start with http:// or https://).",
        },
    },
    "required":["url"],
}

def register(registry: ToolRegistry) -> None:
    """Register the web search and web fetch tools into the given registry."""
    registry.register_sync(
        name="web_search",
        description=(
            "Search the web for current information. Use this when you need" 
            "up-to-date facts, news, documentation, or anything you are unsure" 
            "about. Returns a list of results titles, URLs, and content snippets."
        ),
        parameters= _SEARCH_PARAMETERS,
        handler = _web_search,
    )
    registry.register_sync(
        name="web_fetch",
        description=(
            "Fetch the full text content of a specific web page by URL. Use this" 
            "after web_search when a result's snippet is not enough and you need" 
            "to read the whole page. Takes a single 'url' argument."
        ),
        parameters=_FETCH_PARAMETERS,
        handler = _web_fetch,
    )

        
