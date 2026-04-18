"""Gemini client wrapper (google-genai). Used by agents; pipeline may not call this yet."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv(Path(__file__).resolve().with_name(".env"))

logger = logging.getLogger(__name__)

# Per project conventions: 30s HTTP timeout (milliseconds in HttpOptions).
_LLM_TIMEOUT_MS: int = 30_000


def _api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set; add it to Layer1/.env")
    return key


def _client() -> genai.Client:
    return genai.Client(
        api_key=_api_key(),
        http_options=types.HttpOptions(timeout=_LLM_TIMEOUT_MS),
    )


def _response_text(response: types.GenerateContentResponse) -> str:
    text = getattr(response, "text", None)
    if text is not None and text.strip():
        return text
    # Fallback: concatenate candidate parts
    parts: list[str] = []
    for c in response.candidates or []:
        content = c.content
        if content is None:
            continue
        for p in content.parts or []:
            if p.text:
                parts.append(p.text)
    return "".join(parts).strip()


def _grounding_sources(response: types.GenerateContentResponse) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for cand in response.candidates or []:
        meta = cand.grounding_metadata
        if meta is None or not meta.grounding_chunks:
            continue
        for chunk in meta.grounding_chunks:
            web = chunk.web
            if web is None:
                continue
            out.append(
                {
                    "title": web.title or "",
                    "url": web.uri,
                }
            )
    return out


def chat_json(
    prompt: str,
    model: str = "gemini-2.5-flash",
    temperature: float = 0.3,
) -> dict[str, Any]:
    """Call Gemini with JSON MIME type; parse body as a dict. Retries once on JSON decode errors."""
    client = _client()
    config = types.GenerateContentConfig(
        temperature=temperature,
        response_mime_type="application/json",
    )

    def _call(user_prompt: str) -> dict[str, Any]:
        response = client.models.generate_content(
            model=model,
            contents=user_prompt,
            config=config,
        )
        raw = _response_text(response)
        return json.loads(raw)

    try:
        return _call(prompt)
    except json.JSONDecodeError as first_err:
        logger.warning("chat_json: invalid JSON on first attempt: %s", first_err)
        retry_prompt = (
            f"{prompt}\n\n"
            "Your previous reply was not valid JSON. Return ONLY valid JSON, "
            "no markdown fences, no commentary."
        )
        try:
            return _call(retry_prompt)
        except json.JSONDecodeError as second_err:
            logger.exception("chat_json: invalid JSON after retry")
            raise second_err from first_err


def chat_text(
    prompt: str,
    model: str = "gemini-2.5-flash",
    temperature: float = 0.3,
) -> str:
    """Plain-text completion."""
    client = _client()
    config = types.GenerateContentConfig(temperature=temperature)
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )
    return _response_text(response)


def research_with_search(
    query: str,
    model: str = "gemini-2.5-flash",
) -> dict[str, Any]:
    """Grounded web research via Google Search tool. Returns model text plus raw source dicts."""
    client = _client()
    config = types.GenerateContentConfig(
        tools=[types.Tool(google_search=types.GoogleSearch())],
        temperature=0.3,
    )
    response = client.models.generate_content(
        model=model,
        contents=query,
        config=config,
    )
    text = _response_text(response)
    sources = _grounding_sources(response)
    return {"text": text, "sources": sources}
