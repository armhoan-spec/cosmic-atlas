"""AI Round Table MCP server: sequential delegation to configured model APIs."""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("ai-round-table")

DEFAULT_SEQUENCE = ["anthropic", "openai", "gemini", "deepseek"]
SYSTEM = (
    "You are one specialist in a multi-agent workflow. Work only on the assigned stage. "
    "Be precise, state assumptions, identify risks, and provide actionable output. "
    "Do not claim to have executed actions you did not execute."
)


def _enabled() -> list[str]:
    checks = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
    }
    return [name for name in DEFAULT_SEQUENCE if os.getenv(checks[name])]


def _post(url: str, headers: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Provider HTTP {exc.code}: {detail[:1200]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Provider connection error: {exc}") from exc


def _ask(provider: str, prompt: str) -> str:
    if provider == "anthropic":
        key = os.environ["ANTHROPIC_API_KEY"]
        model = os.getenv("ANTHROPIC_MODEL", "claude-3-7-sonnet-latest")
        data = _post(
            "https://api.anthropic.com/v1/messages",
            {"x-api-key": key, "anthropic-version": "2023-06-01"},
            {"model": model, "max_tokens": 4096, "system": SYSTEM,
             "messages": [{"role": "user", "content": prompt}]},
        )
        return "\n".join(x.get("text", "") for x in data.get("content", []) if x.get("type") == "text")

    if provider == "gemini":
        key = os.environ["GEMINI_API_KEY"]
        model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        data = _post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}",
            {},
            {"systemInstruction": {"parts": [{"text": SYSTEM}]},
             "contents": [{"role": "user", "parts": [{"text": prompt}]}],
             "generationConfig": {"maxOutputTokens": 4096}},
        )
        return "\n".join(
            p.get("text", "") for c in data.get("candidates", [])
            for p in c.get("content", {}).get("parts", [])
        )

    if provider in ("openai", "deepseek"):
        prefix = "OPENAI" if provider == "openai" else "DEEPSEEK"
        key = os.environ[f"{prefix}_API_KEY"]
        base = os.getenv(f"{prefix}_BASE_URL", "https://api.openai.com/v1" if provider == "openai" else "https://api.deepseek.com")
        model = os.getenv(f"{prefix}_MODEL", "gpt-4o-mini" if provider == "openai" else "deepseek-chat")
        data = _post(
            base.rstrip("/") + "/chat/completions",
            {"Authorization": f"Bearer {key}"},
            {"model": model, "messages": [{"role": "system", "content": SYSTEM},
                                           {"role": "user", "content": prompt}],
             "temperature": 0.3},
        )
        return data["choices"][0]["message"]["content"]

    raise ValueError(f"Unknown provider: {provider}")


@mcp.tool()
def list_agents() -> dict[str, Any]:
    """List supported AI providers, which are configured, and the default execution order."""
    enabled = _enabled()
    return {"supported": DEFAULT_SEQUENCE, "enabled": enabled,
            "default_sequence": [x for x in DEFAULT_SEQUENCE if x in enabled],
            "note": "Only providers with API keys configured are available."}


@mcp.tool()
def run_agent(agent: str, task: str, context: str = "") -> dict[str, Any]:
    """Send a task to one configured provider. agent must be anthropic, openai, gemini, or deepseek."""
    agent = agent.strip().lower()
    if agent not in _enabled():
        return {"ok": False, "agent": agent, "error": "Provider is not configured; add its API key."}
    prompt = f"TASK:\n{task}\n\nCONTEXT:\n{context or '(none)'}"
    try:
        return {"ok": True, "agent": agent, "output": _ask(agent, prompt)}
    except Exception as exc:
        return {"ok": False, "agent": agent, "error": str(exc)}


@mcp.tool()
def run_round(task: str, agents: list[str] | None = None, context: str = "") -> dict[str, Any]:
    """Run providers sequentially; every agent receives all previous outputs. Stops on provider failure."""
    enabled = _enabled()
    sequence = [x.strip().lower() for x in (agents or [a for a in DEFAULT_SEQUENCE if a in enabled])]
    if not sequence:
        return {"ok": False, "error": "No providers configured. Add at least one provider API key."}
    invalid = [a for a in sequence if a not in DEFAULT_SEQUENCE]
    missing = [a for a in sequence if a not in enabled]
    if invalid or missing:
        return {"ok": False, "error": "Invalid or unconfigured providers.", "invalid": invalid, "not_configured": missing}

    history: list[dict[str, str]] = []
    for index, agent in enumerate(sequence, start=1):
        prior = "\n\n".join(
            f"OUTPUT FROM {item['agent'].upper()}:\n{item['output']}" for item in history
        ) or "(No prior agent output.)"
        prompt = (
            f"Original user task:\n{task}\n\nAdditional context:\n{context or '(none)'}"
            f"\n\nPrevious agents' work (treat as suggestions, verify it):\n{prior}"
            f"\n\nYour stage: #{index} of {len(sequence)}. Add your specialist contribution, "
            "correct mistakes you notice, and clearly state unresolved questions."
        )
        try:
            output = _ask(agent, prompt)
            history.append({"agent": agent, "output": output})
        except Exception as exc:
            return {"ok": False, "completed": history, "failed_agent": agent, "error": str(exc)}

    return {"ok": True, "task": task, "sequence": sequence, "stages": history,
            "final_instruction": "Claude: synthesize the stages above, resolve disagreements, and present one final answer."}


if __name__ == "__main__":
    mcp.run(transport="stdio")
