"""Dependency-free MCP stdio server for sequential multi-provider AI delegation."""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any

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
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers}, method="POST",
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
        data = _post(
            "https://api.anthropic.com/v1/messages",
            {"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01"},
            {"model": os.getenv("ANTHROPIC_MODEL", "claude-3-7-sonnet-latest"),
             "max_tokens": 4096, "system": SYSTEM,
             "messages": [{"role": "user", "content": prompt}]},
        )
        return "\n".join(x.get("text", "") for x in data.get("content", []) if x.get("type") == "text")
    if provider == "gemini":
        key = os.environ["GEMINI_API_KEY"]
        model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        data = _post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}",
            {}, {"systemInstruction": {"parts": [{"text": SYSTEM}]},
                 "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                 "generationConfig": {"maxOutputTokens": 4096}},
        )
        return "\n".join(p.get("text", "") for c in data.get("candidates", [])
                          for p in c.get("content", {}).get("parts", []))
    if provider in ("openai", "deepseek"):
        prefix = "OPENAI" if provider == "openai" else "DEEPSEEK"
        base = os.getenv(f"{prefix}_BASE_URL",
                         "https://api.openai.com/v1" if provider == "openai" else "https://api.deepseek.com")
        model = os.getenv(f"{prefix}_MODEL", "gpt-4o-mini" if provider == "openai" else "deepseek-chat")
        data = _post(
            base.rstrip("/") + "/chat/completions",
            {"Authorization": f"Bearer {os.environ[f'{prefix}_API_KEY']}"},
            {"model": model, "messages": [{"role": "system", "content": SYSTEM},
                                           {"role": "user", "content": prompt}],
             "temperature": 0.3},
        )
        return data["choices"][0]["message"]["content"]
    raise ValueError(f"Unknown provider: {provider}")


def list_agents() -> dict[str, Any]:
    enabled = _enabled()
    return {"supported": DEFAULT_SEQUENCE, "enabled": enabled,
            "default_sequence": enabled,
            "note": "Only providers with API keys configured are available."}


def run_agent(agent: str, task: str, context: str = "") -> dict[str, Any]:
    agent = agent.strip().lower()
    if agent not in _enabled():
        return {"ok": False, "agent": agent, "error": "Provider is not configured; add its API key."}
    try:
        return {"ok": True, "agent": agent,
                "output": _ask(agent, f"TASK:\n{task}\n\nCONTEXT:\n{context or '(none)'}")}
    except Exception as exc:
        return {"ok": False, "agent": agent, "error": str(exc)}


def run_round(task: str, agents: list[str] | None = None, context: str = "") -> dict[str, Any]:
    enabled = _enabled()
    sequence = [x.strip().lower() for x in (agents or enabled)]
    invalid = [a for a in sequence if a not in DEFAULT_SEQUENCE]
    missing = [a for a in sequence if a not in enabled]
    if not sequence:
        return {"ok": False, "error": "No providers configured. Add at least one provider API key."}
    if invalid or missing:
        return {"ok": False, "error": "Invalid or unconfigured providers.",
                "invalid": invalid, "not_configured": missing}
    history: list[dict[str, str]] = []
    for index, agent in enumerate(sequence, start=1):
        prior = "\n\n".join(f"OUTPUT FROM {x['agent'].upper()}:\n{x['output']}" for x in history)
        prompt = (f"Original user task:\n{task}\n\nAdditional context:\n{context or '(none)'}"
                  f"\n\nPrevious agents' work (treat as suggestions, verify it):\n{prior or '(No prior agent output.)'}"
                  f"\n\nYour stage: #{index} of {len(sequence)}. Add your specialist contribution, "
                  "correct mistakes you notice, and clearly state unresolved questions.")
        try:
            history.append({"agent": agent, "output": _ask(agent, prompt)})
        except Exception as exc:
            return {"ok": False, "completed": history, "failed_agent": agent, "error": str(exc)}
    return {"ok": True, "task": task, "sequence": sequence, "stages": history,
            "final_instruction": "Claude: synthesize the stages above, resolve disagreements, and present one final answer."}


TOOLS = [
    {"name": "list_agents", "description": "List supported AI providers and configured providers.",
     "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False}},
    {"name": "run_agent", "description": "Send a task to one configured AI provider.",
     "inputSchema": {"type": "object", "properties": {
         "agent": {"type": "string", "enum": DEFAULT_SEQUENCE},
         "task": {"type": "string"}, "context": {"type": "string"}},
         "required": ["agent", "task"]}},
    {"name": "run_round", "description": "Run configured AI providers sequentially, passing earlier outputs to later providers.",
     "inputSchema": {"type": "object", "properties": {
         "task": {"type": "string"}, "agents": {"type": "array", "items": {"type": "string", "enum": DEFAULT_SEQUENCE}},
         "context": {"type": "string"}}, "required": ["task"]}},
]


def handle(msg: dict[str, Any]) -> dict[str, Any] | None:
    method, ident = msg.get("method"), msg.get("id")
    if method == "notifications/initialized" or method == "notifications/cancelled":
        return None
    if method == "initialize":
        return {"jsonrpc": "2.0", "id": ident, "result": {
            "protocolVersion": msg.get("params", {}).get("protocolVersion", "2024-11-05"),
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": "ai-round-table", "version": "0.1.2"}}}
    if method == "ping":
        result = {}
    elif method == "tools/list":
        result = {"tools": TOOLS}
    elif method == "tools/call":
        params = msg.get("params", {})
        name, args = params.get("name"), params.get("arguments", {})
        try:
            if name == "list_agents":
                value = list_agents()
            elif name == "run_agent":
                value = run_agent(**args)
            elif name == "run_round":
                value = run_round(**args)
            else:
                raise ValueError(f"Unknown tool: {name}")
            result = {"content": [{"type": "text", "text": json.dumps(value, ensure_ascii=False)}],
                      "isError": not value.get("ok", True)}
        except Exception as exc:
            result = {"content": [{"type": "text", "text": str(exc)}], "isError": True}
    elif method == "shutdown":
        result = {}
    else:
        if ident is None:
            return None
        return {"jsonrpc": "2.0", "id": ident,
                "error": {"code": -32601, "message": f"Method not found: {method}"}}
    return {"jsonrpc": "2.0", "id": ident, "result": result} if ident is not None else None


def main() -> None:
    for line in sys.stdin:
        try:
            message = json.loads(line)
            response = handle(message)
            if response is not None:
                sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
                sys.stdout.flush()
        except Exception as exc:
            sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": None,
                                         "error": {"code": -32603, "message": str(exc)}}) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
