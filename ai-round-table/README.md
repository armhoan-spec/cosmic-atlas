# AI Round Table MCP

An MCP server that lets Claude delegate work to multiple AI providers and run them in a configurable sequence.

## What it does

- Exposes MCP tools: `list_agents`, `run_agent`, and `run_round`.
- Runs providers sequentially and passes prior outputs into later stages.
- Supports Anthropic Claude, Google Gemini, and OpenAI-compatible APIs (OpenAI, DeepSeek, and compatible gateways).
- Keeps API keys in environment variables; never commit secrets.

> This server does not control the Claude desktop/web UI. Claude connects to this local MCP server and invokes its tools.

## Requirements

- Python 3.10+
- API keys for the providers you want to enable

## Install

```bash
cd ai-round-table
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell:
# .venv\\Scripts\\Activate.ps1
pip install -e .
```

Copy `.env.example` to `.env` and set the keys you have. The server reads environment variables directly; load the .env file using your preferred environment manager.

## Configure Claude Desktop

Add this to Claude Desktop's MCP configuration file (adjust the absolute path and environment values):

```json
{
  "mcpServers": {
    "ai-round-table": {
      "command": "python",
      "args": ["/ABSOLUTE/PATH/TO/ai-round-table/server.py"],
      "env": {
        "ANTHROPIC_API_KEY": "YOUR_ANTHROPIC_KEY",
        "OPENAI_API_KEY": "YOUR_OPENAI_KEY",
        "GEMINI_API_KEY": "YOUR_GEMINI_KEY",
        "DEEPSEEK_API_KEY": "YOUR_DEEPSEEK_KEY"
      }
    }
  }
}
```

Restart Claude Desktop. The tools should appear in Claude's MCP tool list.

## Configure Claude Code

Add an MCP server using the absolute path to `server.py`. For example:

```bash
claude mcp add ai-round-table -- python /ABSOLUTE/PATH/TO/ai-round-table/server.py
```

Set provider keys in the environment inherited by Claude Code.

## Available tools

- `list_agents()`: shows enabled providers and the default sequence.
- `run_agent(agent, task, context?)`: sends one task to a selected provider.
- `run_round(task, agents?, context?)`: sends the task through the chosen providers in order. Each stage receives the original task and prior outputs.

Default sequence: Claude → ChatGPT → Gemini → DeepSeek. Only configured providers are enabled. You can specify an explicit sequence, e.g. `["openai", "gemini", "deepseek", "anthropic"]`.

## Provider environment variables

- Anthropic: `ANTHROPIC_API_KEY`, optional `ANTHROPIC_MODEL`
- OpenAI: `OPENAI_API_KEY`, optional `OPENAI_MODEL`
- Gemini: `GEMINI_API_KEY`, optional `GEMINI_MODEL`
- DeepSeek: `DEEPSEEK_API_KEY`, optional `DEEPSEEK_MODEL`

The OpenAI-compatible provider can also be customized with `OPENAI_BASE_URL` and `DEEPSEEK_BASE_URL`.

## Security

- Do not put real API keys in source code, issues, or commits.
- Treat model outputs as untrusted input; review generated code and commands before execution.
- Calls may incur provider charges.
- This is a local stdio MCP server; it does not expose a public network endpoint.
