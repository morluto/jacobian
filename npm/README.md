# jacobian

A thin Node launcher and MCP client installer for
[Jacobian](https://github.com/morluto/jacobian) — the MCP server, CLI, and
Python library that exposes a portfolio of composable capabilities with
mathematically atomic, agent-visible outcomes to AI agents investigating
conjectures and other mathematical problems.

This package does not implement the kernel itself. It bootstraps the Python
distribution (`jacobian`) and provides commands to register
Jacobian with MCP clients, verify the handshake, and forward to the full CLI.
Agents compose capabilities into their own workflows; this launcher only
installs, registers, and forwards.

## Requirements

- Node.js >= 18
- Python 3.12 and `uv` on `$PATH` (the launcher installs the matching stable
  kernel from PyPI on first use)

## Install

```sh
npm install -g jacobian
```

## Usage

```sh
jacobian setup [--client <id>...] [--all] [--yes] [--dry-run] [--json]
               [--source <checkout> --state-dir <path> --profile <name>]
  Configure MCP clients to use Jacobian.
jacobian upgrade
  Refresh the launcher-managed Python package.
jacobian doctor [--json]
  Verify the MCP handshake and tool catalog.
jacobian remove [--client <id>...] [--all] [--yes] [--json]
  Remove Jacobian from MCP client configs.
jacobian mcp
  Run the Jacobian MCP server over stdio.
jacobian <command> [args...]
  Forward to the Python Jacobian CLI.
```

Supported clients: `claude`, `cursor`, `opencode`, `codex`, `gemini`.

Codex setup also installs a small `jacobian-math` skill under
`~/.codex/skills`. Codex does not guarantee that MCP server instructions are
always visible before tool selection; the skill makes relevant mathematical
requests surface Jacobian without prescribing a mathematical workflow. Setup
refuses to overwrite a same-named unmanaged or modified skill, and removal
deletes only the exact managed content.

`--source` writes a launcher bound to an absolute Jacobian checkout and uses
`uv run --project <checkout> --locked --no-sync jacobian-mcp`. Run the
checkout's `scripts/setup-agent` command for the complete locked dependency,
state initialization, source doctor, and client configuration workflow.

## Environment

- `JACOBIAN_STATE_DIR` — state directory (default: `./.jacobian`)
- `JACOBIAN_PACKAGE` — Python package spec override (default: the Python package
  version matching the installed npm launcher)

## Verification model

Search and evaluation may be wrong. A result becomes verified only when an
operator-authorized checker accepts evidence bound to the exact claim,
semantics, candidate, and checker version. This launcher never promotes
evaluator output or solver status to a verified conclusion.

## License

MIT
