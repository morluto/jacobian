# jacobian

A thin Node launcher and MCP client installer for
[Jacobian](https://github.com/morluto/jacobian) — the MCP server, CLI, and
Python library that exposes a portfolio of composable operations with
mathematically atomic, agent-visible outcomes to AI agents investigating
conjectures and other mathematical problems.

This package does not implement the kernel itself. It bootstraps the Python
distribution (`jacobian`) and provides commands to register
Jacobian with MCP clients, verify the handshake, and forward to the full CLI.
Agents compose operations into their own workflows; this launcher only
installs, registers, and forwards.

## Requirements

- Node.js >= 18
- Python 3.12/3.13 or `uv` on `$PATH` (the launcher installs the exactly matching
  kernel from PyPI on first use)

## Install

Guided user-local install:

```sh
curl -fsSL https://raw.githubusercontent.com/morluto/jacobian/main/npm/install.sh | sh
```

The guided installer resolves `latest` or `alpha` to an exact npm version,
installs the launcher under `~/.local/share/jacobian/npm-releases`, and
atomically activates `~/.local/bin/jacobian`. It disables npm lifecycle
scripts, refuses to replace an unmanaged command, and rolls activation back if
setup fails. If `uv` is absent, the installer can download the repository-pinned
official installer and verifies its SHA-256 digest before running it.

By default, the final doctor check installs and verifies the local mathematical
runtime. Its Python package environment is currently about 160 MB on Linux; a
uv-managed Python 3.12 adds about 110 MB when the machine does not already have
one. Pass `--defer-runtime` to leave those downloads until first use. Run
`npm/install.sh --help` for non-interactive client, release, dry-run, and
dependency options.

Manual global install:

```sh
npm install -g jacobian
```

## Usage

```sh
jacobian setup [--client <id>...] [--all] [--yes] [--dry-run] [--json] [--plain]
               [--source <checkout> --state-dir <path> --profile <name>]
  Configure MCP clients to use Jacobian.
jacobian upgrade
  Resolve the latest npm bootstrap, then refresh the launcher-managed Python package.
jacobian doctor [--client <id>...] [--all] [--json]
  Verify configured launchers, the MCP handshake, and the tool catalog.
jacobian remove [--client <id>...] [--all] [--yes] [--dry-run] [--json] [--plain]
  Remove Jacobian from MCP client configs.
jacobian mcp
  Run the Jacobian MCP server over stdio.
jacobian <command> [args...]
  Forward to the Python Jacobian CLI.
```

Supported clients: `claude`, `cursor`, `opencode`, `codex`, `gemini`.

Setup validates external runtime prerequisites and every selected client file
before writing. Its preflight discloses the launcher, exact target paths, and
the deferred Python environment, package-index access, and approximate install
size. Non-interactive setup requires explicit clients plus `--yes`; `--json`
reports contain only redacted public plan fields. Run the tailored `doctor`
command printed after setup to validate those client entries and execute the
configured launcher. For managed installations, doctor also requires the MCP
server to report the exact npm package version; a stale Python runtime fails
with a setup recovery action instead of being accepted by tool-name alone.

Setup writes only the selected client's MCP configuration. It does not install
prompts, skills, or a client-specific mathematical workflow.

`--source` writes a launcher bound to an absolute Jacobian checkout and uses
`uv run --project <checkout> --locked --no-sync jacobian-mcp`. Run the
checkout's `scripts/setup-agent` command for the complete locked dependency,
state initialization, source doctor, and client configuration workflow.

## Environment

- `JACOBIAN_STATE_DIR` — state directory (default: `./.jacobian`)
- `JACOBIAN_PACKAGE` — Python package spec override (default: the Python package
  version matching the installed npm launcher)
- `JACOBIAN_NPM_UPGRADE_HANDOFF` — internal one-time guard used while `upgrade`
  resolves `jacobian@latest`
- `JACOBIAN_DATA_DIR` — guided installer release data root
- `JACOBIAN_BIN_DIR` — guided installer directory for the stable command

## Verification model

Search and evaluation may be wrong. A result becomes verified only when an
operator-authorized checker accepts evidence bound to the exact claim,
semantics, candidate, and checker version. This launcher never promotes
evaluator output or solver status to a verified conclusion.

## License

MIT
