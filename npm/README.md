# jacobian

A thin deterministic npm carrier for
[Jacobian](https://github.com/morluto/jacobian) — the MCP server, CLI, and
Python library that exposes a portfolio of composable operations with
mathematically atomic, agent-visible outcomes to AI agents investigating
conjectures and other mathematical problems.

This package does not implement the kernel, install Python, manage
environments, edit MCP client configuration, or forward the full Python CLI.
It exists because the
[MCP Registry](https://github.com/modelcontextprotocol/registry) metadata
(`server.json`) publishes `npx jacobian mcp` as the installable carrier for the
stdio server. The carrier invokes exactly one canonical Python command and
nothing else.

## Requirements

- Node.js >= 18
- `uv` on `$PATH` (or set `JACOBIAN_UV_BIN`)

## Install

```sh
npm install -g jacobian
```

Install, upgrade, and remove this carrier with npm. Configure MCP clients with
the published command snippet; this package does not mutate client
configuration.

## Usage

```sh
jacobian mcp [args...]
  Run the canonical Jacobian MCP server over stdio.
```

`jacobian mcp` execs the exact canonical command:

```sh
uvx --from jacobian==<version> jacobian-mcp [args...]
```

The npm package version is the single release manifest. The carrier maps it to
the matching Python package spec and pins it for `uvx`, so a reproducible
deployment never floats `latest`. `uvx` owns the ephemeral Python environment;
this carrier does not.

## Environment

- `JACOBIAN_UV_BIN` — override the `uvx` executable used to launch the server.
- `JACOBIAN_PACKAGE` — override the pinned Python package spec (default:
  `jacobian==<version>` matching this carrier).

## Results

Each operation returns its own bounded mathematical outcome: an exact value,
an incomplete result, or an unknown/non-conclusion when its declared scope is
not enough. The carrier only launches Jacobian; it does not reinterpret or
upgrade an operation result.

## License

MIT
