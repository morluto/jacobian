<p align="center">
  <img src="docs/assets/jacobian-hero.jpg" width="100%" alt="An archival-style black-and-white photograph of a mathematician working at a chalkboard, with a constant Jacobian determinant and three distinct inputs mapping to one output.">
</p>

<h1 align="center">Jacobian</h1>

<p align="center">
  <strong>Pure mathematics for agents: search for examples and counterexamples, compute exactly, and independently check what a result proves.</strong>
</p>

<p align="center">
  <a href="https://github.com/morluto/jacobian/actions/workflows/ci.yml"><img src="https://github.com/morluto/jacobian/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://pypi.org/project/jacobian/"><img src="https://img.shields.io/pypi/v/jacobian" alt="PyPI"></a>
  <a href="https://www.npmjs.com/package/jacobian"><img src="https://img.shields.io/npm/v/jacobian" alt="npm"></a>
  <a href="https://pypi.org/project/jacobian/"><img src="https://img.shields.io/pypi/pyversions/jacobian" alt="Supported Python versions"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/morluto/jacobian" alt="MIT license"></a>
</p>

Jacobian is a collection of mathematical operations for AI agents. It runs as
an MCP server and is also available as a CLI and Python library. Agents can use
it to compute invariants, search for examples or counterexamples, work with
solver artifacts, and check formal proofs.

## Quickstart

For a one-time setup:

```sh
npx jacobian setup
```

For a guided user-local install:

```sh
curl -fsSL https://raw.githubusercontent.com/morluto/jacobian/main/npm/install.sh | sh
```

The installer resolves an npm release to an exact version, installs the small
launcher without lifecycle scripts, configures selected MCP clients, and
verifies the local server. The Python package environment is approximately 160
MB; if Python 3.12 is not already available, uv's managed Python adds about 110
MB. Add `--defer-runtime` to postpone both until first use:

```sh
curl -fsSL https://raw.githubusercontent.com/morluto/jacobian/main/npm/install.sh | \
  sh -s -- --client codex --yes --defer-runtime
```

For repeated use:

```sh
npm install -g jacobian
jacobian setup
jacobian upgrade
jacobian doctor
```

For the Python distribution:

```sh
python -m pip install jacobian
```

That package includes Jacobian's exact maintained Python backend stack: SymPy,
NetworkX, Z3, Python-FLINT, and cvc5. A normal Python or npm installation
therefore exposes the same built-in Python-backed operation portfolio. The
tested binary-install contract is CPython 3.12 or 3.13 on glibc Linux x86-64;
the release gate installs the built wheel and starts Jacobian on both Python
versions. Other systems may have compatible upstream wheels, but are not part
of the tested release contract yet. In particular, Alpine/musl cannot install
the complete mandatory stack from PyPI.

The launcher supports Claude, Codex, Cursor, Gemini, and OpenCode. It requires
Node.js 18 or newer plus CPython 3.12/3.13 or
[`uv`](https://docs.astral.sh/uv/); the guided installer can install its pinned
`uv` release after confirmation. Run `jacobian mcp` to start the server
directly.

The Python distribution contains the mathematical kernel, CLI, and MCP server.
The npm package is a sub-100 KB thin launcher and MCP client installer for that
same implementation; it is not a separate JavaScript API. It bundles one TOML
parser for fail-closed Codex configuration updates and runs no install-time
scripts. The larger download is the local Python mathematical runtime, not a
JavaScript dependency tree.

To run the exact code in a clone, follow
[Configure an agent from a source checkout](docs/how-to/setup-agent-from-source.md).

## Compute, then check when needed

An ordinary operation returns mathematics first. For example,
`matrix.determinant.compute` accepts one exact rational matrix and returns its
determinant inline. If independent replay matters, the agent may separately run
`matrix.determinant.verify` with that exact input and candidate result.

The producer and checker are distinct catalog IDs with independent
implementations. Computation does not certify itself, and a timeout,
cancellation, error, or incomplete bounded search remains a non-conclusion.

The [introductory tutorial](docs/tutorials/first-verified-result.md) runs this
determinant pair through the public MCP surface.

## Available mathematics

The installed operations vary with optional external providers, but the
maintained portfolio covers work in:

- polynomial maps and polynomial algebra;
- exact linear algebra;
- graphs, paths, colorings, and isomorphism;
- SAT and SMT models and proof artifacts;
- finite and universal algebra;
- polytopes; and
- Lean declaration discovery and proof checking.

Some operations require optional native or formal backends. Catalog membership
means an operation is installed and invocable; it does not grant verification
authority.
Read `capability://catalog` or use `math.find` to inspect the current
environment. Use `math.run` to invoke a selected operation.

See the [domain operation library](docs/reference/domain-operation-library.md)
for the maintained operation portfolio and
[native and formal provider setup](docs/how-to/install-native-and-formal-providers.md)
for provider requirements.

## Verification model

Jacobian separates mathematical production from independent checking. A
producer cannot certify its own output.

```text
Subject + Candidate → Independent checker → Bound record
```

Only an operator-authorized checker may emit a verified record, bound to the
exact subject, candidate, evidence, protocol, scope, semantics, certificate
format, and checker identity. Availability and provider provenance do not grant
that authority.

> **No witness is not proof.** A failed search, timeout, cancellation, error,
> or completed bounded search without a witness leaves the claim `UNKNOWN`.

The [architecture document](docs/explanation/architecture.md) describes the
complete trust boundary.

## Status

Jacobian 0.11.0 is a pre-stable release. Its published package, capability, and
artifact contracts describe the current supported surface; ongoing capability
research may change experimental contracts between releases.

## Documentation

- [Documentation home](docs/index.md) — tutorials, how-to guides, reference,
  and explanations
- [First verified result](docs/tutorials/first-verified-result.md) — a complete
  runnable example
- [Architecture](docs/explanation/architecture.md) — runtime structure and
  trust boundaries
- [Product model](docs/explanation/product-blueprint.md) — capability contracts,
  ownership, and project boundaries
- [Tool reference](docs/reference/tools.md) — MCP resources and invocation
  contracts
- [Native and formal providers](docs/how-to/install-native-and-formal-providers.md)
  — provider and Lean setup
- [Remote deployment](docs/how-to/deploy-remote-mcp.md) — HTTP deployment and
  authentication

## Contributing

Jacobian uses Python 3.12, `uv`, and a small `Makefile`:

```sh
make setup
make test-unit
make check
```

Read [CONTRIBUTING.md](CONTRIBUTING.md) before changing code. It documents
focused test commands, verification rules, documentation placement, and
pull-request expectations.

## License

[MIT](LICENSE)
