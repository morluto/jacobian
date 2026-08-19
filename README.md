**English** · [简体中文](README.zh-CN.md)

<p align="center">
  <img src="docs/assets/jacobian-hero.jpg" width="100%" alt="An archival-style black-and-white photograph of a mathematician working at a chalkboard, with a constant Jacobian determinant and three distinct inputs mapping to one output.">
</p>

<h1 align="center">Jacobian</h1>

<p align="center">
  <strong>An executable mathematical vocabulary for agents: discover one typed operation, run it, and compose its result.</strong>
</p>

<p align="center">
  <a href="https://github.com/morluto/jacobian/actions/workflows/ci.yml"><img src="https://github.com/morluto/jacobian/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://pypi.org/project/jacobian/"><img src="https://img.shields.io/pypi/v/jacobian" alt="PyPI"></a>
  <a href="https://www.npmjs.com/package/jacobian"><img src="https://img.shields.io/npm/v/jacobian" alt="npm"></a>
  <a href="https://pypi.org/project/jacobian/"><img src="https://img.shields.io/pypi/pyversions/jacobian" alt="Supported Python versions"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/morluto/jacobian" alt="MIT license"></a>
</p>

Jacobian is an MCP server that gives AI agents a searchable vocabulary of typed
mathematical operations. `math.find` discovers an operation, and `math.run`
executes exactly one bounded mathematical contract and returns its typed
result. The same mathematical library is also available through a CLI and
native Python API.

Each operation establishes one stable, reusable mathematical postcondition
rather than prescribing a workflow or proof strategy. Results are exact where
claimed and make approximation, incompleteness, or uncertainty explicit.

**Jacobian's hypothesis is that mathematical reasoning benefits from an
executable vocabulary of semantically scoped, bounded operations.** Rather than
exposing large domain solvers or precomposed workflows, Jacobian exposes
mathematical primitives that agents can search for and compose into solutions
beyond what any individual operation was designed to solve. The library
supplies trustworthy mathematical moves; the reasoning model decides which
moves to make, how to combine their results, and when to stop. Keeping the
operations semantically narrow and domain-owned preserves that search space
instead of baking one proof strategy or workflow into the tools themselves.

## Quickstart

Run the canonical Python MCP command without installing Jacobian globally:

```sh
uvx --from jacobian jacobian-mcp
```

Where an MCP host requires an npm command, the npm package is a deterministic
carrier for that same command:

```sh
npx jacobian mcp
```

For a persistent installation:

```sh
python -m pip install jacobian
jacobian-mcp
```

That package includes Jacobian's exact maintained Python backend stack: SymPy,
NetworkX, Z3, and Python-FLINT. A normal Python or npm installation
therefore exposes the same built-in Python-backed operation portfolio. The
tested binary-install contract is CPython 3.12 or 3.13 on glibc Linux x86-64;
the release gate installs the built wheel and starts Jacobian on both Python
versions. Other systems may have compatible upstream wheels, but are not part
of the tested release contract yet. In particular, Alpine/musl cannot install
the complete mandatory stack from PyPI.

The Python distribution contains the mathematical kernel, CLI, and MCP server.
The npm package deterministically maps its exact package version to the
corresponding `uvx` invocation.

## Compute one bounded result

An ordinary operation returns mathematics first. For example,
`matrix.determinant.compute` accepts one exact rational matrix and returns its
determinant directly. Callers compose results by passing their typed values to a
subsequent operation.

## Available mathematics

The built-in portfolio covers work in:

- polynomial maps and polynomial algebra;
- exact linear algebra;
- graphs, paths, colorings, and isomorphism;
- bounded SAT and SMT solving;
- finite algebra, probability, geometry, and topology; and
- Lean source elaboration.

SAT and SMT operations use the maintained Z3 Python binding directly. The
optional `lean.check` operation runs one bounded source snippet in the fixed
Lean service environment, using a request-scoped temporary directory and
returning typed diagnostics. Use `math.find` to search for an operation, browse
an unfamiliar domain, and inspect one operation before calling `math.run` once.

See the [domain operation library](docs/reference/domain-operation-library.md)
for the maintained operation portfolio and
[backend requirements](docs/how-to/backend-requirements.md).

## Status

Jacobian 0.12.0 is pre-stable. Its published package and operation contracts
describe the supported surface; experimental operation contracts may change
between releases.

## Documentation

- [Documentation home](docs/index.md): tutorials, how-to guides, reference,
  and explanations
- [Architecture](docs/explanation/architecture.md): runtime structure and
  trust boundaries
- [Product model](docs/explanation/product-blueprint.md): operation contracts,
  ownership, and project boundaries
- [Tool reference](docs/reference/tools.md): MCP resources and invocation
  contracts
- [Backend requirements](docs/how-to/backend-requirements.md):
  maintained Python backends and optional Lean
- [Remote deployment](docs/how-to/deploy-remote-mcp.md): HTTP deployment and
  authentication

## Contributing

Jacobian uses Python 3.12, `uv`, and a small `Makefile`:

```sh
make setup
make test-math
make check
```

Read [CONTRIBUTING.md](CONTRIBUTING.md) before changing code. It documents
focused test commands, verification rules, documentation placement, and
pull-request expectations.

## License

[MIT](LICENSE)
