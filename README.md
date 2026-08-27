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

See [Executable mathematical vocabulary](docs/explanation/executable-mathematical-vocabulary.md)
for what semantic atomicity means and how the operation vocabulary grows.

## Quickstart

Set up Jacobian for your agents with a single command. The setup command
requires Node.js 20.17+, 22.13+, or 23.5+ and `uvx` on your `PATH`.

```sh
npx jacobian@latest setup
```

Choose detected agents and review the changes before they are written. Setup
does not install Node.js, Python, `uv`, or an agent. For automation, preview
an explicit plan with `npx jacobian@latest setup --codex --dry-run`; use
`--yes` only with explicit agent flags or `--all`.

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

For a local terminal workflow, inspect the exact installed contract and run one
of its examples with the CLI:

```sh
jacobian inspect integer.compute.extended_gcd
jacobian run integer.compute.extended_gcd --json '{"left":"84","right":"30"}'
```

The second command returns the gcd and Bézout coefficients as JSON. In an MCP
host, use `math.find` to inspect the same contract and `math.run` with the same
payload shape. See [Discover and invoke operations](docs/how-to/invoke-domain-operations.md)
for that agent workflow.

## Available mathematics

The built-in portfolio covers work in:

- polynomial maps and polynomial algebra;
- exact linear algebra;
- graphs, paths, colorings, and isomorphism;
- bounded SAT and SMT solving;
- finite algebra, probability, geometry, and topology.

SAT and SMT operations use the maintained Z3 Python binding directly. Use
`math.find` to search for an operation, browse an unfamiliar domain, and inspect
one operation before calling `math.run` once.

See the [domain operation library](docs/reference/domain-operation-library.md)
for the maintained operation portfolio and
[backend requirements](docs/how-to/backend-requirements.md).

## Status

Jacobian 0.15.0 <!-- x-release-please-version --> is pre-stable. Its published package and operation contracts
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
  maintained Python backends
- [Remote deployment](docs/how-to/deploy-remote-mcp.md): HTTP deployment and
  authentication

## Contributing

Jacobian uses Python 3.12, `uv`, and a small `Makefile`:

```sh
make setup
make handoff LANE=math TESTS=tests/math/graphs/test_graph_distance_matrix.py
```

Read [CONTRIBUTING.md](CONTRIBUTING.md) before changing code. It documents
focused test commands, verification rules, documentation placement, and
pull-request expectations.

## License

[MIT](LICENSE)
