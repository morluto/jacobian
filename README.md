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

<p align="center">
  <a href="README.md">English</a> ·
  <a href="README.zh-CN.md">简体中文</a>
</p>

Jacobian is a collection of mathematical operations for AI agents. It runs as
an MCP server and is also available as a CLI and Python library. Agents can use
it to compute invariants, search for examples or counterexamples, work with
solver artifacts, and check formal proofs.

## Quickstart

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

For a one-off setup without a persistent launcher:

```sh
npx jacobian setup
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

The launcher supports Claude, Codex, Cursor, Gemini, and OpenCode. It requires
Node.js 18 or newer plus Python 3.12/3.13 or
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

## A simple counterexample

Here is a small counterexample an agent can reason about directly:

```text
Claim: every prime is odd
Agent checks: 2 is prime and even
Counterexample: 2
Conclusion: the claim is false
```

Here `2` is a witness: the actual example that disproves the claim. For a much
larger search, Jacobian can preserve the candidate and the exact checks used to
establish it.

## A checked counterexample

Suppose an agent is testing the claim **“`F` is injective.”**

A search returns two points, `p` and `q`, with the same image. That is a
candidate counterexample, not yet a trusted conclusion.

```text
p ≠ q
F(p) - F(q) = 0
```

An independent checker confirms those relations exactly. The checked collision
can then be bound to the original claim and checker identity, producing
`FALSE · VERIFIED`.

If the search finds nothing, times out, is cancelled, or fails, the claim
remains `UNKNOWN`. Absence of a witness is not proof.

The [introductory tutorial](docs/tutorials/first-verified-result.md) shows the
same boundary in a runnable graph example.

## Available mathematics

The installed operations vary with local providers, but the maintained
portfolio covers work in:

- polynomial maps and polynomial algebra;
- exact linear algebra;
- graphs, paths, colorings, and isomorphism;
- SAT and SMT models and proof artifacts;
- finite and universal algebra;
- polytopes; and
- Lean declaration discovery and proof checking.

Some operations require optional local backends. Catalog membership means an
operation is installed and invocable; it does not grant verification authority.
Read `capability://catalog` or use `math.find` to inspect the current
environment. Use `math.run` to invoke a selected operation.

See the [domain operation library](docs/reference/domain-operation-library.md)
for the maintained operation portfolio and
[optional backend setup](docs/how-to/install-optional-backends.md) for provider
requirements.

## Verification model

Jacobian separates finding evidence from deciding what that evidence proves.
Search, generation, evaluation, and computation cannot certify their own
conclusions.

```text
Claim → Candidate → Independent check → Record
```

Only an operator-authorized checker may emit a verified record, bound to the
exact claim, candidate, scope, semantics, certificate format, and checker
identity. Plugins and search code cannot authorize a checker or change
verification policy.

> **No witness is not proof.** A failed search, timeout, cancellation, error,
> or completed bounded search without a witness leaves the claim `UNKNOWN`.

A formal claim may still be a poor translation of the informal conjecture.
Jacobian records that correspondence and its review status; schema validation
does not establish it automatically.

The [architecture document](docs/explanation/architecture.md) describes the
complete trust boundary.

## Status

Jacobian 0.6.0 is a pre-stable release. Its published package, capability, and
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
- [Optional backends](docs/how-to/install-optional-backends.md) — provider and
  Lean setup
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
