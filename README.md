<p align="center">
  <a href="docs/explanation/hero-image.md"><img src="docs/assets/jacobian-hero.jpg" width="100%" alt="An archival-style black-and-white photograph of a mathematician working at a chalkboard, with a constant Jacobian determinant and three distinct inputs mapping to one output."></a>
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

An agent can do plenty of mathematics directly: choose a representation, spot a
useful theorem, simplify a small expression, or propose a proof. Jacobian is
for the part that benefits from an executable mathematical system and a clear
record of what happened: exact computation, finite search, solver certificates,
or formal proof checking.

It does not replace an agent's mathematical strategy or prescribe a workflow.
The agent decides what to investigate and which operation to use. Jacobian
exposes focused operations through a common interface, and keeps results
visible as typed values or durable artifacts.

## Why use a mathematical tool?

The model's weights provide mathematical intuition and strategy, but an answer
produced in conversation is not automatically a reproducible calculation or a
proof. Jacobian gives the agent a way to hand the brittle or exact part of a
problem to a maintained mathematical backend, then retain the result's scope
and evidence.

| An agent can do directly | Jacobian adds when it matters |
| --- | --- |
| Propose an approach, a candidate, or a proof idea | Execute exact algebra, bounded search, SAT/SMT solving, graph computation, or Lean checking |
| Explain why a result seems plausible | Record the inputs, result, scope, status, and provenance |
| Report that a solver or search succeeded | Independently check a witness, certificate, or formal proof for the exact claim |

The important boundary is that a successful computation is not automatically a
proof. Jacobian labels a result according to what has actually happened:

- **heuristic**: a plausible result from a model, search, or unchecked witness;
- **computed**: a deterministic calculation with a tested software contract;
- **verified**: evidence independently checked for the exact claim and scope.

A **witness** is a concrete object that establishes a claim. For example, `2`
is a witness for “there exists an even prime”; `2` is also a counterexample to
“every prime is odd.” Search may find such an object, but a separate checker
must establish that it really satisfies the stated property. Finding no witness
does not prove that none exists unless the search scope is complete and that
completeness is established.

## A simple counterexample

Here is a small counterexample an agent can reason about directly:

```text
Claim: every prime is odd
Agent checks: 2 is prime and even
Counterexample: 2
Conclusion: the claim is false
```

Here `2` is the **witness**: the actual example that disproves the claim. The
same idea applies when the concrete example is difficult to find or check, such
as a counterexample among millions of possible graphs. The agent still chooses
the claim and search strategy; Jacobian can run the exact search and save the
resulting graph and checks for later inspection.

In this documentation:

- **candidate** means an example not yet checked;
- **witness** means an example that establishes or disproves a claim;
- **verification** means checking that exact example really has the claimed
  property; and
- **artifact** means a saved mathematical object or piece of evidence.

## Jacobian, Lean, SAT, and the model

Jacobian supports Lean, SAT/SMT, computer algebra, and other mathematical
systems as backends. It does not replace them or compete with them.

| System | Main job |
| --- | --- |
| Model | Proposes ideas and chooses a mathematical strategy |
| CAS / SAT / SMT | Calculates or searches in a specialized domain |
| Lean | Checks a formal, general mathematical proof |
| Jacobian | Lets an agent discover and use those systems through one interface, while retaining typed results, scope, saved evidence, provenance, and verification status |

## A small example

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
Node.js 18 or newer plus Python 3.12 or
[`uv`](https://docs.astral.sh/uv/); the guided installer can install its pinned
`uv` release after confirmation. Run `jacobian mcp` to start the server
directly.

The Python distribution contains the mathematical kernel, CLI, and MCP server.
The npm package is a sub-100 KB thin launcher and MCP client installer for that
same implementation; it is not a separate JavaScript API. The npm tarball has
no npm runtime dependencies. The larger download is the local Python
mathematical runtime, not a JavaScript dependency tree.

To run the exact code in a clone, follow
[Configure an agent from a source checkout](docs/how-to/setup-agent-from-source.md).

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

The background to the repository artwork is documented in
[About the hero image](docs/explanation/hero-image.md).

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
