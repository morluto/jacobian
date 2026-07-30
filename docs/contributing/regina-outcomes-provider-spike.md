# Regina low-dimensional-topology optional-provider spike

[Documentation home](../index.md)

- Status: provider and outcome-gate evidence; production deferred
- Frozen report contract: `jacobian.regina-outcomes-spike/v1`
- Registered capability IDs: none
- Provider: Regina wheel distribution 7.4.1, engine-reported version 7.4

## Decision

Regina is viable as an isolated producer for bounded 3-manifold
triangulations, algebraic invariants, normal surfaces, and exact recognition
algorithms. This spike does not register a capability. Every examined outcome
still needs a domain-owned contract or stronger independent evidence:

| Candidate outcome | Decision | Evidence obtained | Missing production evidence |
| --- | --- | --- | --- |
| 3-manifold triangulation materialization | `REVISE` | facet gluings are reciprocal and independently reproduce quotient face counts | versioned gluing semantics and an independent binding for canonical isomorphism signatures |
| \(H_1\) of a 3-manifold triangulation | `REVISE` | trivial, \(\mathbb Z/2\), and \(\mathbb Z\) groups reproduce | explicit cellular chain matrices and independent certified-Smith replay |
| embedded vertex normal-surface enumeration | `REVISE` | bounded primitive vectors and quadrilateral constraints replay | matching equations, vertex extremality, completeness, coordinate/list/algorithm scope |
| 3-sphere and 3-ball recognition | `RESEARCH_ONLY` | Regina's exact booleans reproduce on frozen cases | separate decision contracts and portable certificates; a provider boolean cannot establish `VERIFIED` |

Regina remains an operator-installed T1 provider. The upstream engine and
Python-wheel metadata declare GPL-2.0-or-later/GPLv2+. Redistribution,
third-party-library obligations, and operator approval are separate from
provider availability and checker authority. The CPython wheel contains no
license file beneath its `.dist-info` directory, so the spike binds both the
wheel metadata and the upstream source license; this is an observed packaging
fact, not a license exception.

Primary upstream boundaries are the
[Regina 7.4.1 release](https://regina-normal.github.io/),
[calculation-engine documentation](https://regina-normal.github.io/engine-docs/),
[`Triangulation<3>` API](https://regina-normal.github.io/engine-docs/classregina_1_1Triangulation_3_013_01_4.html),
[`NormalSurfaces` API](https://regina-normal.github.io/engine-docs/classregina_1_1NormalSurfaces.html),
[`AbelianGroup` API](https://regina-normal.github.io/engine-docs/classregina_1_1AbelianGroup.html),
and the
[Regina 7.4.1 PyPI files](https://pypi.org/project/regina/7.4.1/).

## Pinned provider identity

[`benchmarks/regina_outcomes_pin.json`](../../benchmarks/regina_outcomes_pin.json)
binds:

- the 7.4.1 upstream source archive and the upstream signed-checksum location;
- the upstream license, core, 3-dimensional triangulation, and normal-surface
  header digests;
- the CPython 3.12 manylinux wheel filename and SHA-256 digest;
- the wheel `METADATA` and `WHEEL` member digests and empty `.dist-info`
  license-file inventory;
- the exact adapter source;
- four isomorphism signatures and per-case tetrahedron bounds; and
- the complete expected mathematical output and digest.

The PyPI distribution version is `7.4.1`, while `regina.versionString()` returns
the engine's major/minor string `7.4`. The worker checks both values instead of
silently treating either as the other.

## Reproduction

Create a separate Python 3.12 environment and install only the pinned wheel:

```sh
uv venv /tmp/jcb-regina-venv --python 3.12
uv pip install \
  --python /tmp/jcb-regina-venv/bin/python \
  /path/to/regina-7.4.1-cp312-cp312-manylinux_2_28_x86_64.whl
```

Run the controller from the locked Jacobian environment:

```sh
uv run python benchmarks/regina_outcomes_spike.py \
  --python-executable /tmp/jcb-regina-venv/bin/python \
  --wheel /path/to/regina-7.4.1-cp312-cp312-manylinux_2_28_x86_64.whl \
  --source-archive /path/to/regina-7.4.1.tar.gz \
  --output /tmp/jcb-regina-outcomes-spike.json
```

The controller hashes and safely inspects the source archive and wheel before
launch. It preserves the selected virtual-environment launcher, runs the
adapter in a bounded process with a sanitized environment, limits stdout and
stderr, and strictly compares the mathematical output with the frozen digest.

The answer-visible cases are:

| Case | Isomorphism signature | Tetrahedra | f-vector | \(H_1\) | Recognition |
| --- | --- | ---: | --- | --- | --- |
| \(S^3\) | `cPcbbbaaa` | 2 | `(4, 6, 4, 2)` | `0` | sphere |
| \(L(2,1)\) | `cMcabbgqw` | 2 | `(1, 3, 4, 2)` | \(\mathbb Z/2\) | neither sphere nor ball |
| \(S^2\times S^1\) | `cMcabbjaj` | 2 | `(1, 3, 4, 2)` | \(\mathbb Z\) | neither sphere nor ball |
| \(B^3\) | `baa` | 1 | `(4, 6, 4, 1)` | `0` | ball |

For \(L(2,1)\), Regina also enumerates four embedded vertex normal surfaces in
standard `7n` coordinates. The observed provider-output digest on CPython
3.12.13 is
`sha256:20276b2bd3df9dc95b78cda265114862f9690280b702f00964c8cc80e8d4a8a4`.
The stable mathematical-output digest, which excludes the Python patch
version, is stored in the pin.

Missing files, source/wheel/adapter mismatch, malformed archives or wheel
metadata, wrong distribution/engine/Python version, timeout, cancellation,
process crash, output overflow, reproduction drift, and independent replay
disagreement are all non-conclusions. They register no capability and produce
no mathematical negation.

## Independent replay boundary

Without importing Regina, the controller:

1. validates every local tetrahedron, facet index, and vertex permutation;
2. checks that every internal facet gluing has the exact inverse gluing;
3. independently forms equivalence classes of local vertices, edges, and
   triangles and compares the quotient f-vector;
4. parses bounded nonnegative normal-coordinate vectors;
5. checks that every vector is primitive; and
6. checks the per-tetrahedron quadrilateral constraints.

This is intentionally `PARTIAL_MATCH`, not verification. It does not establish:

- that an isomorphism signature is canonical or bound to the only possible
  triangulation;
- that Regina's reported \(H_1\) arose from independently reconstructed
  boundary matrices;
- the normal matching equations, surface topology, vertex-ray extremality, or
  completeness of enumeration; or
- a portable proof certificate for either recognition result.

Same-provider round trips, known example names, or frozen expected answers
cannot close these obligations.

## Production contract gates

The first production prerequisite should be one atomic
3-manifold-triangulation artifact, not a generic `regina.call`. It must declare
tetrahedron and local-vertex bases, partial facet pairings, exact permutations,
boundary/ideal semantics, validity scope, orientation, connectedness, and
content identity. Complete cross-field validation must precede provider
execution or artifact writes.

A later homology outcome should expose the exact cellular boundary matrices
and their bases. It can then reuse the transformation-certified Smith
certificate format and an independent checker, while keeping Regina's
invariant factors as producer evidence only. This is adjacent to but not a
replacement for simplicial-complex integral homology: Regina triangulations
permit face identifications that are not finite abstract simplicial complexes.

A normal-surface outcome must choose exactly one coordinate system and list
class per capability. It must bound tetrahedra, coordinate digits, number of
surfaces, runtime, and output size; expose matching equations and algorithm
flags; distinguish complete enumeration from interruption; and retain vectors
already found after timeout only as incomplete evidence. An independent
checker must validate matching and admissibility and must not infer vertex or
fundamental completeness from the absence of more provider output.

Recognition outcomes must remain separate decisions such as 3-sphere or
3-ball recognition. Timeout, cancellation, error, or an unavailable provider
returns no conclusion. Until portable certificates and an independent checker
exist, a successful exact Regina decision can be `COMPUTED` at most and never
`VERIFIED`.

## Handoff

- Baseline tree: `59a51d2c59256a2e17bab811ec3be831d4b3cce4`.
- Candidate stage: discovery/provider gate complete; all production outcomes
  require revision.
- Provider state: absent from the locked Jacobian environment; Regina 7.4.1
  was installed only in the isolated reproduction venv.
- Public/held-out role: every case and answer is visible; no model evaluation,
  prompt, scorer run, or raw model trace exists.
- Raw report: `/tmp/jcb-regina-outcomes-spike.json` on the reproducing host.
  Its canonical pretty-printed report SHA-256 is
  `b5c9bdcce35a873b0d38e0443ca4a98dcad299b41cbc1678845f869e0fc53e05`.
- Validation run: the real isolated-provider reproduction; 7 focused
  fail-closed tests; the complete planner-selected unit, component, domain,
  composition, storage, process, MCP, and end-to-end lanes (1,707 passed and 5
  optional Lean skips); Ruff, formatting, dependency, dead-code, type,
  architecture, build, and documentation-link checks.
- Producer maximum: `COMPUTED`.
- Checker evidence: partial standard-library gluing and local normal-coordinate
  replay; no checker was authorized.
- Decision: retain the spike, register no capability, and begin with the
  domain-owned triangulation contract if a later batch accepts the GPL
  deployment boundary.
- Next action: design and adversarially test the triangulation/gluing artifact,
  then separately revisit cellular homology, normal enumeration, and each
  recognition decision.
