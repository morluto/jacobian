# Lean declaration discovery

[Documentation home](../../../index.md) · [Tool surface](../../tools.md)

Jacobian exposes two ordinary, read-only math tools over the installed, pinned
Lean runtime. Use them to query the formal environment directly instead of
shell-searching a local Mathlib checkout, source tree, or cache:

- `lean.declaration.search` performs bounded declaration retrieval; and
- `lean.declaration.inspect` resolves one exact declaration name.

They return **declaration metadata** (values). Finding or inspecting a theorem
does not check a new proof. Completed proof source must still be run through the
separate checker tool `lean.check`.

## Search contract

`lean.declaration.search` accepts:

- `environment`: `CORE` or `MATHLIB`;
- `name_contains`: an optional case-sensitive declaration-name substring;
- `type_pattern.constants`: an optional list of one to eight exact Lean
  constant names, all of which must occur in the elaborated declaration type;
- optional `namespace_prefixes` and declaration `kinds`; and
- `result_limit`, from 1 through 50.

At least one of `name_contains` or `type_pattern` is required. If both are
present, both must match. Type patterns inspect constants in Lean's elaborated
expression. They are not pretty-printed text matching, unification, typeclass
search, or proof search.

The provider excludes private declarations and visits public declaration names
in deterministic `Name.lt` order. Each result carries its elaborated
pretty-printed type, declaration kind, namespace when present, optional source
module and range, and explicit match reasons.

For example, a `MATHLIB` name search for `irrational_sqrt` returns candidates
such as `irrational_sqrt_two`; pass the exact returned name to inspection rather
than guessing namespace qualification.

The first name search in one backend session atomically materializes a compact
catalog of imported public declaration names, source modules, and kinds. Later
name searches use that catalog to select ordered candidates and their exact scan
positions. Lean resolves every candidate again in the pinned environment,
reapplies the filters, and checks elaborated type constants. Broad queries that
would exceed the bounded candidate handoff use the full Lean scan instead.

`stop_reason` separates the two possible coverage outcomes:

- `RESULT_LIMIT` means the result budget stopped the scan and completeness is
  `PARTIAL`; and
- `EXHAUSTED` means the deterministic scan exhausted the declared environment
  and filters, so completeness is `COMPLETE` with `COMPUTED` assurance.

An exhausted empty result is evidence that this exact scan found no match. It
is not a mathematical nonexistence conclusion.

## Inspect contract

`lean.declaration.inspect` accepts an environment and one exact
`declaration_name`. It returns the declaration's elaborated type, kind,
namespace, documentation and source metadata when available. A missing exact
name is an execution error, not an empty successful result. Exact inspection
uses Lean's environment lookup directly; it does not linearly scan the catalog.

## Environment identity and execution bounds

Both outputs carry `environment_digest`, `lean_version`, `lean_commit`, and
`mathlib_commit` (null for `CORE`). These caller-visible fields report the same
pinned runtime identity used to compute the declaration metadata, so a caller
does not need to invoke an unrelated proof-inspection tool for version data. The
`jacobian.lean.environment-manifest/v2` digest binds the selected import,
platform, pinned Lean version, executable provider digest, and a digest of the
measured semantic runtime. For `MATHLIB`, that runtime includes the Lake
launcher, project manifest and configuration, toolchain declaration, local
source modules, loaded `.olean` modules, and proof-state helper, in addition
to the authorized Mathlib commit. This is an exact runtime-manifest identity,
not an independent proof certificate.

The `CORE` profile exposes only imported `Init.*` declarations compatible with
the checker profile, even though the provider process also loads Lean
metaprogramming modules to implement the query. Provider-local helper
declarations are never searchable. `MATHLIB` exposes declarations imported by
the pinned `Mathlib` module.

The catalog is backend-local optimization state, not mathematical evidence. Its
header binds the exact `environment_digest`; Jacobian records its byte digest,
checks it before and after reuse, and discards the session if either identity
changes. Catalog creation uses an atomic rename, and candidate responses carry a
fresh request identity. A mismatch, partial index, stale response, or tampering
fails closed rather than falling back within the same operation.

Each bounded query still runs in a separate subprocess with `--trust=0` and one
worker. The per-query budget is 40 seconds for `CORE` and 105 seconds for
`MATHLIB`, with a two-MiB structured-output limit and a 128-KiB diagnostic
limit. Timeouts, unavailable profiles, malformed output, and Lean errors remain
execution failures without a mathematical conclusion.

See [Retrieve a Lean theorem and check a proof](../../../tutorials/lean-declaration-discovery.md)
for the public composition.

For dependency subgraphs, proof states, premise retrieval, statement
operations, and checker-bound proof edits, see the
[Lean formal intermediates reference](lean-formal-intermediates.md).
