# Install optional backends

[Documentation home](../index.md)

Some Jacobian capabilities use mathematical backends that are not installed by
default. Backend availability is not verification authority. Provider output
remains unverified until the appropriate independent checker accepts its bound
witness or certificate.

The optional providers currently include:

- CaDiCaL, which finds SAT models and UNSAT proof artifacts;
- cvc5, which produces SMT UNSAT proofs, with Carcara independently checking
  Alethe;
- the `flint` extra, which provides Python-FLINT/Arb operations for exact
  rational systems, integer matrices and lattices, polynomials, and validated
  numerical computation; and
- pinned Lean `CORE` and `MATHLIB` environments, which check formal
  certificates.

Individual capabilities and independent replay support depend on the installed
catalog. The [provider runtime contract](../reference/provider-runtime.md)
defines how Jacobian measures provider availability, compatibility, identity,
and checker runtimes. The
[source setup profiles](setup-agent-from-source.md#profiles) provide maintained
installation paths for `full-python`, `lean`, and `external-proof` checkouts.

For ordinary contributor work, `make setup PROFILE=core` installs the locked
development environment and diagnoses only the core provider surface
(NetworkX, SymPy, and Z3); that is the contributor quick path described in
[CONTRIBUTING.md](../../CONTRIBUTING.md). Use
`make setup PROFILE=full-python` when the change requires every maintained
Python extra. CI owns the full Lean and optional-provider environments, so you
do not need to prepare them locally unless you are reproducing a
boundary-specific failure.

## Lean certificates

The `lean.check` capability binds an exact proposition and proof body to its
result. The bundled environments pin Lean, imports, and their allowed trust
bases; model-supplied imports and packages are rejected.

Prepare the pinned runtime with:

```sh
elan toolchain install leanprover/lean4:v4.31.0
cd lean
lake update
lake build
```

Proof-state interaction and premise retrieval are exploration aids. Their
output cannot become `VERIFIED` without a successful `lean.check`. Continue
with the
[guided declaration-discovery tutorial](../tutorials/lean-declaration-discovery.md)
or consult the [Lean capability references](../reference/capabilities/lean/index.md).

## SAT and SMT proof providers

CaDiCaL finds SAT assignments and produces UNSAT proof artifacts. DRAT-trim can
serve as an independently authorized checker for the supported DRAT lane. Read
the [SAT artifact reference](../reference/capabilities/sat-smt/sat-artifacts.md)
for exact versions, provenance requirements, and replay contracts.

cvc5 produces Alethe proof artifacts for supported SMT profiles. Carcara is the
independent checker for the maintained strict replay path. Read the
[SMT artifact reference](../reference/capabilities/sat-smt/smt-artifacts.md)
before enabling that provider.
