# Install native and formal providers

[Documentation home](../index.md)

Jacobian installs its maintained Python mathematical backends—SymPy, NetworkX,
Z3, Python-FLINT, and cvc5—as exact package dependencies. A missing or
mismatched Python backend is a broken installation, not a supported reduced
catalog. Backend availability is not verification authority: provider output
remains unverified until the appropriate independent checker accepts its bound
witness or certificate.

Native executables and formal runtimes remain optional operator-installed
components:

- CaDiCaL, which finds SAT models and UNSAT proof artifacts;
- Carcara, which independently checks Alethe proofs produced by cvc5; and
- pinned Lean `CORE` and `MATHLIB` environments, which check formal
  certificates.

Individual capabilities and independent replay support depend on the installed
catalog. The [provider runtime contract](../reference/provider-runtime.md)
defines how Jacobian measures provider availability, compatibility, identity,
and checker runtimes. The
[source setup profiles](setup-agent-from-source.md#profiles) provide maintained
installation paths for `core`, `lean`, and `external-proof` checkouts.

For ordinary contributor work, `make setup` installs and diagnoses
the complete locked Python backend surface; that is the contributor quick path
described in [CONTRIBUTING.md](../../CONTRIBUTING.md). CI owns the full Lean and
optional native-provider environments, so you
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
