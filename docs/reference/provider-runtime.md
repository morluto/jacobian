# Provider runtime contract

Jacobian advertises a capability only when its exact provider runtime is
installed and passes its local health probe. Provider availability is a catalog
condition. An absent optional backend is not left for the agent to discover
during invocation.

This contract describes operational provenance. It does not change
mathematical assurance: an available provider, successful measurement, solver
status, or package digest is never evidence that a mathematical claim is
`VERIFIED`.

## Descriptor metadata

Every registered `CapabilityDescriptor` carries `provider_runtime`:

| Field | Meaning |
| --- | --- |
| `provider` | Stable provider family selected for this operation |
| `availability` | `AVAILABLE` for catalog entries; `UNAVAILABLE` is load-time state only |
| `version` | Exact installed provider version |
| `digest` | SHA-256 identity with the coverage declared by `digest_kind` |
| `digest_kind` | `SOURCE_TREE`, `PYTHON_DISTRIBUTION_RECORD`, `EXECUTABLE`, or `COMPOSITE` |
| `platform` | Current Python platform tag |
| `install_tier` | `T0`, `T1`, `T2`, or `T3` deployment rule |
| `license_id` and `license_files` | Declared license and installed license-file paths |
| `features` | Health-probed feature flags used by the adapter |
| `checker_ids` | Exact authorized checker identities, when the operation has fixed checkers |
| `configuration` | Canonical provider-specific data, such as Lean profiles |

`PYTHON_DISTRIBUTION_RECORD` hashes a canonical ordering of the installed
distribution's RECORD paths, recorded hashes, and sizes. It identifies the
installed manifest; the digest kind deliberately does not claim that Jacobian
rehashes every package byte at startup. The runtime identity retains the import
name and required feature attributes separately from operation configuration,
so unchanged checks replay the original health probe. `SOURCE_TREE` covers the
source package used by an entrypoint. `EXECUTABLE` covers the executable bytes.
`COMPOSITE` binds an ordered set of individually measured component runtimes,
for example Jacobian checker source plus the exact Python-FLINT distribution it
executes. Its canonical configuration exposes the component identities, and
resolution remeasures every component before accepting the aggregate digest.

The result repeats the selected `provider` and `provider_digest`. This binds the
invocation to the exact descriptor runtime without repeating all discovery
metadata in every response. Provider metadata remains separate from execution
status, conclusion, evidence type, and assurance.

Identity revalidation and first-use readiness are separate checks. The former
remeasures the declared executable, source tree, distribution RECORD, or every
component of a composite runtime and fails closed with a stable provider error
code when identity is incomplete, unavailable, malformed, or changed. The
latter imports a declared Python module and checks its required attributes (or
checks executable readiness) immediately before first use. A missing callable
is `READINESS_FAILED`; it does not change the import-free availability probe.

## Availability

`CapabilityService.register` rejects descriptors without runtime identity and
descriptors whose runtime is `UNAVAILABLE`. Built-in locked Python providers
must import, expose the required feature symbols, and have a hashed
distribution RECORD. The Lean probe validates the pinned Lean version and
commit, resolves and hashes the actual executable, and binds its semantic
environment: the Lean core library module plus, for Mathlib, the Lake launcher,
project manifest, Lake configuration, toolchain declaration, local source
modules, loaded `.olean` modules, and the proof-state helper. A missing or
changed component leaves the affected Lean runtime unavailable or fails its
later use closed.

CORE-only statement elaboration uses a separate frontend profile over the same
pinned executable. `lean.statement.propose` and `lean.statement.compare` are
registered only when that executable passes its version and commit probe; this
profile does not require a Mathlib checkout.

If the separately managed Lean runtime is absent or unhealthy, the runtime
still starts. Capabilities requiring the failed profile are absent from
`capability://catalog`, and no invocation is attempted. Explicit
operator-installed adapters fail registration instead of silently falling
back to another provider.

The optional cvc5 Alethe producer follows the same rule. The exact 1.3.4 wheel
must expose the required SMT-LIB parser and proof APIs and have a hashed RECORD
manifest. Otherwise `smt.unsat_proof.find` is absent while the base runtime and
SMT artifact schemas remain available.

Source-backed adapters can construct metadata without importing their
implementation:

```python
from jacobian.contracts.capabilities import CapabilityInstallTier
from jacobian.provider_runtime import source_provider_runtime

runtime = source_provider_runtime(
    "example.provider",
    version="1",
    entrypoint="example_adapter:create_adapter",
    install_tier=CapabilityInstallTier.T1,
    license_id="MIT",
    license_files=("LICENSE",),
    features=("exact-example-operation",),
)
```

The adapter places `runtime` in its descriptor. Registration remains
fail-closed if the source identity cannot be resolved.

### External checker runtimes

An independently authorized checker may itself depend on an external
executable or Python distribution. Its `CheckerRegistration` carries the
available `EXECUTABLE`, `PYTHON_DISTRIBUTION_RECORD`, or `COMPOSITE` provider
runtime. The runtime identity contributes to the checker ID, is persisted with
the authorization record, and is remeasured when the checker is selected and
before and after clean-process execution. The verification environment digest
also includes that runtime identity.

Some executables do not provide a trustworthy machine-readable version. Their
provider probe may require a strict operator-managed provenance sidecar that
binds the expected upstream release and source commit to the locally measured
executable digest. Such a sidecar records the operator's authorization basis;
it is not mathematical evidence or upstream attestation. A missing, malformed,
or digest-mismatched sidecar leaves the checker capability unavailable.

The pinned Lean runtime is measured under this rule when it is selected for an
independent checker. Its source/build identity and executable/runtime digest
are recorded with the checker authorization and remeasured around replay.

## Repeatable measurement

Measure the provider selected for one installed capability:

```sh
uv run jacobian --checker-authority NONE \
  provider-measure graph.compute.properties
```

The JSON result records:

- installed-size status and bytes when the footprint measurement completes;
- cold-start elapsed time and peak resident memory;
- a small provider-specific reproduction elapsed time and peak resident
  memory;
- cold-install status, elapsed time, and temporary installed bytes.

Cold install is skipped by default because it performs network and filesystem
work. Request it explicitly:

```sh
uv run jacobian --checker-authority NONE \
  provider-measure graph.compute.properties --include-cold-install
```

For Python-distribution providers, the cold-install probe uses a temporary
target and a fresh `uv` cache, installs the exact recorded version without
dependencies, and deletes the temporary files afterward. Source-tree and T3
providers without a safe automated installer report `SKIPPED` rather than
guessing an install source. Probes use bounded subprocess timeouts, bounded
output, sanitized environments, generic public errors, and temporary
directories.

Measurements are machine-local observations. Compare records only when the
provider digest, platform, probe contract, and environment are appropriate for
the decision being made.
