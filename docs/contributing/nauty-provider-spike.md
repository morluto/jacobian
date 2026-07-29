# nauty/Traces optional-provider spike

This spike decides whether nauty/Traces is a viable optional provider for
atomic graph canonicalization or non-isomorphic graph generation. It does not
register a capability, authorize a checker, or promote provider output beyond
an observed computation.

## Decision

nauty/Traces 2.9.3 passes the bounded provider and reproduction probes, but
production capabilities remain deferred.

- Deployment is an operator-installed T2 external provider. Jacobian should not
  vendor the C sources or discover executables at import time.
- Version 2.9.3 is licensed under Apache-2.0. The older custom restrictions
  mentioned in pre-2.6 history do not govern this release.
- A future adapter must require explicit executable paths and an
  operator-managed provenance record binding the official source release to
  the measured `geng` and `labelg` executable digests. Neither executable has a
  machine-readable version command sufficient for the runtime contract.
- The spike's success is `OBSERVED_PROVIDER_BEHAVIOR`, not mathematical
  assurance. It adds no catalog entries.

The version, source digest, license text, command profiles, and exact expected
outputs are frozen in
[`benchmarks/nauty_provider_pin.json`](../../benchmarks/nauty_provider_pin.json).
The source and command semantics were checked against the
[official nauty/Traces page](https://users.cecs.anu.edu.au/~bdm/nauty/) and the
[2.9.3 user's guide](https://users.cecs.anu.edu.au/~bdm/nauty/nug29.pdf).

## Bounded reproduction

Build the official source archive outside the repository, then run:

```sh
uv run python benchmarks/nauty_provider_spike.py \
  --source-archive /path/to/nauty2_9_3.tar.gz \
  --geng /path/to/nauty2_9_3/geng \
  --labelg /path/to/nauty2_9_3/labelg \
  --output /tmp/nauty-provider-spike.json
```

The probe:

1. hashes the exact official source archive and inspects its bounded
   `COPYRIGHT` and `gtools.h` members without extracting them;
2. resolves only the explicitly supplied executables and records their SHA-256
   digests;
3. checks the pinned `-help` feature surfaces under a sanitized environment and
   bounded wall time/output;
4. runs `geng -q 4` and requires the exact 11 graph6 representatives for all
   simple undirected four-vertex graphs; and
5. runs two differently labelled copies of the four-vertex path through
   `labelg -q` and requires both to produce the frozen canonical graph6 bytes.

The report records the exact command profiles, source and executable identities,
bounded output digests, scope, observed count, limitations, and checker
obligations. Missing files, source-version mismatch, malformed output, timeout,
cancellation, output overflow, and process failure are explicit
non-conclusions.

## Production gates

### Canonical graph labelling

Status: `REVISE`.

The candidate artifact must bind the input graph semantics, output canonical
bytes, and any vertex permutation. `labelg` emits canonical graph bytes but
does not expose the permutation needed for that stronger relationship, so a
thin C adapter may be necessary.

An independent isomorphism checker can establish that the input and output are
isomorphic. It does not establish that the output is the selected canonical
minimum. A production capability therefore needs a separately specified
canonical-order obligation and an operator-authorized checker independent of
the nauty search implementation. The provider cannot authorize itself.

### Non-isomorphic graph generation

Status: `REVISE`.

Pairwise independent isomorphism replay can reject duplicate classes in a
bounded page. It cannot prove that no class is missing. Any production
enumeration must expose:

- the exact graph class and order;
- every `geng` filter and format option;
- `res/mod` partition parameters;
- page boundaries and an explicit truncation/stop reason;
- whether the result claims only a sample/page or exhaustive coverage; and
- coverage evidence independent of pairwise non-isomorphism.

Timeout, cancellation, crash, output overflow, partial partitions, and failure
to find a graph remain non-conclusions. A `COMPLETED` process alone cannot
establish exhaustive completeness.

## Absence isolation

The spike module is not imported by runtime assembly, portfolio construction,
the CLI, or MCP startup. Tests run it against an absent explicit provider and
compare the complete catalog before and after. Since no nauty capability IDs
exist, absence removes no existing entry and unrelated runtime startup remains
unchanged.

If a production adapter is proposed later, it must add installed, absent,
version-mismatch, malformed-output, timeout, crash, catalog-isolation, CLI/MCP
startup, and advertised-package-version tests before registration.
