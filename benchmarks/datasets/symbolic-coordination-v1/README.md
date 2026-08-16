# Symbolic coordination v1

This versioned Harbor dataset is the issue #477 PR1 contract and
hand-auditable polynomial-map pilot. Its 26 independently runnable cases test
terminal exact certificates without requiring Jacobian or prescribing a tool
sequence.

## Pilot families

| Family | Cases | Contract focus |
| --- | ---: | --- |
| Valid two-sided inverse | 5 | Both exact ordered compositions vanish |
| Perturbed near-miss | 4 | A plausible inverse has a nonzero residual |
| One-direction-only evidence | 3 | Supplied partial checking is not authoritative |
| Constant nonzero Jacobian | 4 | Keller condition is separated from global invertibility |
| Bounded collision scope | 6 | Witnesses, grid exhaustion, timeout, and incomplete search |
| Semantic equivalence | 4 | Renaming, reordering, duplicate terms, and cancellation |

Every task freezes one offline `input.json`, a strict family-specific
submission schema, a hidden Oracle solution, and a task-local clean-room
verifier. The public schema admits only the certificate kind licensed by that
task's claim family and frozen case type: two-sided composition replay for
inverse families, Keller determinant replay for constant-Jacobian cases, a
collision witness, complete grid exhaustion, or an honest search
non-conclusion. The verifier uses only Python's standard library and
independently replays rational polynomial normalization, both map
compositions, Jacobian determinants, finite grids, and collision witnesses.
It also binds the exact input, claim, map, subject, semantics, checker
identity, and certificate shape.

A constant nonzero Jacobian does not itself license a global-invertibility
claim. Grid exhaustion licenses only the declared finite scope, while
timeout, cancellation, incomplete work, and missing witnesses remain
non-conclusions.

## Deterministic identity

`generate.py` deterministically renders the 26 bundles from authored exact
fixtures. `pilot-manifest.json` binds the generator/case version, family,
fixture digest, and subject digest for every member. Regeneration is checked by:

```sh
uv run --locked python benchmarks/datasets/symbolic-coordination-v1/generate.py --check
```

An immutable evaluation snapshot is intentionally deferred until an operator
freezes the later comparison design; repository policy does not create a
snapshot merely for task authoring. Existing `mathematical-benchmarks-v1` snapshots are
unchanged.

## Validation

```sh
make harbor-prepare-task DATASET=symbolic-coordination-v1 TASKS="<task ids>"
make harbor-validate-task DATASET=symbolic-coordination-v1 TASKS="<task ids>"
make harbor-plan BASE=origin/main
make harbor-check
```

PR1 contains no comparison job, model run, post-solution audit, or training
contract. Product-surface observations that did not block
the pilot are recorded in [deferred operation gaps](OPERATION_GAPS.md).
