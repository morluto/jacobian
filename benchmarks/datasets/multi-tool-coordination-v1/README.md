# Multi-tool coordination v1

This versioned Harbor dataset is the evidence-driven PR1 pilot for cross-domain
mathematical coordination. Its four hand-auditable tasks were selected only
after the frozen 12-run observation in
`benchmarks/config/multi-tool-coordination-pr1-adjudication.json`.

| Family | Cases | Contract focus |
| --- | ---: | --- |
| Graph set distance | 1 | Derived set to exact all-vertex distances |
| Cycle lattice | 1 | Complex, spanning tree, coordinate matrix, determinant |
| Rational slice binding | 1 | Named scalar replay plus positive-definiteness certificate |
| Directed proportionality | 1 | Two normal forms with an explicit multiplier direction |

Every task has frozen offline input, a strict public submission schema, a
hidden Oracle solution, and a standard-library clean-room verifier. The
verifier scores the terminal mathematical object rather than proof prose. It
accepts alternate valid spanning trees, edge/facet orderings, and either matrix
orientation for the cycle-lattice case. It independently checks the exact
input and evidence bytes, certificate wrapper, scope, completeness, and
assurance. `VERIFIED` is not licensed in this pilot.

`generate.py` deterministically renders all task bundles and
`pilot-manifest.json`. PR1 is a first slice, not a calibrated or frozen
comparison set; PR2 owns expansion, real-model calibration, and the immutable
evaluation freeze before any product change.

```sh
uv run --locked python benchmarks/datasets/multi-tool-coordination-v1/generate.py --check
make harbor-check-task DATASET=multi-tool-coordination-v1 TASKS="coordination-graph-set-distance-01 coordination-cycle-lattice-01 coordination-rational-slice-01 coordination-directed-proportionality-01"
make harbor-oracle-task DATASET=multi-tool-coordination-v1 TASKS="coordination-graph-set-distance-01 coordination-cycle-lattice-01 coordination-rational-slice-01 coordination-directed-proportionality-01"
```
