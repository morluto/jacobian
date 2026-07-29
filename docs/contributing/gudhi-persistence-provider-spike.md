# GUDHI persistent-homology optional-provider spike

This spike tests GUDHI 3.13.0 as a producer for one future atomic outcome:
persistent homology of a bounded filtered finite simplicial complex over a
prime field. It does not register a capability or authorize GUDHI as a
checker.

## Decision

The bounded provider reproduction and an independent modular reduction agree;
a production capability remains `REVISE`.

- The CPython API stores and returns filtration values as `float`. The adapter
  therefore sends only unique integer ranks to GUDHI. Exact rational
  birth/death values are rehydrated from bound input simplex IDs, never from a
  provider float.
- The selected `Simplex_tree` and `Persistent_cohomology` source modules both
  declare MIT, and the CPython 3.12 wheel contains an MIT license. This selected
  slice is license-compatible, but remains an operator-installed T1 optional
  provider rather than a core dependency.
- The worker returns birth and death simplex IDs, ranks, and a typed
  `INFINITE` sentinel. The outer spike independently reduces the mod-2 boundary
  matrix using only the standard library and records every reduced column.
- A provider result remains `COMPUTED` at most. The demonstrated replay is
  evidence that an independent checker is feasible, not operator authorization
  of the spike as that checker.

The [GUDHI 3.13.0 SimplexTree reference][simplex-tree], [GUDHI licensing
inventory][licensing], [3.13.0 source tag][source-tag], and [PyPI 3.13.0
files][pypi] define the upstream boundary. The frozen source and wheel digests,
module-header digests, exact filtered complex, and expected mathematical output
are in
[`benchmarks/gudhi_persistence_pin.json`](../../benchmarks/gudhi_persistence_pin.json).

[simplex-tree]: https://gudhi.inria.fr/python/3.13.0/simplex_tree_ref.html
[licensing]: https://gudhi.inria.fr/licensing/
[source-tag]: https://github.com/GUDHI/gudhi-devel/tree/tags/gudhi-release-3.13.0
[pypi]: https://pypi.org/project/gudhi/3.13.0/

## Reproduce

Create an isolated Python 3.12 environment and install the pinned wheel plus a
compatible NumPy:

```sh
uv venv /tmp/jcb-gudhi-venv --python 3.12
uv pip install --python /tmp/jcb-gudhi-venv/bin/python \
  /path/to/gudhi-3.13.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl \
  numpy

uv run python benchmarks/gudhi_persistence_spike.py \
  --python-executable /tmp/jcb-gudhi-venv/bin/python \
  --wheel /path/to/gudhi-3.13.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl \
  --source-archive /path/to/gudhi-3.13.0-source.tar.gz \
  --output /tmp/jcb-gudhi-provider-spike.json
```

The runner hashes and safely inspects the source archive and wheel before
launch. It binds the checked-in adapter source, preserves the selected virtual
environment launcher rather than resolving through its symlink, and executes a
bounded worker with a sanitized environment.

The frozen complex has eleven simplices. Four vertices and four boundary edges
create a square cycle, a diagonal creates a second transient cycle, and two
triangles fill them. Its eleven exact filtration values include non-binary
rationals. GUDHI receives ranks 0 through 10, returns six persistence pairs,
and never receives those rational values. The independent replay agrees on four
dimension-0 pairs and two dimension-1 pairs, including one essential connected
component represented by `{"kind": "INFINITE"}`.

Missing files, source/wheel/adapter mismatch, malformed archives or output,
wrong GUDHI or Python version, timeout, cancellation, output overflow, process
crash, reproduction drift, and independent-replay disagreement are all
non-conclusions.

## Production contract gate

A future `topology.filtered_complex.persistent_homology.compute` request must
bind:

- a canonical filtered-simplicial-complex artifact with stable simplex IDs;
- exact rational filtration values and validated face monotonicity;
- a prime coefficient field, requested homology dimensions, and size bounds;
- a canonical total tie order in which every face precedes its cofaces;
- the provider runtime identity and exact rank transport; and
- execution status, mathematical conclusion, completeness, assurance, and
  open obligations as separate fields.

Its result should expose birth/death simplex IDs, exact values rehydrated from
the input, finite versus essential interval type, provider pairing evidence,
and a relationship to the source filtered complex. Pairing and reduced-column
ledgers should be first-class artifacts rather than oversized summary fields.

An operator-authorized independent checker must not import or call GUDHI. It
should reconstruct the oriented boundary matrix over the requested prime,
validate the filtration and tie order, replay column reduction, compare every
pairing, and bind exact values to the corresponding input simplices. Forged
simplex IDs, ranks, rational values, coefficients, pairings, infinity
sentinels, artifact digests, and checker identities must all fail closed.

Production remains deferred until that contract, artifact split, independent
checker package, authorization declaration, and adversarial suite exist. GUDHI
absence must continue to leave the built-in topology foundation and complete
unrelated catalog available.

## Handoff

- Candidate: `persistent homology`; discovery decision `REVISE`.
- Producer maximum: `COMPUTED`.
- Provider: GUDHI 3.13.0, optional T1 CPython 3.12 wheel.
- License audit: wheel, SimplexTree, and Persistent_cohomology selected slice
  are MIT; unrelated GUDHI modules are outside this decision.
- Public reproduction: one answer-visible deterministic filtered square over
  `F_2`; this is not held-out portfolio evidence.
- Checker evidence: independent stdlib modular reduction matches all six pairs
  and preserves eleven reduced columns.
- Open action: design the production filtered-complex contract and independent
  checker, then run a held-out matched comparison before keeping the provider
  capability.
