# Indexed pairwise identity-loss audit

This Assurance benchmark freezes the `Set.range` quantification defect reported in google-deepmind/formal-conjectures issue #4045 against source commit `71d34fc2958f1010ee43ccd4d54a9631574d7ef9`.

The agent must construct a nontrivial exact coset covering of a cyclic group whose repeated part value is deduplicated by `Set.range`. The verifier reconstructs the subgroup, binds an explicit shared-part artifact to it, follows every covering-index reference, rebuilds every coset, and evaluates both the value-range and indexwise pairwise predicates. Alternative groups satisfying the frozen bounds are accepted.

The core defect is **index identity loss**: value-level deduplication cannot preserve multiplicity-sensitive or index-sensitive predicates. Singleton-range vacuity is the manifestation in this witness, not the full semantic category.

Difficulty is **Medium-Hard / Hard (provisional)** because the certificate combines finite-group construction, exact-cover completeness, and a higher-order quantifier audit, while the construction architecture itself is supplied. Baseline calibration is still pending. This does not prove or disprove Erdős problem 274.
