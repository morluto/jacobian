# total-coloring-contract-audit

An Assurance benchmark derived from retained conjecture row C-048. It tests whether an agent can distinguish two valid projected colorings from a valid total coloring, produce a concrete counterexample to the incomplete validator, and repair the assignment.

The fixture is deliberately non-tiny: all 25 graph objects and all three constraint families are replayed. The verifier accepts non-oracle colorings, recomputes the exact collision set, and caps assurance at `COMPUTED` for this Petersen graph only. Provisional difficulty is Hard because the task combines adversarial contract analysis with a complete four-color repair; empirical calibration is pending.
