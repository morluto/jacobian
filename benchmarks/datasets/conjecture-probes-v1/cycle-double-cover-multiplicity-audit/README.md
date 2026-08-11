# cycle-double-cover-multiplicity-audit

An Assurance benchmark derived from retained conjecture row C-049. It tests whether an agent detects the difference between set-union edge coverage and exact multiplicity-two coverage, then constructs a genuine cycle double cover of the Petersen graph.

The clean-room verifier parses arbitrary simple cycles, canonicalizes cyclic symmetry, and recomputes all fifteen edge multiplicities. The task resists label-only and tiny-witness shortcuts and accepts non-oracle cycle systems. Provisional difficulty is Hard because it combines adversarial contract diagnosis with a constrained cycle-system construction; calibration is pending.
