# Local-ring diagonal similarity certificate

Regression benchmark from DeepTheorem row 10040. The agent must replay `PA=BP` over `Z/125Z`, compute an exact determinant, and extract a determinant-expansion permutation whose selected entries are all units. That unit term forces the corresponding diagonal entries of `A` and `B` to match.

Family: Regression. Primary objective: exact algebraic proof certification. Quality: 86/100. Difficulty: Hard (provisional). The six-dimensional block-mixed matrix prevents a tiny permutation-matrix shortcut; the verifier independently evaluates all 720 determinant terms and all modular products. It checks only this frozen certificate, not the general local-ring theorem.
