# Finite-scheme rational-points audit

This Assurance benchmark strengthens DeepTheorem row 10223: instead of the source's empty-scheme shortcut, the agent must construct two nonempty affine finite schemes over `F_5`, enumerate every rational point from multiplication tables, bind the induced point map, and exhibit a nilpotent obstruction to isomorphism.

Primary objective: exact semantic-countermodel certification. Quality: 89/100. Difficulty: Hard (provisional). The verifier exhaustively enumerates all linear functionals on both finite algebras and checks unital multiplicativity, the algebra morphism, point-map bijectivity, and reducedness separation. The result is `COMPUTED`; no general scheme theorem is machine proved.
