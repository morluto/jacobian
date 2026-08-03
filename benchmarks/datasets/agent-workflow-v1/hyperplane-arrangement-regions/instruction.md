# Certify the cube-and-tetrahedron plane arrangement

Use coordinates `A=(0,0,0)`, `B=(1,0,0)`, `C=(1,1,0)`, `D=(0,1,0)` and the corresponding top vertices with `z=1`. Consider the six cube-face planes and the four face planes of tetrahedron `B A_1 C_1 D_1`.

Submit all ten labeled plane equations in any order, the exact number of newly created regions at each insertion, the duplicate-plane group, and the final region count. Coefficients may be scaled by any nonzero integer. The verifier canonicalizes planes and independently computes the induced affine-line arrangement on every inserted plane using exact rational arithmetic. A memorized total or a generic-position formula fails. Bind the result with exactly one `RESULT_JSON:` evidence line and do not claim proof-assistant verification.
