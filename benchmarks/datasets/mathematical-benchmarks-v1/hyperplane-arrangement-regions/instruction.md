# Certify the cube-and-tetrahedron plane arrangement

Use coordinates `A=(0,0,0)`, `B=(1,0,0)`, `C=(1,1,0)`, `D=(0,1,0)` and the corresponding top vertices with `z=1`. Consider the six cube-face planes and the four face planes of tetrahedron `B A_1 C_1 D_1`.

Label the ten planes exactly with this vocabulary: `cube_x0`, `cube_x1`, `cube_y0`, `cube_y1`, `cube_z0`, `cube_z1` for the cube faces `x=0`, `x=1`, `y=0`, `y=1`, `z=0`, `z=1`, and `tetra_A1C1D1`, `tetra_BC1D1`, `tetra_BA1D1`, `tetra_BA1C1` for the tetrahedron face planes through the named vertices.

Submit all ten labeled plane equations in any order, the exact number of newly created regions at each insertion, the duplicate-plane group, and the final region count. Each plane is given as four integer coefficients `[a, b, c, d]` in the convention `a*x + b*y + c*z = d` (not the homogeneous form `a*x + b*y + c*z + d = 0`); coefficients may be scaled by any nonzero integer. The verifier canonicalizes planes and independently computes the induced affine-line arrangement on every inserted plane using exact rational arithmetic. A memorized total or a generic-position formula fails.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
