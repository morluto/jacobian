# Replay one exact rational SU(2) plaquette gauge transformation

Represent SU(2) elements as unit quaternions `(w,x,y,z)` over `QQ`. Choose four
nonidentity link quaternions on the oriented square `0→1→2→3→0` and four
nonidentity, pairwise-distinct vertex gauge quaternions, within the frozen
coefficient bounds. Submit the transformed links `g_i U_ij g_j^-1`, the
original and transformed ordered plaquette products, and `g_0 P g_0^-1`.

Represent each quaternion component as an object with integer `numerator` and
positive integer `denominator`. Equivalent encodings such as `6/10` and `3/5`
are accepted after exact `Fraction` normalization. The verifier independently
checks unit norms, Hamilton products, inverses, every transformed link,
plaquette conjugacy, and invariance of the plaquette scalar part.
Identity-only and commutative-scalar shortcuts are rejected.

This is one finite lattice-gauge identity. It says nothing about continuum
construction or a Yang–Mills mass gap.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
