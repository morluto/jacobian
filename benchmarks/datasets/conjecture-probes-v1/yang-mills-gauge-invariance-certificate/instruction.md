# Replay one exact rational SU(2) plaquette gauge transformation

Represent SU(2) elements as unit quaternions `(w,x,y,z)` over `QQ`. Choose four
nonidentity link quaternions on the oriented square `0→1→2→3→0` and four
nonidentity, pairwise-distinct vertex gauge quaternions, within the frozen
coefficient bounds. Submit the transformed links `g_i U_ij g_j^-1`, the
original and transformed ordered plaquette products, and `g_0 P g_0^-1`.

The verifier independently checks canonical rational form, unit norms,
Hamilton products, inverses, every transformed link, plaquette conjugacy, and
invariance of the plaquette scalar part. Identity-only and commutative-scalar
shortcuts are rejected.

Evidence is a matching JSON object with exactly `schema_version`, `task_id`,
`result`, and `limitations`, This is one finite lattice-gauge
identity. It says nothing about continuum construction or a Yang–Mills mass
gap.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result and validates the declared task-specific witness.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result` and the declared `witness`.

- **Witness:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`; media type(s): `text/plain`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
