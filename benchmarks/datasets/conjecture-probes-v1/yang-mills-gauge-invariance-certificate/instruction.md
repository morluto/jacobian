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

Exact finite SU(2) quaternion replay only; no continuum or mass-gap conclusion.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** exactly `FINITE_SU2_PLAQUETTE_GAUGE_INVARIANCE`
- **Assurance:** scoreable values are `UNVERIFIED`, `COMPUTED`, `CHECKED` (ceiling `CHECKED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** the exact value declared in `submission_schema.json`
- **Completeness:** `COMPLETE`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `text/plain`.
- **Required artifact filenames:** `evidence/answer.txt`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
