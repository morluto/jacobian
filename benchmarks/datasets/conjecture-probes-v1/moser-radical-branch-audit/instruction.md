# Audit a radical-branch error in a Moser-spindle embedding

A claimed Moser-spindle embedding uses the complete labeled coordinates in
`input.json`. Each x-coordinate pair `[a,b]` denotes `a+b*s`, and the y-coordinate
strings use the named positive radicals. The corrupted embedding chooses the
negative branch `-B/2` for vertex 5, where
`s^2=33`, `A^2=(17+s)/6`, `B^2=(17-s)/6`, `t^2=11`, all radicals are positive,
`At=(33+s)/6`, `Bt=(33-s)/6`, and `AB=8/3`.

Audit all 21 unordered vertex pairs exactly. Express each squared distance as
`a+b*s` using canonical rational strings, identify every claimed edge that is
not unit under the corrupted branch, then repair vertex 5 to `+B/2` and submit
the complete corrected pair table and exact 11-edge unit-distance graph.
Pair-table rows and edge collections may appear in any order, and either endpoint
order denotes the same unordered pair.

The verifier performs arithmetic in `Q(sqrt(33))`, derives all y-difference
squares from the radical relations, and does not use floating-point tolerance.
This checks one exact finite embedding and does not determine the chromatic
number of the plane.

After every protocol, input, mathematical, evidence, and scope gate passes,
assurance contributes `1.0` for `COMPUTED`, `0.5` for `COMPUTED`, and `0.25` for
`UNVERIFIED`. Thus full reward is reserved for the intended checked claim.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result and validates the declared task-specific witness.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result` and the declared `witness`.

- **Witness:** 1-1 item(s); allowed path(s): `evidence/answer.json`; digest must match `^sha256:[0-9a-f]{64}$`; media type(s): `application/json`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
