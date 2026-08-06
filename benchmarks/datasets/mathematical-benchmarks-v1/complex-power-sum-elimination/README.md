# Complex power-sum elimination

This task freezes ConstructiveBench row 1672 (`omnimath1603`) from ECP commit
`ac6b9ff5614ce8454c03d8c03bff571b91f6d31a`. The source row and dataset-file
digests are recorded in the frozen input.

The benchmark requires a complete symmetric-polynomial elimination rather than
answer recovery. Its verifier independently rebuilds Newton's power-sum
recurrence, derives the two quadratic-field branches, checks all denominator
obligations, evaluates the target, and confirms complex-root achievability.
Branch order is not fixed.

The ECP repository is MIT licensed. The selected row records HMMT_2 as its
upstream source collection; downstream users remain responsible for upstream
problem attribution.
