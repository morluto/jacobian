# Evaluation methods

[Benchmark home](../README.md) · [Benchmark contracts](benchmark-contracts.md)

Jacobian evaluations are operator-run evidence exercises, not server features
or routine pull-request gates. Compare a control with no Jacobian against a
treatment that has only the public MCP surface. Hold the model, task inputs,
budget, environment, and repetitions fixed; do not turn the treatment into a
prescribed workflow.

Select the task set for the mathematical capabilities it reveals, not for how
well existing operations happen to fit it. The control/treatment comparison
then asks whether the current tool surface helps on that fixed capability set;
persistent failures are evidence for a future operation or environment change.

Each Harbor task owns its hidden verifier and Oracle. Author that contract from
the [task template](../templates/task/README.md) and
[benchmark contracts](benchmark-contracts.md); do not copy an existing task's
hidden `expected.json` predicate, lowest-terms wording, keyword gate, or
universal certificate union. Report mathematical correctness, tool use, failure
modes, cost, and the limits of the task set separately. Atomic mathematical task
correctness is normally binary; only explicit independent replayable subclaims
justify partial credit. An evaluation score or solver outcome is evidence about
the experiment, not a new mathematical conclusion returned by Jacobian.

## Descriptive comparison and repeated observations

Comparison report v3 is descriptive only. The previous pair-level bootstrap,
McNemar p-value, and promotion to `comparative` after ten task/repetition pairs
were invalid for the declared nested task/family design. Repetition counts do
not establish independent task or family samples. The interval and p-value
fields are now null for every comparison, with an explicit explanation.

The report retains raw means and deltas, weighting each complete observed pair
equally. It also reports each task's paired-repetition mean and an equal-task
average of those means. Family summaries average observed task means equally
within each declared family. No overall equal-family estimand is substituted.
Optional metrics expose missing pair and task counts; available-case summaries
do not represent missing observations. Unpaired trial keys still invalidate the
comparison. Counts of family labels describe coverage, not statistical
independence or an effective sample size.

Normalized evidence v5 adds `task_family_binding`. Canonical held-out collection
projects the selected stage's task IDs, digests, and family labels only after
checking the exact manifest file digest against both plan and ledger and checking
complete stage trial coverage. Ordinary observations have a null binding.
Historical v4 evidence remains readable for descriptive comparisons, with
unknown family identity; no labels or sampling assumptions are invented.
Report v2 consumers must migrate to v3 and must not interpret null uncertainty as
zero uncertainty or a nonsignificant statistical test.

A future inferential analysis needs a frozen target estimand, task/family
weighting, sampling or assignment assumptions, missingness policy, and a method
appropriate to the number and structure of independent units. A family-label
minimum or randomized execution order does not specify these choices. This
repair prevents misleading certainty; it does not resolve that methodological
choice or introduce a cluster-bootstrap guarantee.
