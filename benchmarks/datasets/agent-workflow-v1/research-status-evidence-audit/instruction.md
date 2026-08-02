# Audit cited evidence for mathematical problem status

Read `/app/input.json`. For every case, decide only what the frozen cited evidence supports at the declared cutoff. Do not solve any problem and do not treat the dataset's `reported_status` as authority.

Write `/app/submission.json` following `/app/submission_schema.json` and a concise explanation to `/app/evidence/answer.txt`. Include one `RESULT_JSON:` line containing the exact submitted `result` object as compact JSON, so the explanation is bound to the classified cases. For each case, select exactly the citation IDs needed for your classification and list every unsupported inference made by the dataset report.

Use exactly these classifications: `RESOLVED_BY_CITED_WORK`, `PARTIAL_PROGRESS_NO_FULL_RESOLUTION`, `HISTORICALLY_OPEN_CURRENT_STATUS_NOT_ESTABLISHED`, or `PROBLEM_LISTED_STATUS_UNSUPPORTED`.

Claim at most `COMPUTED`. The result audits frozen citation-to-claim support; it does not establish current theorem status.
Set `scope` to: `the four frozen citation packets only; present mathematical truth and current literature status are not assessed`.
