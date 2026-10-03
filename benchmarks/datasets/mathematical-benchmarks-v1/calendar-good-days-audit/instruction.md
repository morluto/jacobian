# Audit a finite calendar claim

Audit the claim in the offline input by exhaustively checking the declared
finite date range. Return the exact count and every qualifying
date in calendar order, including each concatenated integer.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The answer is bounded by the 92 dates in the public calendar: good_dates has at most 92 rows and count is an integer from 0 through 92. Each row uses month 3, 4, or 5 and a day from 1 through that month's published day count (31, 30, or 31). The concatenated integer is between 31 and 331 for March, 41 and 430 for April, or 51 and 531 for May. These are feasibility bounds, not the number or identity of good dates. The verifier separately checks concatenation, divisibility, whether every good date is listed, exact count, and calendar order.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
