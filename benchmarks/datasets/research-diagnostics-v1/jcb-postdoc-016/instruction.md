# Erdős Problem 364: exact local checks versus an unbounded conjecture

A positive integer is powerful when every prime in its factorization has
exponent at least two. The public reference result checks that no three
consecutive powerful numbers occur with the largest member at most `10^14`;
the corresponding unbounded conjecture remains open.

Produce an independently replayable finite diagnostic for every integer from
8 through 16. Record the complete prime factorization, the exact powerful
decision, and every violating prime. Then check each consecutive triple
starting at 8 through 14 and record all non-powerful witnesses in that triple.
Write the structured evidence to `evidence/powerful-window.json`, bind its
SHA-256 digest in `submission.json`, and follow the exact public schemas.

The public `10^14` artifact and its pinned replay environment are not included.
Set `bounded_reference_replayed` to `false`; do not extrapolate the local
window to `10^14` or to all positive integers. The maximum allowed assurance is
`COMPUTED`. Timeout, error, incomplete factorization, or incomplete coverage is
a non-conclusion.
