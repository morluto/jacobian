# Reduce an ordered-simplex linear program

Use chain differences before calling
`optimization.linear.rational_general_optimum.compute` for a program of the form

\[
\ell\le x_1\le\cdots\le x_n,\qquad \sum_i x_i=s.
\]

Set \(y_1=x_1-\ell\) and \(y_i=x_i-x_{i-1}\) for \(i>1\).
The inverse is \(x_i=\ell+\sum_{j\le i}y_j\). This is a bijection to

\[
y_j\ge0,\qquad \sum_{j=1}^n(n-j+1)y_j=s-n\ell.
\]

For objective \(\sum_i c_i x_i\), use coefficient
\(\sum_{i=j}^n c_i\) for \(y_j\), with the same optimization sense, and
add the constant \(\ell\sum_i c_i\) to the returned objective. The transformed
program has one equality instead of a separate slack column for each ordering
constraint. Variable names and constraint labels do not affect the transform;
identify the chain order from the actual coefficients.

For example, maximizing the first four coordinates with \(n=9\),
\(\ell=1/10\), and \(s=1\) gives nonnegative variables with equality
coefficients `(9, 8, 7, 6, 5, 4, 3, 2, 1)`, right side `1/10`, and objective
coefficients `(4, 3, 2, 1, 0, 0, 0, 0, 0)`. The reduced optimum is `2/45`,
attained at `y1=1/90` with the remaining coordinates zero. Reconstruction gives
all original coordinates `1/9` and objective `2/5 + 2/45 = 4/9`.
Independently, the mean of the first four coordinates of an increasing sequence
cannot exceed the mean of all nine, proving the matching upper bound `4/9`.

This is caller-side modeling guidance. Automatic chain recognition is not
implemented, and the unreduced nine-variable request still exceeds the basis
work budget. Returned dual multipliers refer to the reduced program; they must
not be interpreted as multipliers of the original ordering constraints. Check
the reconstructed point against the original constraints and objective. Additional
constraints must also be transformed by substitution; upper bounds, unequal
lower bounds, or a different equality do not permit simply dropping those rows.

The executable example and original-coordinate checks are in
`tests/integration/linear/test_general_rational_linear_program.py`.
