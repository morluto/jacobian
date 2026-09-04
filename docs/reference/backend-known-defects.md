# Known backend defects

[Documentation home](../index.md)

This registry lists defects in maintained mathematical backends that Jacobian
adapters compensate for. A backend defect is any behavior where the backend
returns a mathematically wrong or unsound value for inputs inside its own
documented domain — not a limitation that narrows what the backend supports.

Before trusting backend output for a new claim, check this registry. Add an
entry whenever an adapter compensates for backend behavior instead of
narrowing the public domain, and remove the entry (and the compensation) when
an upgraded backend version makes it redundant. The guard test named in each
entry must fail if the backend behavior changes in either direction.

| Upstream defect | Affected operations | Adapter compensation | Guard tests |
| --- | --- | --- | --- |
| [SymPy #10666](https://github.com/sympy/sympy/issues/10666) — `resultant` returns `(-1)^(m*n)` times the true Sylvester determinant when `deg(left) < deg(right)` and both degrees are odd; the subresultant PRS swaps the inputs without recompensing the swap sign. Still present in SymPy 1.14. | `polynomial.multivariate.resultant.compute` (`SYLVESTER_DETERMINANT` convention) | `_sylvester_resultant_value` in `src/jacobian/math/polynomials/multivariate/_resultant.py` negates the backend value under exactly that degree condition. | `tests/math/polynomials/test_multivariate_polynomial.py`: `test_resultant_matches_sylvester_determinant_oracle` (independent determinant oracle), `test_resultant_sign_matches_sylvester_orientation`, `test_resultant_swap_law_with_degenerate_rows`. Remove the compensation when a SymPy release fixes #10666; the swap-law and oracle tests then hold with the compensation deleted. |
| SymPy 1.14.0 exact simplex returns infeasible candidates: [Jacobian #3194](https://github.com/morluto/jacobian/issues/3194) and [#3192](https://github.com/morluto/jacobian/issues/3192) record the defect; these are local reports, not upstream issue IDs. Matrix `linprog` returns objective 1 at `(0,1)` for `min x+y`, `x+y>=1`, `x>=1`, `y>=1`, although the optimum is 2. Switching from symbolic `lpmin` to `linprog` does not repair the defect. | `optimization.linear.rational_optimum.compute` and `optimization.linear.rational_general_optimum.compute` | `_linear_basis.py` under `src/jacobian/math/optimization` replaces simplex with bounded basis enumeration and FLINT 0.9.0 exact elimination. `operations.py` checks primal/dual feasibility and objective equality, Farkas signs and pairing, or a feasible point and improving nonnegative ray before trusted result construction. Invalid candidates remain execution failures. | `tests/math/optimization/test_linear_certificates.py`: `test_pinned_sympy_matrix_lp_defect` records the exact bad backend answer and fails when it changes. `test_cover_row_order_preserves_exact_optimum`, `test_standard_cover_has_valid_objective_two_certificate`, and `test_pair_cover_original_and_trimmed_encodings` independently check the repaired public certificates. A backend upgrade warrants reassessment, not automatic restoration of the old solve path. |

## Exact LP workload calibration

The LP replacement uses the already-pinned FLINT binding. Maintained exact LP
alternatives include [SoPlex](https://soplex.zib.de/) and
[GMP cddlib through pycddlib](https://pycddlib.readthedocs.io/en/stable/);
adopting either would require an additional native dependency. The finite basis
families and conservative scalar-update/minor-height bounds are documented in
`src/jacobian/math/optimization/_linear_basis.py` and exposed through inspection.
They establish admission; timings do not.

On 2026-09-05, Python 3.12.13 / FLINT 0.9.0 on x86-64 completed five runs of the
#3192 pair cover at objective 2 in 0.0048–0.0054 seconds for ten source variables
and 0.0060–0.0068 seconds for the original fifteen-variable encoding. These
measurements include parsing, normalization, solving, source mapping, and JSON
serialization. Both reserve 12,376 bases and 32,088,000 scalar updates after
removing unused columns.

A full-rank six-row, seventeen-column Vandermonde control with negative RHS
forces all 12,376 original bases to fail feasibility before phase I. Its complete
reservation is 18,564 bases and 49,909,832 updates, just below the 50,000,000
update limit; it completed in 0.236 seconds with small coefficients. Replacing
its nodes by `10**24+j+1` gives coefficients up to 121 digits and completed in
0.430 seconds including serialization. The LP-specific 600-second safety
deadline leaves over a thousandfold margin over these measurements for other
admitted coefficient regimes and host variation. It is cooperative between
bounded FLINT primitives, not a hard interruption inside a native matrix call.
Normalization, both searches, certificate construction and dispatch projection
share the original request deadline; no phase receives a fresh allowance.

## Adding an entry

Name the upstream issue, the exact affected operation and declared
convention, the compensating code location, and the guard test whose failure
signals that the backend changed. Prefer a guard built from an independent
construction — an object assembled from the definition rather than another
path through the same backend — so the test cannot reproduce the shared
defect. Record the backend versions known to be affected.
