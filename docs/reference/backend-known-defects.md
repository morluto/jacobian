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
| [SymPy #10666](https://github.com/sympy/sympy/issues/10666) — `resultant` returns `(-1)^(m*n)` times the true Sylvester determinant when `deg(left) < deg(right)` and both degrees are odd; the subresultant PRS swaps the inputs without recompensing the swap sign. Still present in SymPy 1.14. | `polynomial.multivariate.resultant.compute` (`SYLVESTER_DETERMINANT` convention) | `_sylvester_resultant_value` in `src/jacobian/math/polynomials/multivariate/_operations.py` negates the backend value under exactly that degree condition. | `tests/math/polynomials/test_multivariate_polynomial.py`: `test_resultant_matches_sylvester_determinant_oracle` (independent determinant oracle), `test_resultant_sign_matches_sylvester_orientation`, `test_resultant_swap_law_with_degenerate_rows`. Remove the compensation when a SymPy release fixes #10666; the swap-law and oracle tests then hold with the compensation deleted. |

## Adding an entry

Name the upstream issue, the exact affected operation and declared
convention, the compensating code location, and the guard test whose failure
signals that the backend changed. Prefer a guard built from an independent
construction — an object assembled from the definition rather than another
path through the same backend — so the test cannot reproduce the shared
defect. Record the backend versions known to be affected.
