"""Preflight for SymPy 1.14's full recursive polynomial Smith kernel.

Heights below describe a polynomial as F/q with integer F and positive q;
both the coefficient infinity norm and q are at most 2**height. They do
not multiply unrelated rational denominators at every coefficient operation.
All arithmetic here is on size metadata, before backend conversion.
"""

from dataclasses import dataclass

from jacobian.catalog.models import OperationResourceAdmissionError

_MAX_DEGREE = 4095
_MAX_HEIGHT = 100_000
_MAX_WORK = 20_000_000


def _reject() -> None:
    raise OperationResourceAdmissionError(
        location=("matrix",),
        code="matrix.polynomial_smith_budget",
        message="polynomial Smith work, intermediate growth, or exact output exceeds the admitted envelope",
    )


@dataclass(frozen=True)
class PolynomialSmithBound:
    degree: int
    height: int
    work: int


def _check(degree: int, height: int, work: int) -> PolynomialSmithBound:
    if degree > _MAX_DEGREE or height > _MAX_HEIGHT or work > _MAX_WORK:
        _reject()
    return PolynomialSmithBound(degree, height, work)


def _division_height(degree: int, height: int, *, constant: bool) -> int:
    # Pseudo-division by an integer polynomial multiplies coefficient height
    # by at most its leading coefficient and adds one product per step.
    # At most degree+1 steps; restoring the two source denominators adds one
    # further divisor-height term. Constant division is just scalar scaling.
    return 2 * height if constant else (degree + 4) * (height + 1)


def _bezout_height(degree: int, pivot_degree: int, height: int) -> int:
    if pivot_degree == 1:
        # The nondivisible branch with a linear pivot has one constant
        # remainder. One long division, one scalar division, normalization,
        # and reconstruction of the second Bezout coefficient suffice.
        return (8 * degree + 32) * (height + 1)
    # Extended Euclid first divides degree D by degree p, then the divisor
    # degree strictly descends. Consecutive descending degrees maximize the
    # product of the pseudo-division height factors: merging k degree drops
    # replaces factors at least six by a factor growing only linearly in k.
    # Track accumulator products too, not only the Euclidean remainders.
    bound = height
    dividend_degree = degree
    for divisor_degree in range(pivot_degree, -1, -1):
        quotient_height = (
            2 * bound
            if divisor_degree == 0
            else bound + (dividend_degree - divisor_degree + 3) * (bound + 1)
        )
        bound = quotient_height + 2 * bound + (degree + 1).bit_length() + 1
        dividend_degree = divisor_degree
        if bound > _MAX_HEIGHT:
            _reject()
    # Monic normalization, t=(gcd-s*f)/g in SymPy's dup_gcdex, and
    # exact division of both original operands by the resulting monic gcd.
    return (2 * degree + 4) * (6 * bound + degree.bit_length() + 2)


def _clear_pivot_bound(
    rows: int,
    columns: int,
    degree: int,
    height: int,
    pivot_degree: int,
) -> PolynomialSmithBound:
    cells = rows * columns + rows * rows + columns * columns
    # A single row or column is cleared in one pass: no opposite operation
    # can reintroduce an entry along that pivot axis.
    passes = 1 if min(rows, columns) == 1 else pivot_degree + 1
    steps = (rows + columns - 2) * passes
    if steps > _MAX_WORK:
        _reject()
    states = {(0, False): _check(degree, height, cells)}
    completed: list[PolynomialSmithBound] = []
    for step in range(steps):
        next_states: dict[tuple[int, bool], PolynomialSmithBound] = {}
        for (used, changed), state in states.items():
            p = max(0, pivot_degree - used)
            d, h = state.degree, state.height
            for gcd_step in (False, True) if p else (False,):
                coefficient_height = (
                    _bezout_height(d, p, h)
                    if gcd_step
                    else _division_height(d, h, constant=p == 0)
                )
                # Two polynomial products and their sum cover every tracked
                # matrix entry, including both transformation matrices.
                new = _check(
                    2 * d,
                    2 * (coefficient_height + h + (d + 1).bit_length()) + 1,
                    state.work + 100 * (d + 1) ** 3 + 8 * cells * (d + 1) ** 2,
                )
                key = used + int(gcd_step), changed or gcd_step
                old = next_states.get(key)
                next_states[key] = (
                    new
                    if old is None
                    else _check(
                        max(old.degree, new.degree),
                        max(old.height, new.height),
                        max(old.work, new.work),
                    )
                )
        states = next_states
        if (step + 1) % (rows + columns - 2) == 0:
            completed.extend(states.values())
            # A pass without a gcd has cleared both axes and must terminate.
            # Continuing it would compound bounds along impossible paths.
            states = {
                (used, False): state
                for (used, changed), state in states.items()
                if changed
            }
            if not states:
                break
    return _check(
        max(state.degree for state in completed),
        max(state.height for state in completed),
        max(state.work for state in completed),
    )


def polynomial_smith_bound(
    rows: int,
    columns: int,
    degree: int,
    height: int,
    *,
    pivot_degree: int,
    diagonal: bool = False,
) -> PolynomialSmithBound:
    """Bound full transforms, recursive products and divisibility repairs.

    Every nonexact gcd update strictly lowers the pivot degree. Each pass
    without such an update finishes clearing the pivot, so there are at most
    p+1 passes of r+c-2 updates. Keep alternatives by gcd count: exact
    divisions after a constant pivot must not inherit polynomial-division
    growth. Zero support is removed by the caller before this recurrence.
    """
    if not rows or not columns:
        return _check(0, 1, rows * rows + columns * columns)
    if degree == 0:
        # Over QQ every nonzero pivot divides every entry. The maintained
        # kernel is Gaussian elimination, with Schur entries and elementary
        # factors ratios of source minors. Products of at most size factors
        # bound accumulated transforms; there are no divisibility repairs.
        size = max(rows, columns)
        return _check(
            0,
            8 * size * size * (height + size.bit_length() + 2),
            64 * size**4,
        )
    if rows == columns == 1:
        return _check(degree, height, degree + 1)
    cells = rows * columns + rows * rows + columns * columns
    cleared = (
        _check(degree, height, cells)
        if diagonal
        else _clear_pivot_bound(rows, columns, degree, height, pivot_degree)
    )
    if min(rows, columns) == 1:
        return cleared
    child = polynomial_smith_bound(
        rows - 1,
        columns - 1,
        cleared.degree,
        cleared.height,
        pivot_degree=cleared.degree,
        diagonal=diagonal,
    )
    size = max(rows, columns)
    if diagonal:
        # Before embedding the child, both outer transforms are identity.
        # Their maintained matrix products add work, but no entry growth.
        combined = _check(
            max(degree, child.degree),
            max(height, child.height),
            child.work + 64 * size**4,
        )
    else:
        combined = _check(
            cleared.degree + child.degree,
            size * (cleared.height + child.height + (cleared.degree + 1).bit_length())
            + size.bit_length(),
            cleared.work + child.work + 8 * size**3 * (cleared.degree + 1) ** 2,
        )
    # A constant pivot divides the entire child diagonal and requires no
    # repair. Otherwise each adjacent repair uses one extended gcd and five
    # elementary transform updates. Invariant-factor degrees are bounded by
    # the degree of a maximal nonzero source minor.
    if pivot_degree:
        factor_degree = min(rows, columns) * degree
        for repair in range(min(rows, columns) - 1):
            repair_height = _bezout_height(
                factor_degree,
                min(pivot_degree, factor_degree) if repair == 0 else factor_degree,
                combined.height,
            )
            combined = _check(
                combined.degree + 5 * factor_degree,
                32 * (combined.height + repair_height + factor_degree.bit_length() + 2),
                combined.work
                + 100 * (factor_degree + 1) ** 3
                + 40 * cells * (combined.degree + factor_degree + 1) ** 2,
            )
    return combined


__all__ = ["PolynomialSmithBound", "polynomial_smith_bound"]
