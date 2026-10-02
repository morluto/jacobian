"""Controlled square-free orchestration over maintained integer PRS primitives.

No heuristic polynomial GCD dispatch or global backend configuration is used.
The owner admits the complete recursive content/subresultant envelope first.
"""

from __future__ import annotations

from math import gcd
from typing import Any

from jacobian._execution import request_checkpoint


def _positive(poly: Any) -> Any:
    return -poly if poly.LC() < 0 else poly


def _lift(poly: Any, generators: tuple[Any, ...]) -> Any:
    from sympy import ZZ, Poly

    return Poly.from_dict(
        {(0, *powers): coefficient for powers, coefficient in poly.terms()},
        *generators,
        domain=ZZ,
    )


def _content(poly: Any) -> Any:
    """Main-variable content, including its integer ground content."""
    from sympy import ZZ, Poly

    rows: dict[int, dict[tuple[int, ...], Any]] = {}
    for powers, coefficient in poly.terms():
        rows.setdefault(powers[0], {})[powers[1:]] = coefficient
    result = None
    for row in rows.values():
        coefficient = Poly.from_dict(row, *poly.gens[1:], domain=ZZ)
        result = (
            _positive(coefficient)
            if result is None
            else integer_gcd(result, coefficient)
        )
        if result.is_one:
            break
    if result is None:
        return Poly(0, *poly.gens[1:], domain=ZZ)
    return result


def integer_gcd(left: Any, right: Any) -> Any:
    """Positive-leading ZZ GCD via Gauss content and maintained subresultants."""
    from sympy import ZZ, Poly

    request_checkpoint("controlled square-free integer GCD")
    if left.is_zero:
        return _positive(right)
    if right.is_zero:
        return _positive(left)
    if left == right:
        return _positive(left)
    left_ground, left = left.primitive()
    right_ground, right = right.primitive()
    ground = gcd(abs(int(left_ground)), abs(int(right_ground)))
    if left.total_degree() == 0 or right.total_degree() == 0:
        return Poly(ground, *left.gens, domain=ZZ)
    if len(left.gens) == 1:
        last = left.subresultants(right)[-1]
        return ground * _positive(last.primitive()[1])

    left_content, right_content = _content(left), _content(right)
    primitive_left = left.exquo(_lift(left_content, left.gens), auto=False)
    primitive_right = right.exquo(_lift(right_content, right.gens), auto=False)
    last = primitive_left.subresultants(primitive_right)[-1]
    primitive_last = last.exquo(_lift(_content(last), last.gens), auto=False)
    common_content = integer_gcd(left_content, right_content)
    result = primitive_last * _lift(common_content, left.gens)
    return ground * _positive(result.primitive()[1])


def grouped_factors(source: Any) -> list[tuple[Any, int]]:
    """Factor a primitive positive-leading ZZ source by exact multiplicity."""
    common = source
    for generator in source.gens:
        if source.degree(generator):
            common = integer_gcd(common, source.diff(generator))
        if common.total_degree() == 0:
            break
    remaining = source.exquo(common, auto=False)
    multiplicity = 1
    factors = []
    while remaining.total_degree() > 0:
        request_checkpoint("controlled square-free multiplicity group")
        shared = integer_gcd(remaining, common)
        factor = remaining.exquo(shared, auto=False)
        if factor.total_degree() > 0:
            factors.append((factor, multiplicity))
        remaining = shared
        common = common.exquo(shared, auto=False)
        multiplicity += 1
    if not common.is_one:
        raise RuntimeError("square-free GCD chain did not exhaust its repeated part")
    return factors
