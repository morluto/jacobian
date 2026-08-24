"""Exact bounded powerful-number decision and declaration."""

from __future__ import annotations

from jacobian.canonical import format_canonical_integer, parse_canonical_integer
from jacobian.catalog._examples import example
from jacobian.math.number_theory._models import (
    PowerfulNumberRequest,
    PowerfulNumberResult,
    PrimePower,
    ResidualPerfectPower,
)
from jacobian.math.number_theory._powerful_kernels import decide_powerful_data
from jacobian.math.number_theory._support import number_theory_operation


def decide_powerful(request: PowerfulNumberRequest) -> PowerfulNumberResult:
    """Return the exact source-bound powerful-number decision."""

    data = decide_powerful_data(parse_canonical_integer(request.value))
    return PowerfulNumberResult(
        semantics_version="powerful-number.partial-factor.v2",
        value=request.value,
        conclusion=data.conclusion,
        is_powerful=data.conclusion == "POWERFUL",
        cutoff=data.cutoff,
        checked_through=data.checked_through,
        stripped_factors=tuple(
            PrimePower(prime=format_canonical_integer(prime), power=exponent)
            for prime, exponent in data.stripped_factors
        ),
        residual=format_canonical_integer(data.residual),
        residual_perfect_power=(
            None
            if data.perfect_power is None
            else ResidualPerfectPower(
                base=format_canonical_integer(data.perfect_power[0]),
                exponent=data.perfect_power[1],
            )
        ),
    )


POWERFUL_NUMBER_OPERATION = number_theory_operation(
    "integer.decide.powerful",
    "Decide powerful-number status",
    "Decide exactly whether every prime exponent of one positive integer is at "
    "least two. Trial division through the derived cutoff B=ceil(n^(1/5)) and "
    "exact perfect-power classification of the B-rough residual avoid complete "
    "factorization while returning a source-bound replayable certificate.",
    PowerfulNumberRequest,
    PowerfulNumberResult,
    decide_powerful,
    "number-theory",
    "2-full",
    "powerful",
    "powerful-number",
    "predicate",
    "certificate",
    examples=(
        example(
            "powerful_12168",
            "Decide whether 12168 is powerful from a bounded partial-factor "
            "certificate; the value must be a positive canonical integer with "
            "at most 25 digits.",
            {"value": "12168"},
        ),
    ),
    version="3",
)
