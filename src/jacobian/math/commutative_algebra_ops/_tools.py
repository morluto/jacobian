"""Commutative algebra operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.commutative_algebra_ops._models import (
    IdealQuotientRequest,
    IdealQuotientResult,
    IdealRadicalMembershipRequest,
    IdealRadicalMembershipResult,
    IdealRadicalRequest,
    IdealRadicalResult,
)
from jacobian.math.commutative_algebra_ops._operations import (
    compute_ideal_quotient,
    compute_ideal_radical,
    compute_ideal_radical_membership,
)


def _op[RequestT: StrictModel, ResultT: StrictModel](
    operation_id: str,
    title: str,
    description: str,
    request_model: type[RequestT],
    result_model: type[ResultT],
    operation: Callable[[RequestT], ResultT],
    *tags: str,
    examples: tuple[OperationExample, ...] = (),
    version: str = "1",
) -> MathTool[RequestT, ResultT]:
    return MathTool(
        operation_id=operation_id,
        version=version,
        title=title,
        description=description,
        request_type=request_model,
        result_type=result_model,
        run=operation,
        tags=tags,
        examples=examples,
    )


TOOLS: tuple[MathTool[Any, Any], ...] = (
    _op(
        "polynomial.ideal.radical.compute",
        "Compute the radical of an ideal",
        "Compute the radical sqrt(I) of a polynomial ideal using SymPy's "
        "Groebner basis machinery.",
        IdealRadicalRequest,
        IdealRadicalResult,
        compute_ideal_radical,
        "commutative-algebra",
        "radical",
        "exact",
        examples=(
            example(
                "ideal_xy",
                "Radical of <x^2, xy> in Q[x,y].",
                {
                    "variables": ["x", "y"],
                    "generators": ["x**2", "x*y"],
                },
            ),
        ),
    ),
    _op(
        "polynomial.ideal.radical_membership.decide",
        "Check membership in the radical of an ideal",
        "Check whether a polynomial f lies in the radical sqrt(I) of the "
        "ideal I = <generators> in Q[variables], using the Rabinowitsch "
        "trick.",
        IdealRadicalMembershipRequest,
        IdealRadicalMembershipResult,
        compute_ideal_radical_membership,
        "commutative-algebra",
        "radical-membership",
        "exact",
        examples=(
            example(
                "membership_xy",
                "Check if x is in sqrt(<x^2>) in Q[x].",
                {
                    "variables": ["x"],
                    "generators": ["x**2"],
                    "polynomial": "x",
                },
            ),
        ),
    ),
    _op(
        "polynomial.ideal.quotient.compute",
        "Compute the ideal quotient (I : J)",
        "Compute the colon ideal (I : J) = {f : f*J subseteq I} using SymPy.",
        IdealQuotientRequest,
        IdealQuotientResult,
        compute_ideal_quotient,
        "commutative-algebra",
        "ideal-quotient",
        "exact",
        examples=(
            example(
                "quotient_xy",
                "Compute (<x^2, xy> : <x>) in Q[x,y].",
                {
                    "variables": ["x", "y"],
                    "generators_a": ["x**2", "x*y"],
                    "generators_b": ["x"],
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
