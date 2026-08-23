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
    IdealSaturationRequest,
    IdealSaturationResult,
)
from jacobian.math.commutative_algebra_ops._operations import (
    compute_ideal_quotient,
    compute_ideal_radical,
    compute_ideal_radical_membership,
    compute_ideal_saturation,
)


def _polynomial(
    variables: tuple[str, ...],
    terms: tuple[tuple[int, int, tuple[int, ...]], ...],
) -> dict[str, Any]:
    return {
        "polynomial_schema_version": "1",
        "domain": "QQ",
        "variables": list(variables),
        "polynomial": {
            "terms": [
                {
                    "coefficient": {"num": str(numerator), "den": str(denominator)},
                    "exponents": list(exponents),
                }
                for numerator, denominator, exponents in terms
            ]
        },
    }


def _ideal(
    variables: tuple[str, ...],
    *generators: tuple[tuple[int, int, tuple[int, ...]], ...],
) -> dict[str, Any]:
    return {
        "variables": list(variables),
        "generators": [_polynomial(variables, generator) for generator in generators],
    }


def _op[RequestT: StrictModel, ResultT: StrictModel](
    operation_id: str,
    title: str,
    description: str,
    request_model: type[RequestT],
    result_model: type[ResultT],
    operation: Callable[[RequestT], ResultT],
    *tags: str,
    examples: tuple[OperationExample, ...] = (),
    version: str = "2",
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
        "Compute the exact radical sqrt(I) of a bounded polynomial ideal over "
        "QQ using the private Singular backend.",
        IdealRadicalRequest,
        IdealRadicalResult,
        compute_ideal_radical,
        "commutative-algebra",
        "radical",
        "exact",
        examples=(
            example(
                "ideal_xy",
                "Compute the radical of <x^2, xy> in Q[x,y]; every generator "
                "must use the same canonical ordered QQ polynomial ring.",
                {
                    "ideal": _ideal(
                        ("x", "y"),
                        ((1, 1, (2, 0)),),
                        ((1, 1, (1, 1)),),
                    ),
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
                "Check if x is in sqrt(<x^2>) in Q[x]; the ideal and "
                "polynomial must use the same canonical ordered QQ ring.",
                {
                    "ideal": _ideal(("x",), ((1, 1, (2,)),)),
                    "polynomial": _polynomial(("x",), ((1, 1, (1,)),)),
                },
            ),
        ),
    ),
    _op(
        "polynomial.ideal.quotient.compute",
        "Compute the ideal quotient (I : J)",
        "Compute the exact colon ideal (I : J) = {f : f*J subseteq I} over QQ "
        "using the private Singular backend.",
        IdealQuotientRequest,
        IdealQuotientResult,
        compute_ideal_quotient,
        "commutative-algebra",
        "ideal-quotient",
        "exact",
        examples=(
            example(
                "quotient_xy",
                "Compute (<x^2, xy> : <x>) in Q[x,y]; both ideals must use "
                "the same canonical ordered QQ polynomial ring.",
                {
                    "dividend": _ideal(
                        ("x", "y"),
                        ((1, 1, (2, 0)),),
                        ((1, 1, (1, 1)),),
                    ),
                    "divisor": _ideal(
                        ("x", "y"),
                        ((1, 1, (1, 0)),),
                    ),
                },
            ),
        ),
    ),
    _op(
        "polynomial.ideal.saturation.compute",
        "Compute ideal saturation I : <d>^infinity",
        "Compute the exact saturation I : <d>^infinity of a bounded "
        "polynomial ideal I by a single nonzero polynomial d over QQ using "
        "the private Singular backend. The result is the saturated ideal "
        "with all components supported on the zero locus of d removed.",
        IdealSaturationRequest,
        IdealSaturationResult,
        compute_ideal_saturation,
        "commutative-algebra",
        "saturation",
        "exact",
        examples=(
            example(
                "saturation_xy",
                "Compute <xy> : <x>^infinity in Q[x,y]; this equals <y>. The "
                "denominator is one nonzero polynomial in the same canonical "
                "ordered QQ polynomial ring as the ideal.",
                {
                    "ideal": _ideal(
                        ("x", "y"),
                        ((1, 1, (1, 1)),),
                    ),
                    "denominator": _polynomial(
                        ("x", "y"),
                        ((1, 1, (1, 0)),),
                    ),
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
