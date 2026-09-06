"""Supported exact polynomial-interpolation API."""

from jacobian.math.polynomials.interpolation._models import (
    DividedDifferencesResult,
    HermiteConstraintReplay,
    HermiteInterpolationResult,
    OrdinaryDerivativeJet,
    OrdinaryDerivativeJetTable,
    OrdinaryDerivativeValue,
)
from jacobian.math.polynomials.interpolation.operations import (
    divided_differences,
    evaluate_newton,
    hermite_interpolation,
    newton_form,
    verify_divided_differences,
    verify_hermite_interpolation,
    verify_newton_evaluation,
)

__all__ = [
    "DividedDifferencesResult",
    "HermiteConstraintReplay",
    "HermiteInterpolationResult",
    "OrdinaryDerivativeJet",
    "OrdinaryDerivativeJetTable",
    "OrdinaryDerivativeValue",
    "divided_differences",
    "evaluate_newton",
    "hermite_interpolation",
    "newton_form",
    "verify_divided_differences",
    "verify_hermite_interpolation",
    "verify_newton_evaluation",
]
