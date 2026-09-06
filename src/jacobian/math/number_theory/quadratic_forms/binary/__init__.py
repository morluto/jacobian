"""Supported native APIs for integral binary quadratic forms."""

from jacobian.math.number_theory.quadratic_forms.binary._models import (
    BinaryQuadraticFormClassCompositionResult,
    BinaryQuadraticFormRepresentation,
    PrimitivePositiveDefiniteBinaryQuadraticForm,
    ProperBinaryQuadraticFormClass,
    ProperFormChangeOfVariables,
)
from jacobian.math.number_theory.quadratic_forms.binary.operations import (
    check,
    compose_classes,
    evaluate,
    proper_equivalence,
    reduced_classes,
    reduced_form,
    representations,
    verify_change_of_variables,
    verify_proper_equivalence,
    verify_reduction,
)

__all__ = [
    "BinaryQuadraticFormClassCompositionResult",
    "BinaryQuadraticFormRepresentation",
    "PrimitivePositiveDefiniteBinaryQuadraticForm",
    "ProperBinaryQuadraticFormClass",
    "ProperFormChangeOfVariables",
    "check",
    "compose_classes",
    "evaluate",
    "proper_equivalence",
    "reduced_classes",
    "reduced_form",
    "representations",
    "verify_change_of_variables",
    "verify_proper_equivalence",
    "verify_reduction",
]
