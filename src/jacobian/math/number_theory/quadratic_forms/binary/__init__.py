"""Supported native APIs for integral binary quadratic forms."""

from jacobian.math.number_theory.quadratic_forms.binary._models import (
    BinaryQuadraticFormClassCompositionResult,
    BinaryQuadraticFormRepresentation,
    PrimitivePositiveDefiniteBinaryQuadraticForm,
    ProperBinaryQuadraticFormClass,
)
from jacobian.math.number_theory.quadratic_forms.binary.operations import (
    check,
    compose_classes,
    evaluate,
    proper_equivalence,
    reduced_classes,
    reduced_form,
    representations,
)

__all__ = [
    "BinaryQuadraticFormClassCompositionResult",
    "BinaryQuadraticFormRepresentation",
    "PrimitivePositiveDefiniteBinaryQuadraticForm",
    "ProperBinaryQuadraticFormClass",
    "check",
    "compose_classes",
    "evaluate",
    "proper_equivalence",
    "reduced_classes",
    "reduced_form",
    "representations",
]
