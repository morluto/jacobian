"""Supported native operations on canonical integral binary quadratic forms."""

from __future__ import annotations

from jacobian.math.number_theory.quadratic_forms.binary._kernel import (
    reduce as _reduce,
)
from jacobian.math.number_theory.quadratic_forms.binary._kernel import (
    representations as _representations,
)
from jacobian.math.number_theory.quadratic_forms.binary._models import (
    BinaryQuadraticFormRepresentation,
    PrimitivePositiveDefiniteBinaryQuadraticForm,
    _require_evaluated_value_bound,
    _require_representation_coordinate,
)


def evaluate(form: PrimitivePositiveDefiniteBinaryQuadraticForm, x: int, y: int) -> int:
    """Return the exact value ``Q(x,y)`` within the public coordinate envelope."""
    _require_representation_coordinate(x)
    _require_representation_coordinate(y)
    return _require_evaluated_value_bound(form, x, y)


def reduced_form(
    form: PrimitivePositiveDefiniteBinaryQuadraticForm,
) -> PrimitivePositiveDefiniteBinaryQuadraticForm:
    """Return the canonical Gauss-reduced representative of ``form``."""
    a, b, c, _p, _q, _r, _s = _reduce(form.a, form.b, form.c)
    return PrimitivePositiveDefiniteBinaryQuadraticForm(a=a, b=b, c=c)


def representations(
    form: PrimitivePositiveDefiniteBinaryQuadraticForm, target: int
) -> tuple[BinaryQuadraticFormRepresentation, ...]:
    """Return every ordered signed representation of ``target`` by ``form``."""
    return _representations(form, target)


__all__ = ["evaluate", "reduced_form", "representations"]
