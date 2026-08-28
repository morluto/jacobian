"""Supported native operations on canonical integral binary quadratic forms."""

from __future__ import annotations

from jacobian.math.number_theory.quadratic_forms.binary._models import (
    MAX_REPRESENTATION_TARGET,
    BinaryQuadraticFormRepresentation,
    PrimitivePositiveDefiniteBinaryQuadraticForm,
    _require_evaluated_value_bound,
    _require_representation_budget,
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
    from jacobian.math.number_theory.quadratic_forms.binary._operations import _reduce

    a, b, c, _p, _q, _r, _s = _reduce(form.a, form.b, form.c)
    return PrimitivePositiveDefiniteBinaryQuadraticForm(a=a, b=b, c=c)


def representations(
    form: PrimitivePositiveDefiniteBinaryQuadraticForm, target: int
) -> tuple[BinaryQuadraticFormRepresentation, ...]:
    """Return every ordered signed representation of ``target`` by ``form``."""
    from jacobian.math.number_theory.quadratic_forms.binary._operations import (
        _enumerate_representations,
    )

    if not 0 <= target <= MAX_REPRESENTATION_TARGET:
        raise ValueError(f"target must be between 0 and {MAX_REPRESENTATION_TARGET}")
    _require_representation_budget(form, target)
    return _enumerate_representations(form, target)


__all__ = ["evaluate", "reduced_form", "representations"]
