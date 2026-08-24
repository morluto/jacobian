"""Supported native operations on canonical integral binary quadratic forms."""

from __future__ import annotations

from jacobian.math.integral_binary_quadratic_forms._models import (
    BinaryQuadraticFormEvaluateRequest,
    BinaryQuadraticFormReduceRequest,
    BinaryQuadraticFormRepresentation,
    BinaryQuadraticFormRepresentationsRequest,
    PrimitivePositiveDefiniteBinaryQuadraticForm,
)


def evaluate(form: PrimitivePositiveDefiniteBinaryQuadraticForm, x: int, y: int) -> int:
    """Return the exact value ``Q(x,y)`` within the public coordinate envelope."""
    from jacobian.math.integral_binary_quadratic_forms._operations import _evaluate

    request = BinaryQuadraticFormEvaluateRequest(form=form, x=x, y=y)
    return _evaluate(
        request.form.a, request.form.b, request.form.c, request.x, request.y
    )


def reduced_form(
    form: PrimitivePositiveDefiniteBinaryQuadraticForm,
) -> PrimitivePositiveDefiniteBinaryQuadraticForm:
    """Return the canonical Gauss-reduced representative of ``form``."""
    from jacobian.math.integral_binary_quadratic_forms._operations import _reduce

    request = BinaryQuadraticFormReduceRequest(form=form)
    a, b, c, _p, _q, _r, _s = _reduce(request.form.a, request.form.b, request.form.c)
    return PrimitivePositiveDefiniteBinaryQuadraticForm(a=a, b=b, c=c)


def representations(
    form: PrimitivePositiveDefiniteBinaryQuadraticForm, target: int
) -> tuple[BinaryQuadraticFormRepresentation, ...]:
    """Return every ordered signed representation of ``target`` by ``form``."""
    from jacobian.math.integral_binary_quadratic_forms._operations import (
        _enumerate_representations,
    )

    request = BinaryQuadraticFormRepresentationsRequest(form=form, target=target)
    return _enumerate_representations(request.form, request.target)


__all__ = ["evaluate", "reduced_form", "representations"]
