"""Small typed constructors used by modular-form tests."""

from __future__ import annotations

from typing import Literal, cast

from jacobian._exact import CanonicalRational
from jacobian.math.matrices.cyclic_linear._models import RationalCyclotomicElement

BasisId = Literal[
    "level-one-e4-e6-monomials-v1",
    "gamma0-two-weight-2-4-monomials-v1",
    "gamma0-three-weight-2-4-6-hypersurface-v1",
    "gamma0-four-weight-2-generators-v1",
    "gamma0-four-chi4-weight-one-v1",
    "gamma0-four-chi4-weight-three-v1",
    "gamma0-rational-gamma0-sturm-rref-v1",
    "gamma0-13-even-order6-character-sturm-v1",
    "gamma0-cyclotomic-character-sturm-rref-v1",
]
RationalBasisId = Literal[
    "level-one-e4-e6-monomials-v1",
    "gamma0-two-weight-2-4-monomials-v1",
    "gamma0-three-weight-2-4-6-hypersurface-v1",
    "gamma0-four-weight-2-generators-v1",
    "gamma0-four-chi4-weight-one-v1",
    "gamma0-four-chi4-weight-three-v1",
    "gamma0-rational-gamma0-sturm-rref-v1",
]


def basis_id(value: str) -> BasisId:
    allowed: tuple[str, ...] = (
        "level-one-e4-e6-monomials-v1",
        "gamma0-two-weight-2-4-monomials-v1",
        "gamma0-three-weight-2-4-6-hypersurface-v1",
        "gamma0-four-weight-2-generators-v1",
        "gamma0-four-chi4-weight-one-v1",
        "gamma0-four-chi4-weight-three-v1",
        "gamma0-rational-gamma0-sturm-rref-v1",
        "gamma0-13-even-order6-character-sturm-v1",
        "gamma0-cyclotomic-character-sturm-rref-v1",
    )
    if value not in allowed:
        raise AssertionError(f"unsupported modular-form basis id: {value}")
    return cast(BasisId, value)


def rational_basis_id(value: str) -> RationalBasisId:
    if value not in (
        "level-one-e4-e6-monomials-v1",
        "gamma0-two-weight-2-4-monomials-v1",
        "gamma0-three-weight-2-4-6-hypersurface-v1",
        "gamma0-four-weight-2-generators-v1",
        "gamma0-four-chi4-weight-one-v1",
        "gamma0-four-chi4-weight-three-v1",
        "gamma0-rational-gamma0-sturm-rref-v1",
    ):
        raise AssertionError(f"unsupported rational modular-form basis id: {value}")
    return cast(RationalBasisId, value)


def rational(value: CanonicalRational | RationalCyclotomicElement) -> CanonicalRational:
    if not isinstance(value, CanonicalRational):
        raise AssertionError("expected a rational modular-form coordinate")
    return value


def cyclotomic(
    value: CanonicalRational | RationalCyclotomicElement,
) -> RationalCyclotomicElement:
    if not isinstance(value, RationalCyclotomicElement):
        raise AssertionError("expected a cyclotomic modular-form coordinate")
    return value
