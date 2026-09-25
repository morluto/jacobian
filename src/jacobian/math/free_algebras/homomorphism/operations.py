"""Bounded exact evaluation of free associative algebra homomorphisms."""

from jacobian.math.free_algebras._models import (
    FreeAlgebraPolynomial,
    FreeAlgebraPolynomialHomomorphism,
)
from jacobian.math.free_algebras.operations import substitute_polynomial


def apply(
    homomorphism: FreeAlgebraPolynomialHomomorphism,
    polynomial: FreeAlgebraPolynomial,
) -> FreeAlgebraPolynomial:
    """Apply the canonical QQ-algebra map using the admitted substitution kernel."""

    return substitute_polynomial(homomorphism, polynomial)
