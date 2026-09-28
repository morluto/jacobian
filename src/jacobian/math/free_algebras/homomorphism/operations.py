"""Bounded exact evaluation of free associative algebra homomorphisms."""

from jacobian.catalog.models import OperationResourceAdmissionError
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

    try:
        return substitute_polynomial(homomorphism, polynomial)
    except OperationResourceAdmissionError as exc:
        diagnostic = exc.errors()[0]
        location = diagnostic["loc"]
        if location[:1] != ("substitution",):
            raise
        raise OperationResourceAdmissionError(
            location=("homomorphism", *location[1:]),
            code=diagnostic["type"],
            message=diagnostic["msg"],
        ) from exc
