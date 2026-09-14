"""Public ring-of-integers value and kernel for presented simple number fields.

The kernel reuses the owner-local SymPy ``round_two`` adapter.  SymPy returns
an integral basis in the monic norm basis ``ZZ[beta]`` with ``beta = L*alpha``
for the presented leading coefficient ``L``; the kernel pulls each basis
element back to rational coordinates on the presentation's own ascending power
basis so the returned value composes with the existing field-element carrier.
"""

from __future__ import annotations

from pydantic import model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_theory.number_fields._integral_basis_process import (
    run_integral_basis_worker,
)
from jacobian.math.number_theory.number_fields._models import (
    MAX_INTEGRAL_BASIS_DEGREE,
    NumberFieldRequest,
)
from jacobian.math.number_theory.number_fields.values import (
    NumberFieldDiscriminantInteger,
    SimpleNumberFieldElement,
    SimpleNumberFieldPresentation,
)


class NumberFieldRingOfIntegersResult(StrictModel):
    """Deterministic integral-basis witness for ``O_K`` on the power basis.

    Integral bases are not mathematical singletons.  The operation's pinned
    SymPy kernel chooses one deterministic basis, while this value records the
    exact witness and the field it belongs to.
    """

    field: SimpleNumberFieldPresentation
    basis: tuple[SimpleNumberFieldElement, ...]
    field_discriminant: NumberFieldDiscriminantInteger

    @model_validator(mode="after")
    def validate_basis_shape(self) -> NumberFieldRingOfIntegersResult:
        """Keep malformed authored results from crossing the value boundary."""

        degree = self.field.degree
        if len(self.basis) != degree:
            raise PydanticCustomError(
                "number_field.ring_of_integers_basis_length",
                "an integral basis needs exactly one vector per field degree",
            )
        if self.basis[0].coefficients_ascending != _one_coordinates(degree):
            raise PydanticCustomError(
                "number_field.ring_of_integers_unit_basis",
                "every integral basis contains the unit as its first vector",
            )
        if any(element.presentation != self.field for element in self.basis):
            raise PydanticCustomError(
                "number_field.ring_of_integers_basis_field",
                "every integral basis element must belong to the result field",
            )
        if any(len(element.coefficients_ascending) != degree for element in self.basis):
            raise PydanticCustomError(
                "number_field.ring_of_integers_vector_length",
                "every integral basis vector spans the complete power basis",
            )
        return self

    def require_canonical_basis(self) -> None:
        if len(self.basis) != self.field.degree:
            raise OperationDomainValidationError(
                location=("basis",),
                code="number_field.ring_of_integers_basis_length",
                message="an integral basis needs exactly one vector per field degree",
            )
        if self.basis[0].coefficients_ascending != _one_coordinates(self.field.degree):
            raise OperationDomainValidationError(
                location=("basis",),
                code="number_field.ring_of_integers_unit_basis",
                message="every integral basis contains the unit as its first vector",
            )
        for element in self.basis:
            if element.presentation != self.field:
                raise OperationDomainValidationError(
                    location=("basis",),
                    code="number_field.ring_of_integers_basis_field",
                    message="every integral basis element must belong to the result field",
                )
            if len(element.coefficients_ascending) != self.field.degree:
                raise OperationDomainValidationError(
                    location=("basis",),
                    code="number_field.ring_of_integers_vector_length",
                    message="every integral basis vector spans the complete power basis",
                )


def _one_coordinates(degree: int) -> tuple[CanonicalRational, ...]:
    return (
        CanonicalRational(num=1, den=1),
        *(CanonicalRational(num=0, den=1) for _ in range(degree - 1)),
    )


def ring_of_integers(
    field: SimpleNumberFieldPresentation,
) -> NumberFieldRingOfIntegersResult:
    """Return the integral basis in the presentation's own power basis.

    Both the catalog operation and this package-exported native entry share one
    implementation, so the irreducibility, discriminant, and ``round_two`` work
    always runs behind the request-owned killable boundary. A native caller that
    cancels, or whose request envelope expires mid-computation, is therefore
    observed on the same path as a ``math.run`` request.
    """

    if not isinstance(field, SimpleNumberFieldPresentation):
        raise OperationDomainValidationError(
            location=("field",),
            code="number_field.ring_of_integers_field_type",
            message=(
                "ring_of_integers requires a SimpleNumberFieldPresentation "
                "field argument"
            ),
        )
    if field.degree > MAX_INTEGRAL_BASIS_DEGREE:
        raise OperationDomainValidationError(
            location=("field",),
            code="number_field.ring_of_integers_degree_bound",
            message=(
                "the ring-of-integers operation is limited to degree "
                f"{MAX_INTEGRAL_BASIS_DEGREE}"
            ),
        )
    if field.degree == 1:
        # ``SimpleNumberFieldPresentation`` presents ``x`` as QQ, whose maximal
        # order is ZZ with the single power-basis coordinate 1. SymPy's
        # round_two only accepts degree at least two, so answer this case
        # before launching the worker.
        result = NumberFieldRingOfIntegersResult(
            field=field,
            basis=(
                SimpleNumberFieldElement(
                    presentation=field,
                    coefficients_ascending=(CanonicalRational(num=1, den=1),),
                ),
            ),
            field_discriminant=1,
        )
        result.require_canonical_basis()
        return result
    worker_result = run_integral_basis_worker(
        NumberFieldRequest(field=field),
        include_basis=True,
    )
    if worker_result is None:
        raise OperationDomainValidationError(
            location=("field",),
            code="number_field.defining_polynomial_must_be_irreducible",
            message="a number field requires an irreducible defining polynomial",
        )
    if worker_result.basis is None:
        raise RuntimeError("number-field worker returned no integral basis")
    result = NumberFieldRingOfIntegersResult(
        field=field,
        basis=tuple(
            SimpleNumberFieldElement(
                presentation=field,
                coefficients_ascending=vector,
            )
            for vector in worker_result.basis
        ),
        field_discriminant=worker_result.field_discriminant,
    )
    result.require_canonical_basis()
    return result


__all__ = [
    "MAX_INTEGRAL_BASIS_DEGREE",
    "NumberFieldRingOfIntegersResult",
    "ring_of_integers",
]
