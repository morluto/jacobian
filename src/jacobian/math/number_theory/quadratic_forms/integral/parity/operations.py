"""Exact mod-two parity profiles for integral quadratic forms."""

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.number_theory.quadratic_forms.integral._models import (
    MAX_INTEGRAL_QUADRATIC_FORM_AXIS,
    MAX_INTEGRAL_QUADRATIC_FORM_TERMS,
    IntegralQuadraticForm,
)
from jacobian.math.number_theory.quadratic_forms.integral.modular.operations import (
    _reduce_integral_form_modulus_value,
)
from jacobian.math.number_theory.quadratic_forms.integral.parity._models import (
    ParityProfile,
)

MAX_PARITY_PROFILE_ALLOCATION_UNITS = (
    MAX_INTEGRAL_QUADRATIC_FORM_AXIS + MAX_INTEGRAL_QUADRATIC_FORM_TERMS + 2
)


def parity_profile(form: IntegralQuadraticForm) -> ParityProfile:
    """Return diagonal ``Q(e_i)`` and polar cross-term residues over ``F_2``."""

    if not isinstance(form, IntegralQuadraticForm):
        raise OperationDomainValidationError(
            location=("form",),
            code="quadratic_form.parity_profile.form_type",
            message="expected a canonical integral quadratic form",
        )
    if (
        not isinstance(form.axis, tuple)
        or not isinstance(form.diagonal_coefficients, tuple)
        or not isinstance(form.cross_terms, tuple)
    ):
        raise OperationDomainValidationError(
            location=("form",),
            code="quadratic_form.parity_profile.form_shape",
            message="integral quadratic form containers must be tuples",
        )
    allocation_units = (
        2 + len(form.axis) + len(form.diagonal_coefficients) + len(form.cross_terms)
    )
    if allocation_units > MAX_PARITY_PROFILE_ALLOCATION_UNITS:
        raise OperationResourceAdmissionError(
            location=("form",),
            code="quadratic_form.parity_profile.allocation_bound",
            message="the mod-two parity profile exceeds its exact-value allocation bound",
        )

    # The generic reduction kernel owns coefficient digit-work, canonical
    # residue construction, and modular polynomial output admission.
    polynomial = _reduce_integral_form_modulus_value(form, 2)
    return ParityProfile(
        polynomial=polynomial,
        all_basis_norms_even=all(
            residue == 0 for residue in polynomial.diagonal_residues
        ),
    )


__all__ = ["MAX_PARITY_PROFILE_ALLOCATION_UNITS", "parity_profile"]
