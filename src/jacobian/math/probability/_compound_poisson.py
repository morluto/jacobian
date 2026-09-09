"""Exact compound-Poisson cumulant prefixes."""

from fractions import Fraction
from typing import Self

from pydantic import Field, StrictInt, model_validator

from jacobian._exact import CanonicalRational, require_bounded_rational
from jacobian._models import StrictModel
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.probability._distribution import (
    FiniteRationalDistribution,
    require_input_distribution,
)
from jacobian.math.probability._models import (
    MAX_RESULT_RATIONAL_DIGITS,
    _require_bounded_fraction,
)
from jacobian.math.probability.operations import _plan_raw_moment


class CompoundPoissonCumulantRequest(StrictModel):
    intensity: CanonicalRational
    jump_distribution: FiniteRationalDistribution
    max_order: StrictInt = Field(ge=0, le=128)

    @model_validator(mode="after")
    def require_nonnegative_intensity(self) -> Self:
        require_bounded_rational(
            self.intensity, max_digits=256, label="compound-Poisson intensity"
        )
        if self.intensity.num < 0:
            raise ValueError("compound-Poisson intensity must be nonnegative")
        require_input_distribution(
            self.jump_distribution.atoms, require_canonical=True, max_digits=256
        )
        return self


class CompoundPoissonCumulantRow(StrictModel):
    order: StrictInt = Field(ge=1, le=128)
    jump_raw_moment: CanonicalRational
    cumulant: CanonicalRational


class CompoundPoissonCumulantResult(StrictModel):
    source: CompoundPoissonCumulantRequest
    cumulants: tuple[CompoundPoissonCumulantRow, ...] = Field(max_length=128)


def compound_poisson_cumulant_prefix(
    request: CompoundPoissonCumulantRequest,
) -> CompoundPoissonCumulantResult:
    atoms = request.jump_distribution.atoms
    intensity = request.intensity.as_fraction()
    for order in range(1, request.max_order + 1):
        moment = _plan_raw_moment(atoms, order).total
        try:
            _require_bounded_fraction(
                intensity * moment,
                max_digits=MAX_RESULT_RATIONAL_DIGITS,
                label="compound-Poisson cumulant",
            )
        except ValueError as exc:
            raise OperationResourceAdmissionError(
                location=("intensity",),
                code="probability.compound_poisson.cumulant_height_bound",
                message=str(exc),
            ) from exc
    powers = [Fraction(1) for _ in atoms]
    values = [atom.value.as_fraction() for atom in atoms]
    probabilities = [atom.probability.as_fraction() for atom in atoms]
    rows = []
    for order in range(1, request.max_order + 1):
        for index, value in enumerate(values):
            powers[index] *= value
        moment = sum(
            (
                probability * power
                for probability, power in zip(probabilities, powers, strict=True)
            ),
            Fraction(),
        )
        cumulant = intensity * moment
        rows.append(
            CompoundPoissonCumulantRow(
                order=order,
                jump_raw_moment=CanonicalRational.from_fraction(moment),
                cumulant=CanonicalRational.from_fraction(cumulant),
            )
        )
    return CompoundPoissonCumulantResult(source=request, cumulants=tuple(rows))


__all__ = ["compound_poisson_cumulant_prefix"]
