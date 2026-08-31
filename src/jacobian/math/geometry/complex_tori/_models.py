"""Canonical values and requests for exact lattice-presented complex tori."""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable, Mapping
from typing import Annotated, Any, Literal, Self, cast

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalInteger
from jacobian._models import StrictModel, canonicalize_json_containers
from jacobian.math._labels import OpaqueLabel
from jacobian.math.lattices.invariant_forms._models import (
    MAX_ACTION_DIMENSION,
    IntegralBilinearForm,
)
from jacobian.math.matrices.analysis._models import InertiaResult
from jacobian.math.matrices.values import (
    EmbeddedRealSimpleNumberFieldMatrix,
    ExactRealMatrix,
    RationalMatrix,
    SmithNormalForm,
)

HermitianDefiniteness = Literal[
    "positive_definite",
    "positive_semidefinite",
    "negative_definite",
    "negative_semidefinite",
    "zero",
    "indefinite",
]


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"complex_torus.{reason}", message)


class LatticeComplexStructure(StrictModel):
    """A labelled lattice together with a candidate exact real complex structure.

    The ordered axis presents ``Lambda = ZZ^(2g)``.  Mathematical consumers
    recognize ``J^2 = -I`` once during operation admission.
    """

    coordinate_axis: tuple[OpaqueLabel, ...] = Field(
        min_length=2,
        max_length=MAX_ACTION_DIMENSION,
    )
    complex_structure: ExactRealMatrix

    @model_validator(mode="before")
    @classmethod
    def require_raw_axis_envelope(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        axis = data.get("coordinate_axis")
        if axis is not None and not isinstance(
            axis, (list, tuple, str, bytes, Mapping)
        ):
            try:
                axis_iterator = iter(cast(Iterable[object], axis))
            except TypeError:
                pass
            else:
                axis_values: list[object] = []
                for _index in range(MAX_ACTION_DIMENSION + 1):
                    try:
                        axis_values.append(next(axis_iterator))
                    except StopIteration:
                        break
                else:
                    raise _validation_error(
                        "budget_exceeded",
                        f"a complex-torus lattice has at most {MAX_ACTION_DIMENSION} axes",
                    )
                axis = tuple(axis_values)
        if isinstance(axis, (list, tuple)) and len(axis) > MAX_ACTION_DIMENSION:
            raise _validation_error(
                "budget_exceeded",
                f"a complex-torus lattice has at most {MAX_ACTION_DIMENSION} axes",
            )
        if isinstance(axis, (list, tuple)):
            for label in axis:
                if not isinstance(label, str):
                    raise _validation_error(
                        "invalid_coordinate_label",
                        "coordinate_axis labels must be strings",
                    )
                if unicodedata.normalize("NFC", label) != label:
                    raise _validation_error(
                        "noncanonical_coordinate_label",
                        "coordinate_axis labels must use NFC Unicode normalization",
                    )
        normalized = dict(data)
        if isinstance(axis, list):
            normalized["coordinate_axis"] = tuple(axis)
        elif isinstance(axis, tuple):
            normalized["coordinate_axis"] = axis
        # Canonicalize only the bounded outer containers; leave the nested
        # complex_structure matrix for its owning model validator.
        for key in ("coordinate_axis",):
            normalized[key] = canonicalize_json_containers(normalized.get(key))
        return normalized

    @model_validator(mode="after")
    def require_even_common_axis(self) -> Self:
        dimension = len(self.coordinate_axis)
        if dimension % 2:
            raise _validation_error(
                "odd_lattice_rank",
                "a complex-torus lattice must have even positive rank",
            )
        if len(set(self.coordinate_axis)) != dimension:
            raise _validation_error(
                "duplicate_coordinate_label",
                "complex-torus coordinate labels must be pairwise distinct",
            )
        entries = self.complex_structure.entries
        if len(entries) != dimension or any(len(row) != dimension for row in entries):
            raise _validation_error(
                "complex_structure_shape",
                "the complex-structure matrix must be square on coordinate_axis",
            )
        return self

    @property
    def complex_dimension(self) -> int:
        return len(self.coordinate_axis) // 2


class NeronSeveriLatticeRequest(StrictModel):
    """One exact lattice complex structure whose integral Hodge lattice is sought."""

    torus: LatticeComplexStructure


class RiemannFormProfileRequest(StrictModel):
    """One exact complex torus and one selected integral alternating form."""

    torus: LatticeComplexStructure
    form: IntegralBilinearForm


class HermitianInertia(StrictModel):
    """Complex inertia of a Hermitian form."""

    n_positive: int = Field(ge=0, le=MAX_ACTION_DIMENSION // 2)
    n_negative: int = Field(ge=0, le=MAX_ACTION_DIMENSION // 2)
    n_zero: int = Field(ge=0, le=MAX_ACTION_DIMENSION // 2)
    definiteness: HermitianDefiniteness

    @model_validator(mode="after")
    def require_definiteness_label(self) -> Self:
        if self.n_positive == 0 and self.n_negative == 0:
            expected = "zero"
        elif self.n_zero == 0:
            if self.n_negative == 0:
                expected = "positive_definite"
            elif self.n_positive == 0:
                expected = "negative_definite"
            else:
                expected = "indefinite"
        elif self.n_negative == 0:
            expected = "positive_semidefinite"
        elif self.n_positive == 0:
            expected = "negative_semidefinite"
        else:
            expected = "indefinite"
        if self.definiteness != expected:
            raise _validation_error(
                "hermitian_inertia",
                "Hermitian definiteness must agree with its inertia counts",
            )
        return self


class RiemannFormNotHodge(StrictModel):
    """The selected alternating form is not of Hodge type ``(1,1)``."""

    status: Literal["NOT_HODGE"]
    is_riemann_form: Literal[False] = False


class _RiemannFormHodgeProfile(StrictModel):
    """Common exact profile of one Hodge ``(1,1)`` alternating form."""

    associated_form_inertia: InertiaResult
    hermitian_inertia: HermitianInertia
    associated_form_convention: Literal["J_TRANSPOSE_TIMES_E"] = "J_TRANSPOSE_TIMES_E"
    hermitian_form_convention: Literal["G_PLUS_I_E_LINEAR_IN_FIRST"] = (
        "G_PLUS_I_E_LINEAR_IN_FIRST"
    )


class RiemannFormHodgeNonPositive(_RiemannFormHodgeProfile):
    """A Hodge ``(1,1)`` form whose associated real form is not positive."""

    status: Literal["HODGE_NON_POSITIVE"]
    is_riemann_form: Literal[False] = False


class RiemannFormPositive(_RiemannFormHodgeProfile):
    """A positive Riemann form and its required polarization type."""

    status: Literal["RIEMANN_FORM"]
    is_riemann_form: Literal[True] = True
    polarization_type: tuple[CanonicalInteger, ...]


RiemannFormOutcome = Annotated[
    RiemannFormNotHodge | RiemannFormHodgeNonPositive | RiemannFormPositive,
    Field(discriminator="status"),
]


def _require_associated_scalar_domain(
    torus: LatticeComplexStructure,
    inertia: InertiaResult,
) -> None:
    complex_structure = torus.complex_structure
    if isinstance(complex_structure, RationalMatrix):
        if not isinstance(inertia.matrix, RationalMatrix):
            raise _validation_error(
                "associated_scalar_domain",
                "the associated form must retain the torus's rational domain",
            )
        return
    if (
        not isinstance(inertia.matrix, EmbeddedRealSimpleNumberFieldMatrix)
        or inertia.matrix.embedding != complex_structure.embedding
    ):
        raise _validation_error(
            "associated_scalar_domain",
            "the associated form must retain the torus's identical selected "
            "real embedding",
        )


class RiemannFormProfile(StrictModel):
    """A source-bound exact profile of one selected integral alternating form."""

    torus: LatticeComplexStructure
    form: IntegralBilinearForm
    smith_normal_form: SmithNormalForm
    alternating_elementary_divisors: tuple[CanonicalInteger, ...]
    is_degenerate: bool
    outcome: RiemannFormOutcome

    @model_validator(mode="after")
    def require_source_bound_profile(self) -> Self:
        dimension = len(self.torus.coordinate_axis)
        if self.form.coordinate_axis != self.torus.coordinate_axis:
            raise _validation_error(
                "form_axis",
                "the selected form must use the complex torus coordinate axis",
            )
        if self.form.kind != "ALTERNATING":
            raise _validation_error(
                "form_kind", "a Riemann-form profile requires an alternating form"
            )
        normal_form = self.smith_normal_form.normal_form.entries
        if len(normal_form) != dimension or any(
            len(row) != dimension for row in normal_form
        ):
            raise _validation_error(
                "smith_source",
                "the Smith normal form must have the selected form's shape",
            )
        factors = self.smith_normal_form.invariant_factors
        if len(factors) % 2 or any(
            factors[index] != factors[index + 1] for index in range(0, len(factors), 2)
        ):
            raise _validation_error(
                "alternating_smith_pairs",
                "nonzero Smith factors of an alternating form must occur in pairs",
            )
        expected_elementary_divisors = factors[::2]
        if self.alternating_elementary_divisors != expected_elementary_divisors:
            raise _validation_error(
                "alternating_type",
                "alternating elementary divisors must select one factor per pair",
            )
        expected_degenerate = self.smith_normal_form.rank < dimension
        if self.is_degenerate != expected_degenerate:
            raise _validation_error(
                "degeneracy", "degeneracy must agree with the Smith rank"
            )
        if isinstance(self.outcome, RiemannFormNotHodge):
            return self

        inertia = self.outcome.associated_form_inertia
        _require_associated_scalar_domain(self.torus, inertia)
        real_counts = (inertia.n_positive, inertia.n_negative, inertia.n_zero)
        if any(count % 2 for count in real_counts):
            raise _validation_error(
                "hermitian_inertia",
                "a compatible real form must have even inertia counts",
            )
        hermitian_counts = (
            self.outcome.hermitian_inertia.n_positive,
            self.outcome.hermitian_inertia.n_negative,
            self.outcome.hermitian_inertia.n_zero,
        )
        if tuple(count // 2 for count in real_counts) != hermitian_counts:
            raise _validation_error(
                "hermitian_inertia",
                "Hermitian inertia must halve the associated real inertia",
            )
        if sum(hermitian_counts) != self.torus.complex_dimension:
            raise _validation_error(
                "hermitian_dimension",
                "Hermitian inertia must sum to the complex dimension",
            )
        positive = inertia.n_positive == dimension
        if isinstance(self.outcome, RiemannFormPositive) != positive:
            raise _validation_error(
                "riemann_positivity",
                "Riemann-form status must agree with positive definiteness of J^T E",
            )
        if isinstance(self.outcome, RiemannFormPositive) and (
            self.is_degenerate
            or self.smith_normal_form.rank != dimension
            or len(self.outcome.polarization_type) != self.torus.complex_dimension
        ):
            raise _validation_error(
                "riemann_nondegenerate",
                "a positive Riemann form must have full Smith rank and one "
                "polarization divisor per complex dimension",
            )
        if isinstance(self.outcome, RiemannFormPositive) and (
            self.outcome.polarization_type != self.alternating_elementary_divisors
        ):
            raise _validation_error(
                "polarization_type",
                "a positive Riemann form's polarization type must equal its "
                "alternating elementary divisors",
            )
        return self


__all__ = [
    "HermitianInertia",
    "LatticeComplexStructure",
    "NeronSeveriLatticeRequest",
    "RiemannFormHodgeNonPositive",
    "RiemannFormNotHodge",
    "RiemannFormOutcome",
    "RiemannFormPositive",
    "RiemannFormProfile",
    "RiemannFormProfileRequest",
]
