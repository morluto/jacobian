"""Canonical values and requests for exact lattice-presented complex tori."""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable, Mapping
from typing import Annotated, Any, Literal, Self, cast

from pydantic import ConfigDict, Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import ExactInteger
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
    polarization_type: tuple[ExactInteger, ...]


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
    alternating_elementary_divisors: tuple[ExactInteger, ...]
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
    "PolarizationSearchRequest",
    "PolarizationSearchResult",
    "RiemannFormHodgeNonPositive",
    "RiemannFormNotHodge",
    "RiemannFormOutcome",
    "RiemannFormPositive",
    "RiemannFormProfile",
    "RiemannFormProfileRequest",
]


MAX_POLARIZATION_COEFFICIENT = 8
MAX_POLARIZATION_EXAMINED = 5_000


class PolarizationSearchRequest(StrictModel):
    """Search the Neron-Severi lattice for a polarization by bounded search."""

    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "Bounded polarization search over an exact complex torus. "
                "Integer combinations of the saturated Neron-Severi basis "
                "inside the coefficient box are profiled in deterministic "
                "order; at most examination_budget profiles run before "
                "reporting UNKNOWN."
            )
        }
    )

    torus: LatticeComplexStructure = Field(
        description="Exact complex torus with J^2 = -I on its coordinate axis.",
    )
    coefficient_bound: int = Field(
        ge=1,
        le=MAX_POLARIZATION_COEFFICIENT,
        description="Search integer coefficients in [-bound, bound] per basis form.",
    )
    examination_budget: int = Field(
        ge=1,
        le=MAX_POLARIZATION_EXAMINED,
        description="Profile at most this many classes before reporting UNKNOWN.",
    )


class PolarizationSearchResult(StrictModel):
    """A bounded polarization-search outcome with its receipt.

    - ``FOUND``: ``form`` is an integral Hodge class whose associated real
      form is positive definite; ``profile`` is its Riemann-form profile.
    - ``INFEASIBLE``: no polarization exists, certified exactly -- either
      the Neron-Severi lattice is zero, or it has rank one and neither sign
      of its primitive generator is positive definite (scaling covers every
      nonzero class).
    - ``UNKNOWN``: the coefficient box was exhausted without a polarization
      (inconclusive beyond the box), or the examination budget ran out at
      ``current_coefficients``.
    """

    torus: LatticeComplexStructure
    coefficient_bound: int = Field(ge=1, le=MAX_POLARIZATION_COEFFICIENT)
    examination_budget: int = Field(ge=1, le=MAX_POLARIZATION_EXAMINED)
    status: Literal["FOUND", "INFEASIBLE", "UNKNOWN"]
    ns_rank: int = Field(ge=0)
    form: IntegralBilinearForm | None = None
    profile: RiemannFormProfile | None = None
    examined: int = Field(default=0, ge=0)
    total: int = Field(default=0, ge=0)
    current_coefficients: tuple[int, ...] | None = None
    infeasibility_reason: (
        Literal["NS_RANK_ZERO", "RANK_ONE_NO_DEFINITE_SIGN"] | None
    ) = None
    unknown_reason: (
        Literal["COEFFICIENT_BOX_EXHAUSTED", "EXAMINATION_BUDGET_EXCEEDED"] | None
    ) = None

    @model_validator(mode="after")
    def require_polarization_payload(self) -> Self:
        if (self.infeasibility_reason is None) == (self.status == "INFEASIBLE"):
            raise _validation_error(
                "polarization_infeasibility_payload",
                "INFEASIBLE carries an infeasibility reason and nothing else does",
            )
        if (self.unknown_reason is None) == (self.status == "UNKNOWN"):
            raise _validation_error(
                "polarization_unknown_payload",
                "UNKNOWN carries an unknown reason and nothing else does",
            )
        if self.status == "FOUND":
            if self.form is None or self.profile is None:
                raise _validation_error(
                    "polarization_found_payload",
                    "a found search carries its class and profile",
                )
            if self.ns_rank < 1:
                raise _validation_error(
                    "polarization_found_rank",
                    "a found search has a nonzero Neron-Severi lattice",
                )
        elif self.form is not None or self.profile is not None:
            raise _validation_error(
                "polarization_undecided_payload",
                "an unfinished search carries no class or profile",
            )
        if (self.current_coefficients is None) != (
            self.unknown_reason != "EXAMINATION_BUDGET_EXCEEDED"
        ):
            raise _validation_error(
                "polarization_current_payload",
                "only a budget-exhausted search carries its next coefficients",
            )
        return self

    @model_validator(mode="after")
    def require_polarization_certificate(self) -> Self:
        if self.status != "FOUND":
            return self
        assert self.form is not None and self.profile is not None
        if self.profile.form != self.form or self.profile.torus != self.torus:
            raise _validation_error(
                "polarization_certificate_binding",
                "the profile must classify the found class on the searched torus",
            )
        if self.profile.outcome.status != "RIEMANN_FORM":
            raise _validation_error(
                "polarization_certificate_status",
                "the found class must profile as a Riemann form",
            )
        return self

    @model_validator(mode="after")
    def require_polarization_receipt(self) -> Self:
        if self.status == "FOUND":
            if self.examined < 1 or self.examined > self.total:
                raise _validation_error(
                    "polarization_found_receipt",
                    "a found search examined its witness within the box",
                )
        elif self.status == "INFEASIBLE":
            if self.infeasibility_reason == "NS_RANK_ZERO":
                if self.ns_rank != 0 or self.examined != 0 or self.total != 1:
                    raise _validation_error(
                        "polarization_rank_zero_receipt",
                        "a rank-zero search examines nothing over one empty box",
                    )
            elif self.examined != 2 or self.ns_rank != 1:
                raise _validation_error(
                    "polarization_rank_one_receipt",
                    "a rank-one search examines both signs of its generator",
                )
        elif self.unknown_reason == "COEFFICIENT_BOX_EXHAUSTED":
            if self.examined != self.total - 1:
                raise _validation_error(
                    "polarization_box_receipt",
                    "a box-exhausted search examined every nonzero combination",
                )
        elif self.examined != self.examination_budget:
            raise _validation_error(
                "polarization_budget_receipt",
                "a budget-exhausted search spent its full budget",
            )
        elif self.total - 1 <= self.examination_budget:
            raise _validation_error(
                "polarization_budget_scope",
                "a budget-exhausted search left combinations unexamined",
            )
        return self

    @classmethod
    def _from_kernel(cls, **values: Any) -> Self:
        return cls.model_construct(**values)
