"""Typed contracts for exact projective plane-curve singular loci."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Any, Literal, Self

from pydantic import Field, StrictInt, StringConstraints, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel, canonicalize_json_containers
from jacobian.math.geometry.projective.values import AlgebraicProjectivePlanePoint
from jacobian.math.number_theory.number_fields.values import SimpleNumberFieldElement
from jacobian.math.polynomials.values import (
    PolynomialVariable,
    RationalPolynomial,
    RationalPolynomialIdeal,
    require_polynomial_budget,
)

MAX_PROJECTIVE_PLANE_CURVE_DEGREE = 3
MAX_PROJECTIVE_PLANE_CURVE_TERMS = 10
MAX_PROJECTIVE_PLANE_CURVE_COEFFICIENT_DIGITS = 16
MAX_PROJECTIVE_SINGULAR_POINTS = 4
MAX_PROJECTIVE_SINGULAR_FIELD_DEGREE = 4
MAX_PROJECTIVE_SINGULAR_COMPONENTS = 16
MAX_PROJECTIVE_SINGULARITY_WALL_SECONDS = 60


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"projective_plane_curve.{reason}", message)


class ProjectivePlaneCurveSingularityBudget(StrictModel):
    """The one request-scoped wall deadline for all exact backend phases."""

    wall_seconds: StrictInt = Field(
        default=30,
        ge=1,
        le=MAX_PROJECTIVE_SINGULARITY_WALL_SECONDS,
        description=(
            "One shared deadline, in seconds, covering saturation, prime "
            "decomposition, exact point construction, and serialization."
        ),
    )


class ProjectivePlaneCurveSingularityRequest(StrictModel):
    """One bounded nonzero homogeneous ternary polynomial over ``QQ``."""

    polynomial: RationalPolynomial = Field(
        description=(
            "A nonzero homogeneous polynomial in exactly three ordered variables. "
            "The first envelope admits total degree 1 through 3, at most 10 terms, "
            "and at most 16 digits in each rational coefficient component."
        )
    )
    resource_budget: ProjectivePlaneCurveSingularityBudget = Field(
        default_factory=ProjectivePlaneCurveSingularityBudget
    )

    @model_validator(mode="before")
    @classmethod
    def require_raw_term_bound(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data
        polynomial = data.get("polynomial")
        if isinstance(polynomial, Mapping):
            sparse = polynomial.get("polynomial")
            if isinstance(sparse, Mapping):
                terms = sparse.get("terms")
                if isinstance(terms, (list, tuple)) and len(terms) > (
                    MAX_PROJECTIVE_PLANE_CURVE_TERMS
                ):
                    raise _validation_error(
                        "term_bound",
                        "projective plane-curve source exceeds the 10-term envelope",
                    )
        return canonicalize_json_containers(data)

    @model_validator(mode="after")
    def require_backend_domain(self) -> Self:
        try:
            require_polynomial_budget(
                self.polynomial,
                maximum_terms=MAX_PROJECTIVE_PLANE_CURVE_TERMS,
                maximum_exponent=MAX_PROJECTIVE_PLANE_CURVE_DEGREE,
                maximum_coefficient_digits=(
                    MAX_PROJECTIVE_PLANE_CURVE_COEFFICIENT_DIGITS
                ),
                label="projective plane-curve source",
            )
        except ValueError as exc:
            raise _validation_error("source_bound", str(exc)) from exc
        if len(self.polynomial.variables) != 3:
            raise _validation_error(
                "axis",
                "a projective plane curve requires exactly three ordered variables",
            )
        terms = self.polynomial.polynomial.terms
        if not terms:
            raise _validation_error(
                "zero_source", "the zero polynomial does not define a plane curve"
            )
        total_degrees = {sum(term.exponents) for term in terms}
        if len(total_degrees) != 1:
            raise _validation_error(
                "homogeneity", "the projective plane-curve source must be homogeneous"
            )
        degree = next(iter(total_degrees))
        if not 1 <= degree <= MAX_PROJECTIVE_PLANE_CURVE_DEGREE:
            raise _validation_error(
                "degree_bound",
                "the first projective singularity envelope admits degree 1 through 3",
            )
        return self


class ProjectivePlaneCurveFirstJet(StrictModel):
    """The exact value and ordered first partials at one algebraic point."""

    value: SimpleNumberFieldElement
    gradient: tuple[
        SimpleNumberFieldElement,
        SimpleNumberFieldElement,
        SimpleNumberFieldElement,
    ]

    @model_validator(mode="after")
    def require_one_field(self) -> Self:
        presentation = self.value.presentation
        entries = (self.value, *self.gradient)
        if any(entry.presentation != presentation for entry in entries):
            raise _validation_error(
                "first_jet_field",
                "a projective first jet must use one number-field presentation",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        value: SimpleNumberFieldElement,
        gradient: tuple[
            SimpleNumberFieldElement,
            SimpleNumberFieldElement,
            SimpleNumberFieldElement,
        ],
    ) -> Self:
        """Construct the exact zero jet established by point construction."""

        return cls.model_construct(value=value, gradient=gradient)


class ProjectivePlaneCurveSingularPointRecord(StrictModel):
    """One normalized geometric singular point and its exact zero first jet."""

    point: AlgebraicProjectivePlanePoint
    first_jet: ProjectivePlaneCurveFirstJet

    @model_validator(mode="after")
    def bind_point_and_jet(self) -> Self:
        if self.first_jet.value.presentation != self.point.embedding.presentation:
            raise _validation_error(
                "point_jet_field",
                "a singular point and its first jet must share one field presentation",
            )
        return self


class SmoothProjectivePlaneCurve(StrictModel):
    status: Literal["SMOOTH_OVER_ALGEBRAIC_CLOSURE"]
    saturated_jacobian_ideal: RationalPolynomialIdeal

    @classmethod
    def _from_kernel(cls, ideal: RationalPolynomialIdeal) -> Self:
        """Construct the smooth branch after the kernel established ``<1>``."""

        return cls.model_construct(
            status="SMOOTH_OVER_ALGEBRAIC_CLOSURE",
            saturated_jacobian_ideal=ideal,
        )


class ZeroDimensionalProjectivePlaneCurveSingularLocus(StrictModel):
    status: Literal["SINGULAR_ZERO_DIMENSIONAL"]
    saturated_jacobian_ideal: RationalPolynomialIdeal
    projective_dimension: Literal[0] = 0
    points: tuple[ProjectivePlaneCurveSingularPointRecord, ...] = Field(
        min_length=1, max_length=MAX_PROJECTIVE_SINGULAR_POINTS
    )

    @model_validator(mode="after")
    def require_canonical_complete_point_family(self) -> Self:
        keys = tuple(record.point.model_dump_json() for record in self.points)
        if keys != tuple(sorted(keys)) or len(set(keys)) != len(keys):
            raise _validation_error(
                "point_order",
                "singular points must be unique and canonically ordered",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        ideal: RationalPolynomialIdeal,
        points: tuple[ProjectivePlaneCurveSingularPointRecord, ...],
    ) -> Self:
        """Construct the finite branch after exact chart enumeration."""

        return cls.model_construct(
            status="SINGULAR_ZERO_DIMENSIONAL",
            saturated_jacobian_ideal=ideal,
            projective_dimension=0,
            points=points,
        )


class PositiveDimensionalProjectivePlaneCurveSingularLocus(StrictModel):
    status: Literal["SINGULAR_POSITIVE_DIMENSIONAL"]
    saturated_jacobian_ideal: RationalPolynomialIdeal
    affine_cone_dimension: Literal[2] = 2
    projective_dimension: Literal[1] = 1
    rational_minimal_components: tuple[RationalPolynomialIdeal, ...] = Field(
        min_length=1, max_length=MAX_PROJECTIVE_SINGULAR_COMPONENTS
    )

    @model_validator(mode="after")
    def bind_component_ring(self) -> Self:
        if any(
            component.variables != self.saturated_jacobian_ideal.variables
            for component in self.rational_minimal_components
        ):
            raise _validation_error(
                "component_axis",
                "all rational minimal components must use the projective axis",
            )
        keys = tuple(
            component.model_dump_json()
            for component in self.rational_minimal_components
        )
        if keys != tuple(sorted(keys)) or len(set(keys)) != len(keys):
            raise _validation_error(
                "component_order",
                "rational minimal components must be unique and canonically ordered",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        ideal: RationalPolynomialIdeal,
        components: tuple[RationalPolynomialIdeal, ...],
    ) -> Self:
        """Construct the positive-dimensional branch after exact decomposition."""

        return cls.model_construct(
            status="SINGULAR_POSITIVE_DIMENSIONAL",
            saturated_jacobian_ideal=ideal,
            affine_cone_dimension=2,
            projective_dimension=1,
            rational_minimal_components=components,
        )


SafeExecutionDetail = Annotated[
    str,
    StringConstraints(min_length=1, max_length=256, strict=True),
]


class IncompleteProjectivePlaneCurveSingularityComputation(StrictModel):
    status: Literal[
        "BACKEND_UNAVAILABLE",
        "TIMEOUT",
        "CANCELLED",
        "LIMIT_EXCEEDED",
        "BACKEND_ERROR",
    ]
    stage: Literal[
        "SATURATION",
        "PROJECTIVE_COMPONENTS",
        "CHART_ZERO_COMPONENTS",
        "CHART_ONE_COMPONENTS",
        "POINT_CONSTRUCTION",
        "RESULT_CONSTRUCTION",
    ]
    detail: SafeExecutionDetail


ProjectivePlaneCurveSingularityOutcome = Annotated[
    SmoothProjectivePlaneCurve
    | ZeroDimensionalProjectivePlaneCurveSingularLocus
    | PositiveDimensionalProjectivePlaneCurveSingularLocus
    | IncompleteProjectivePlaneCurveSingularityComputation,
    Field(discriminator="status"),
]


class ProjectivePlaneCurveSingularityProfile(StrictModel):
    """A source-bound exact global singular-locus computation over ``QQbar``."""

    source_polynomial: RationalPolynomial
    axis: tuple[
        PolynomialVariable,
        PolynomialVariable,
        PolynomialVariable,
    ]
    partial_derivatives: tuple[
        RationalPolynomial,
        RationalPolynomial,
        RationalPolynomial,
    ]
    base_field: Literal["QQ"] = "QQ"
    geometric_scope: Literal["ALGEBRAIC_CLOSURE"] = "ALGEBRAIC_CLOSURE"
    outcome: ProjectivePlaneCurveSingularityOutcome

    @model_validator(mode="after")
    def bind_profile_axis(self) -> Self:
        if self.source_polynomial.variables != self.axis:
            raise _validation_error(
                "source_axis", "the normalized source must use the declared axis"
            )
        if any(
            derivative.variables != self.axis for derivative in self.partial_derivatives
        ):
            raise _validation_error(
                "partial_axis", "all partial derivatives must use the source axis"
            )
        mathematical = (
            SmoothProjectivePlaneCurve,
            ZeroDimensionalProjectivePlaneCurveSingularLocus,
            PositiveDimensionalProjectivePlaneCurveSingularLocus,
        )
        if (
            isinstance(self.outcome, mathematical)
            and self.outcome.saturated_jacobian_ideal.variables != self.axis
        ):
            raise _validation_error(
                "saturation_axis",
                "the saturated Jacobian ideal must use the source axis",
            )
        if isinstance(
            self.outcome, ZeroDimensionalProjectivePlaneCurveSingularLocus
        ) and any(record.point.axis != self.axis for record in self.outcome.points):
            raise _validation_error(
                "point_axis", "every singular point must use the source axis"
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        source: RationalPolynomial,
        axis: tuple[
            PolynomialVariable,
            PolynomialVariable,
            PolynomialVariable,
        ],
        partials: tuple[
            RationalPolynomial,
            RationalPolynomial,
            RationalPolynomial,
        ],
        outcome: ProjectivePlaneCurveSingularityOutcome,
    ) -> Self:
        """Construct one trusted profile after its exact kernel transaction."""

        return cls.model_construct(
            source_polynomial=source,
            axis=axis,
            partial_derivatives=partials,
            base_field="QQ",
            geometric_scope="ALGEBRAIC_CLOSURE",
            outcome=outcome,
        )


__all__ = [
    "MAX_PROJECTIVE_PLANE_CURVE_COEFFICIENT_DIGITS",
    "MAX_PROJECTIVE_PLANE_CURVE_DEGREE",
    "MAX_PROJECTIVE_PLANE_CURVE_TERMS",
    "MAX_PROJECTIVE_SINGULAR_COMPONENTS",
    "MAX_PROJECTIVE_SINGULAR_FIELD_DEGREE",
    "MAX_PROJECTIVE_SINGULAR_POINTS",
    "IncompleteProjectivePlaneCurveSingularityComputation",
    "PositiveDimensionalProjectivePlaneCurveSingularLocus",
    "ProjectivePlaneCurveFirstJet",
    "ProjectivePlaneCurveSingularPointRecord",
    "ProjectivePlaneCurveSingularityBudget",
    "ProjectivePlaneCurveSingularityOutcome",
    "ProjectivePlaneCurveSingularityProfile",
    "ProjectivePlaneCurveSingularityRequest",
    "SmoothProjectivePlaneCurve",
    "ZeroDimensionalProjectivePlaneCurveSingularLocus",
]
