"""Typed contracts for exact components of plane semialgebraic sets."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Annotated, Any, Literal, Self

from pydantic import Field, StrictInt, WithJsonSchema, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational, require_bounded_rational
from jacobian._models import StrictModel, canonicalize_json_containers
from jacobian.canonical import (
    CanonicalLimits,
    encode_strict_json,
    parse_canonical_integer,
    sha256_digest,
)
from jacobian.math.analysis.intervals import ClosedRationalInterval, RationalBox
from jacobian.math.number_theory.algebraic_numbers.real import (
    MAX_REAL_ALGEBRAIC_DEGREE,
    RealAlgebraicValue,
    _UnrecognizedRealAlgebraicValue,
)
from jacobian.math.polynomials.values import (
    PolynomialVariable,
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)

MAX_PLANE_COMPONENT_POLYNOMIALS = 4
MAX_PLANE_COMPONENT_TERMS_PER_POLYNOMIAL = 15
MAX_PLANE_COMPONENT_TOTAL_TERMS = 48
MAX_PLANE_COMPONENT_TOTAL_DEGREE = 4
MAX_PLANE_COMPONENT_COEFFICIENT_DIGITS = 32
MAX_PLANE_COMPONENT_SIGN_CONDITIONS = 3**MAX_PLANE_COMPONENT_POLYNOMIALS
MAX_PLANE_COMPONENT_SAMPLES = 8
MAX_PLANE_COMPONENTS = 128
MAX_PLANE_COMPONENT_POINT_DEGREE = MAX_REAL_ALGEBRAIC_DEGREE
MAX_PLANE_COMPONENT_POINT_TERMS = MAX_PLANE_COMPONENT_POINT_DEGREE + 1
MAX_PLANE_COMPONENT_POINT_COEFFICIENT_DIGITS = 512
MAX_PLANE_COMPONENT_POINT_ISOLATOR_DIGITS = 8_192
MAX_PLANE_COMPONENT_SAMPLE_DEGREE = MAX_PLANE_COMPONENT_POINT_DEGREE
MAX_PLANE_COMPONENT_SAMPLE_COEFFICIENT_DIGITS = (
    MAX_PLANE_COMPONENT_POINT_COEFFICIENT_DIGITS
)
# Conservative owner headroom below the canonical structured-value boundary.
MAX_PLANE_COMPONENT_RESULT_BYTES = CanonicalLimits().max_output_bytes - 4_096


def _plane_coordinate_schema() -> dict[str, object]:
    schema = RealAlgebraicValue.model_json_schema()
    schema = dict(schema)
    polynomial = schema.get("properties", {}).get("polynomial")
    if isinstance(polynomial, dict):
        polynomial = dict(polynomial)
        polynomial["maxItems"] = MAX_PLANE_COMPONENT_POINT_TERMS
        schema["properties"] = {
            **schema.get("properties", {}),
            "polynomial": polynomial,
        }
    return schema


_PlaneCoordinate = Annotated[
    _UnrecognizedRealAlgebraicValue,
    WithJsonSchema(_plane_coordinate_schema()),
]


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"plane_semialgebraic.{reason}", message)


class PlaneSign(StrEnum):
    """One exact sign of a real polynomial value."""

    NEGATIVE = "NEGATIVE"
    ZERO = "ZERO"
    POSITIVE = "POSITIVE"


_SIGN_ORDER = {
    PlaneSign.NEGATIVE: -1,
    PlaneSign.ZERO: 0,
    PlaneSign.POSITIVE: 1,
}


class PlaneSignCondition(StrictModel):
    """One complete sign assignment for an ordered polynomial family."""

    signs: tuple[PlaneSign, ...] = Field(max_length=MAX_PLANE_COMPONENT_POLYNOMIALS)


def _polynomial_key(polynomial: RationalPolynomial) -> bytes:
    return encode_strict_json(polynomial.model_dump(mode="json"))


class PlaneSemialgebraicSet(StrictModel):
    """A normalized finite union of complete plane sign conditions.

    Each row is a conjunction assigning one sign to every polynomial; the
    rows are joined by disjunction.  This finite sign table is a canonical
    Boolean representation for the bounded family, so atom order and Boolean
    association do not survive as incidental syntax.
    """

    axis: tuple[PolynomialVariable, PolynomialVariable]
    polynomials: tuple[RationalPolynomial, ...] = Field(
        max_length=MAX_PLANE_COMPONENT_POLYNOMIALS,
        description=(
            "At most four nonzero QQ[x,y] polynomials. Component-profile "
            "admission permits at most 15 terms per polynomial, 48 terms "
            "in total, total degree four, and 32 decimal digits per rational "
            "coefficient numerator or denominator."
        ),
    )
    sign_conditions: tuple[PlaneSignCondition, ...] = Field(
        max_length=MAX_PLANE_COMPONENT_SIGN_CONDITIONS,
        description=(
            "Canonical disjunction of complete sign rows; every row has one "
            "NEGATIVE, ZERO, or POSITIVE entry per polynomial."
        ),
    )

    @model_validator(mode="after")
    def require_one_canonical_sign_table(self) -> Self:
        if self.axis[0] == self.axis[1]:
            raise _validation_error(
                "axis", "plane semialgebraic sets need two distinct axes"
            )
        if any(polynomial.variables != self.axis for polynomial in self.polynomials):
            raise _validation_error(
                "polynomial_axis",
                "every plane sign polynomial must use the complete ordered axis",
            )
        if any(not polynomial.polynomial.terms for polynomial in self.polynomials):
            raise _validation_error(
                "zero_polynomial", "zero sign polynomials must be omitted"
            )
        if any(
            len(condition.signs) != len(self.polynomials)
            for condition in self.sign_conditions
        ):
            raise _validation_error(
                "sign_condition_axis",
                "each sign condition needs one sign per polynomial",
            )

        keys = tuple(_polynomial_key(polynomial) for polynomial in self.polynomials)
        if len(set(keys)) != len(keys):
            raise _validation_error(
                "duplicate_polynomial",
                "the normalized sign family cannot repeat a polynomial",
            )
        permutation = tuple(sorted(range(len(keys)), key=keys.__getitem__))
        polynomials = tuple(self.polynomials[index] for index in permutation)
        conditions = tuple(
            sorted(
                {
                    PlaneSignCondition(
                        signs=tuple(condition.signs[index] for index in permutation)
                    )
                    for condition in self.sign_conditions
                },
                key=lambda condition: tuple(
                    _SIGN_ORDER[sign] for sign in condition.signs
                ),
            )
        )
        object.__setattr__(self, "polynomials", polynomials)
        object.__setattr__(self, "sign_conditions", conditions)
        return self


class IsolatedRealPlanePoint(StrictModel):
    """Two canonical real-algebraic coordinates and a rational isolating box.

    Each coordinate reuses Jacobian's domain-owned real-algebraic scalar rather
    than introducing another minimal-polynomial representation. Producers use
    the deterministic rational isolator of each selected real root. Consumers
    that rely on isolation recognize the structurally decoded values inside
    their own admitted envelope.
    """

    axis: tuple[PolynomialVariable, PolynomialVariable]
    coordinates: tuple[
        _PlaneCoordinate,
        _PlaneCoordinate,
    ] = Field(
        description=(
            "Canonical real-algebraic values for the first and second coordinates."
        )
    )
    isolating_box: RationalBox = Field(
        description=(
            "A rational box selecting one real root of each coordinate polynomial."
        )
    )

    @model_validator(mode="after")
    def require_coordinate_system_shape(self) -> Self:
        if self.axis[0] == self.axis[1]:
            raise _validation_error("point_axis", "a plane point needs two axes")
        if (
            self.isolating_box.domain != "QQ"
            or self.isolating_box.variables != self.axis
        ):
            raise _validation_error(
                "point_box_axis",
                "point isolating box must use the complete ordered plane axis",
            )
        for coordinate in self.coordinates:
            if len(coordinate.polynomial) > MAX_PLANE_COMPONENT_POINT_TERMS:
                raise _validation_error(
                    "point_bound",
                    "plane algebraic point coordinate exceeds the degree-sixteen bound",
                )
            if any(
                len(coefficient.lstrip("-"))
                > MAX_PLANE_COMPONENT_POINT_COEFFICIENT_DIGITS
                for coefficient in coordinate.polynomial
            ):
                raise _validation_error(
                    "point_bound",
                    "plane algebraic point coordinate exceeds the coefficient-height bound",
                )
        try:
            for interval in self.isolating_box.intervals:
                require_bounded_rational(
                    interval.lower,
                    max_digits=MAX_PLANE_COMPONENT_POINT_ISOLATOR_DIGITS,
                    label="plane algebraic point isolating endpoint",
                )
                require_bounded_rational(
                    interval.upper,
                    max_digits=MAX_PLANE_COMPONENT_POINT_ISOLATOR_DIGITS,
                    label="plane algebraic point isolating endpoint",
                )
        except ValueError as exc:
            raise _validation_error("point_bound", str(exc)) from exc
        return self

    @property
    def coordinate_polynomials(
        self,
    ) -> tuple[RationalPolynomial, RationalPolynomial]:
        """Return coordinate minimal polynomials on the point's complete axis."""

        polynomials: list[RationalPolynomial] = []
        for coordinate_index, coordinate in enumerate(self.coordinates):
            degree = len(coordinate.polynomial) - 1
            terms: list[RationalPolynomialTerm] = []
            for offset, encoded_coefficient in enumerate(coordinate.polynomial):
                coefficient = parse_canonical_integer(encoded_coefficient)
                if coefficient == 0:
                    continue
                exponents = [0, 0]
                exponents[coordinate_index] = degree - offset
                terms.append(
                    RationalPolynomialTerm(
                        coefficient=CanonicalRational.from_integer_ratio(
                            coefficient, 1
                        ),
                        exponents=tuple(exponents),
                    )
                )
            polynomials.append(
                RationalPolynomial(
                    variables=self.axis,
                    polynomial=SparseRationalPolynomial(terms=tuple(terms)),
                )
            )
        return polynomials[0], polynomials[1]


type _PlanePointIdentityKey = tuple[
    tuple[PolynomialVariable, PolynomialVariable],
    tuple[tuple[tuple[str, ...], int], ...],
]


def _plane_point_identity_key(
    point: IsolatedRealPlanePoint,
) -> _PlanePointIdentityKey:
    """Return exact point identity without its non-unique isolating evidence."""

    return (
        point.axis,
        tuple(
            (coordinate.polynomial, coordinate.real_root_index)
            for coordinate in point.coordinates
        ),
    )


class PlaneSemialgebraicComponent(StrictModel):
    """One connected component and one exact point belonging to it."""

    component_id: StrictInt = Field(ge=0, lt=MAX_PLANE_COMPONENTS)
    representative: IsolatedRealPlanePoint


class PlaneSampleDisposition(StrictModel):
    """Membership and component identity of one supplied sample point."""

    sample_index: StrictInt = Field(ge=0, lt=MAX_PLANE_COMPONENT_SAMPLES)
    status: Literal["INSIDE", "OUTSIDE"]
    component_id: StrictInt | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def bind_component_only_to_interior_samples(self) -> Self:
        if (self.status == "INSIDE") != (self.component_id is not None):
            raise _validation_error(
                "sample_disposition",
                "inside samples need a component ID and outside samples cannot have one",
            )
        return self


class PlaneComponentProfileComputed(StrictModel):
    """The complete exact connected-component profile."""

    status: Literal["COMPUTED"] = "COMPUTED"
    components: tuple[PlaneSemialgebraicComponent, ...] = Field(
        max_length=MAX_PLANE_COMPONENTS,
        description="Every connected component, in representative order.",
    )
    sample_dispositions: tuple[PlaneSampleDisposition, ...] = Field(
        max_length=MAX_PLANE_COMPONENT_SAMPLES
    )

    @model_validator(mode="after")
    def require_canonical_component_and_sample_indices(self) -> Self:
        if tuple(component.component_id for component in self.components) != tuple(
            range(len(self.components))
        ):
            raise _validation_error(
                "component_order", "component IDs must be consecutive canonical indices"
            )
        representative_keys = tuple(
            _plane_point_identity_key(component.representative)
            for component in self.components
        )
        if representative_keys != tuple(sorted(set(representative_keys))):
            raise _validation_error(
                "component_order",
                "component representatives must be unique and canonically ordered",
            )
        if tuple(
            disposition.sample_index for disposition in self.sample_dispositions
        ) != tuple(range(len(self.sample_dispositions))):
            raise _validation_error(
                "sample_order", "sample dispositions must retain request order"
            )
        if any(
            disposition.component_id is not None
            and disposition.component_id >= len(self.components)
            for disposition in self.sample_dispositions
        ):
            raise _validation_error(
                "sample_component", "sample component ID must name a returned component"
            )
        return self


PlaneComponentNoncompletionStatus = Literal[
    "BACKEND_UNAVAILABLE",
    "TIMEOUT",
    "RESOURCE_LIMIT",
    "BACKEND_ERROR",
]
PlaneComponentNoncompletionReason = Literal[
    "SUPPORTED_QEPCAD_NOT_INSTALLED",
    "UNSUPPORTED_QEPCAD_VERSION",
    "QEPCAD_DEADLINE_EXPIRED",
    "QEPCAD_OUTPUT_LIMIT",
    "QEPCAD_CELL_LIMIT",
    "QEPCAD_INVALID_OUTPUT",
    "QEPCAD_EXECUTION_FAILED",
    "SAMPLE_RECOGNITION_DEADLINE_EXPIRED",
    "SAMPLE_RECOGNITION_OUTPUT_LIMIT",
    "RESULT_OUTPUT_LIMIT",
    "SAMPLE_RECOGNITION_INVALID_OUTPUT",
    "SAMPLE_RECOGNITION_EXECUTION_FAILED",
]


class PlaneComponentProfileNoncompletion(StrictModel):
    """An operational non-completion carrying no topological conclusion."""

    status: PlaneComponentNoncompletionStatus
    reason: PlaneComponentNoncompletionReason
    request_digest: str | None = None
    budget_seconds: StrictInt | None = Field(default=None, ge=1)
    elapsed_ms: StrictInt | None = Field(default=None, ge=0)
    timeout_layer: Literal["QEPCAD", "SAMPLE_RECOGNITION"] | None = None
    operation_version: Literal["1"] = "1"
    repository_revision: str = "unknown"

    @model_validator(mode="after")
    def bind_reason_to_status(self) -> Self:
        reasons_by_status: dict[
            PlaneComponentNoncompletionStatus,
            frozenset[PlaneComponentNoncompletionReason],
        ] = {
            "BACKEND_UNAVAILABLE": frozenset(
                {
                    "SUPPORTED_QEPCAD_NOT_INSTALLED",
                    "UNSUPPORTED_QEPCAD_VERSION",
                }
            ),
            "TIMEOUT": frozenset(
                {
                    "QEPCAD_DEADLINE_EXPIRED",
                    "SAMPLE_RECOGNITION_DEADLINE_EXPIRED",
                }
            ),
            "RESOURCE_LIMIT": frozenset(
                {
                    "QEPCAD_OUTPUT_LIMIT",
                    "QEPCAD_CELL_LIMIT",
                    "SAMPLE_RECOGNITION_OUTPUT_LIMIT",
                    "RESULT_OUTPUT_LIMIT",
                }
            ),
            "BACKEND_ERROR": frozenset(
                {
                    "QEPCAD_INVALID_OUTPUT",
                    "QEPCAD_EXECUTION_FAILED",
                    "SAMPLE_RECOGNITION_INVALID_OUTPUT",
                    "SAMPLE_RECOGNITION_EXECUTION_FAILED",
                }
            ),
        }
        if self.reason not in reasons_by_status[self.status]:
            raise _validation_error(
                "noncompletion_reason",
                "plane-component non-completion reason does not match its status",
            )
        if self.status == "TIMEOUT" and (
            self.request_digest is None
            or self.budget_seconds is None
            or self.elapsed_ms is None
            or self.timeout_layer is None
        ):
            raise _validation_error(
                "timeout_metadata",
                "timeout outcomes must retain replay metadata",
            )
        if self.status == "TIMEOUT":
            expected_layer = (
                "QEPCAD"
                if self.reason == "QEPCAD_DEADLINE_EXPIRED"
                else "SAMPLE_RECOGNITION"
            )
            if self.timeout_layer != expected_layer:
                raise _validation_error(
                    "timeout_metadata",
                    "timeout_layer must match the timeout reason",
                )
        return self


PlaneComponentProfileOutcome = Annotated[
    PlaneComponentProfileComputed | PlaneComponentProfileNoncompletion,
    Field(discriminator="status"),
]


def _raw_collection_limit(
    value: object,
    *,
    maximum: int,
    reason: str,
    label: str,
) -> None:
    if value is None:
        return
    if not isinstance(value, (list, tuple)):
        raise _validation_error(reason, f"{label} must be a JSON array")
    if len(value) > maximum:
        raise _validation_error(reason, f"{label} admits at most {maximum} entries")


def _raw_mapping_keys(
    value: Mapping[object, object],
    *,
    allowed: frozenset[str],
    reason: str,
    label: str,
) -> None:
    if any(not isinstance(key, str) for key in value):
        raise _validation_error(reason, f"{label} field names must be strings")
    unknown = set(value) - allowed
    if unknown:
        raise _validation_error(reason, f"{label} contains unknown fields")


def _raw_string_collection(value: object, *, label: str) -> None:
    if isinstance(value, (list, tuple)) and any(
        not isinstance(item, str) for item in value
    ):
        raise _validation_error(
            "sequence_item_shape", f"every {label} entry must be a string"
        )


def _raw_optional_string(value: object, *, label: str) -> None:
    if value is not None and not isinstance(value, str):
        raise _validation_error("field_shape", f"{label} must be a string")


def _raw_rational_limit(value: object, *, maximum_digits: int, label: str) -> None:
    if isinstance(value, CanonicalRational):
        if max(len(value.num.lstrip("-")), len(value.den)) > maximum_digits:
            raise _validation_error(
                "coefficient_digits",
                f"{label} admits at most {maximum_digits} decimal digits",
            )
        return
    if not isinstance(value, Mapping):
        if value is not None:
            raise _validation_error(
                "rational_shape", f"{label} must be a canonical rational object"
            )
        return
    _raw_mapping_keys(
        value,
        allowed=frozenset({"num", "den"}),
        reason="rational_shape",
        label=label,
    )
    for part in ("num", "den"):
        component = value.get(part)
        if component is None:
            continue
        if not isinstance(component, str):
            raise _validation_error(
                "rational_shape",
                f"{label} numerator and denominator must be decimal strings",
            )
        digits = component.removeprefix("-")
        if len(digits) > maximum_digits:
            raise _validation_error(
                "coefficient_digits",
                f"{label} admits at most {maximum_digits} decimal digits",
            )


def _raw_polynomial_limit(  # noqa: C901
    value: object,
    *,
    maximum_terms: int,
    maximum_exponent: int,
    maximum_coefficient_digits: int,
    label: str,
) -> None:
    terms: object
    if isinstance(value, RationalPolynomial):
        terms = value.polynomial.terms
        if len(terms) > maximum_terms:
            raise _validation_error(
                "term_count", f"{label} admits at most {maximum_terms} terms"
            )
        for term in terms:
            _raw_rational_limit(
                term.coefficient,
                maximum_digits=maximum_coefficient_digits,
                label=f"{label} coefficient",
            )
            if any(
                exponent < 0 or exponent > maximum_exponent
                for exponent in term.exponents
            ):
                raise _validation_error(
                    "exponent_bound",
                    f"{label} exceeds the degree-{maximum_exponent} operation bound",
                )
        return
    if not isinstance(value, Mapping):
        raise _validation_error(
            "polynomial_shape", f"{label} must be a polynomial object"
        )
    _raw_mapping_keys(
        value,
        allowed=frozenset({"domain", "variables", "polynomial"}),
        reason="polynomial_shape",
        label=label,
    )
    _raw_optional_string(value.get("domain"), label=f"{label} domain")
    variables = value.get("variables")
    _raw_collection_limit(
        variables,
        maximum=2,
        reason="polynomial_axis",
        label=f"{label} axis",
    )
    _raw_string_collection(variables, label=f"{label} axis")
    sparse = value.get("polynomial")
    if sparse is None:
        return
    if isinstance(sparse, SparseRationalPolynomial):
        terms = sparse.terms
    elif isinstance(sparse, Mapping):
        _raw_mapping_keys(
            sparse,
            allowed=frozenset({"terms"}),
            reason="polynomial_shape",
            label=f"{label} sparse value",
        )
        terms = sparse.get("terms")
    else:
        raise _validation_error(
            "polynomial_shape", f"{label} sparse value must be an object"
        )
    _raw_collection_limit(
        terms,
        maximum=maximum_terms,
        reason="term_count",
        label=label,
    )
    if not isinstance(terms, (list, tuple)):
        return
    for term in terms:
        if isinstance(term, RationalPolynomialTerm):
            _raw_rational_limit(
                term.coefficient,
                maximum_digits=maximum_coefficient_digits,
                label=f"{label} coefficient",
            )
            if any(
                exponent < 0 or exponent > maximum_exponent
                for exponent in term.exponents
            ):
                raise _validation_error(
                    "exponent_bound",
                    f"{label} exceeds the degree-{maximum_exponent} operation bound",
                )
            continue
        if not isinstance(term, Mapping):
            raise _validation_error(
                "term_shape", f"every {label} term must be an object"
            )
        _raw_mapping_keys(
            term,
            allowed=frozenset({"coefficient", "exponents"}),
            reason="term_shape",
            label=f"{label} term",
        )
        _raw_rational_limit(
            term.get("coefficient"),
            maximum_digits=maximum_coefficient_digits,
            label=f"{label} coefficient",
        )
        exponents = term.get("exponents")
        _raw_collection_limit(
            exponents,
            maximum=2,
            reason="monomial_axis",
            label=f"{label} exponent tuple",
        )
        if isinstance(exponents, (list, tuple)):
            if any(
                not isinstance(exponent, int) or isinstance(exponent, bool)
                for exponent in exponents
            ):
                raise _validation_error(
                    "exponent_shape", f"every {label} exponent must be an integer"
                )
            if any(
                exponent < 0 or exponent > maximum_exponent for exponent in exponents
            ):
                raise _validation_error(
                    "exponent_bound",
                    f"{label} exceeds the degree-{maximum_exponent} operation bound",
                )


def _raw_semialgebraic_envelope(  # noqa: C901
    value: object, *, validate_model: bool = False
) -> None:
    if isinstance(value, PlaneSemialgebraicSet):
        if not validate_model:
            return
        value = {
            "axis": value.axis,
            "polynomials": value.polynomials,
            "sign_conditions": value.sign_conditions,
        }
    if not isinstance(value, Mapping):
        if value is not None:
            raise _validation_error(
                "set_shape", "plane semialgebraic set must be an object"
            )
        return
    _raw_mapping_keys(
        value,
        allowed=frozenset({"axis", "polynomials", "sign_conditions"}),
        reason="set_shape",
        label="plane semialgebraic set",
    )
    _raw_collection_limit(
        value.get("axis"),
        maximum=2,
        reason="axis",
        label="plane component axis",
    )
    _raw_string_collection(value.get("axis"), label="plane component axis")
    polynomials = value.get("polynomials")
    _raw_collection_limit(
        polynomials,
        maximum=MAX_PLANE_COMPONENT_POLYNOMIALS,
        reason="polynomial_count",
        label="plane component polynomial family",
    )
    if isinstance(polynomials, (list, tuple)):
        total_terms = 0
        for polynomial in polynomials:
            _raw_polynomial_limit(
                polynomial,
                maximum_terms=MAX_PLANE_COMPONENT_TERMS_PER_POLYNOMIAL,
                maximum_exponent=MAX_PLANE_COMPONENT_TOTAL_DEGREE,
                maximum_coefficient_digits=MAX_PLANE_COMPONENT_COEFFICIENT_DIGITS,
                label="plane sign polynomial",
            )
            terms: object
            if isinstance(polynomial, RationalPolynomial):
                terms = polynomial.polynomial.terms
            elif isinstance(polynomial, SparseRationalPolynomial):
                terms = polynomial.terms
            elif isinstance(polynomial, Mapping):
                nested = polynomial.get("polynomial")
                if isinstance(nested, SparseRationalPolynomial):
                    terms = nested.terms
                elif isinstance(nested, Mapping):
                    terms = nested.get("terms")
                else:
                    terms = None
            else:
                terms = None
            if isinstance(terms, (list, tuple)):
                total_terms += len(terms)
                for term in terms:
                    exponents: object
                    if isinstance(term, RationalPolynomialTerm):
                        exponents = term.exponents
                    elif isinstance(term, Mapping):
                        exponents = term.get("exponents")
                    else:
                        exponents = None
                    if isinstance(exponents, (list, tuple)) and sum(exponents) > (
                        MAX_PLANE_COMPONENT_TOTAL_DEGREE
                    ):
                        raise _validation_error(
                            "total_degree",
                            "plane sign polynomial total degree exceeds the "
                            "degree-four bound",
                        )
        if total_terms > MAX_PLANE_COMPONENT_TOTAL_TERMS:
            raise _validation_error(
                "total_terms",
                "plane sign family admits at most "
                f"{MAX_PLANE_COMPONENT_TOTAL_TERMS} terms",
            )
    sign_conditions = value.get("sign_conditions")
    _raw_collection_limit(
        sign_conditions,
        maximum=MAX_PLANE_COMPONENT_SIGN_CONDITIONS,
        reason="sign_condition_count",
        label="plane component sign table",
    )
    if isinstance(sign_conditions, (list, tuple)):
        for condition in sign_conditions:
            if isinstance(condition, PlaneSignCondition):
                continue
            if not isinstance(condition, Mapping):
                raise _validation_error(
                    "sign_condition_shape",
                    "every plane sign condition must be an object",
                )
            _raw_mapping_keys(
                condition,
                allowed=frozenset({"signs"}),
                reason="sign_condition_shape",
                label="plane sign condition",
            )
            signs = condition.get("signs")
            _raw_collection_limit(
                signs,
                maximum=MAX_PLANE_COMPONENT_POLYNOMIALS,
                reason="sign_condition_axis",
                label="plane sign condition",
            )
            _raw_string_collection(signs, label="plane sign condition")


def _raw_algebraic_coordinate_envelope(coordinate: object) -> None:
    if not isinstance(coordinate, Mapping):
        raise _validation_error(
            "sample_coordinate_shape",
            "every plane sample coordinate must be a real-algebraic value",
        )
    _raw_mapping_keys(
        coordinate,
        allowed=frozenset({"polynomial", "real_root_index"}),
        reason="sample_coordinate_shape",
        label="plane sample coordinate",
    )
    polynomial = coordinate.get("polynomial")
    _raw_collection_limit(
        polynomial,
        maximum=MAX_PLANE_COMPONENT_SAMPLE_DEGREE + 1,
        reason="sample_coordinate_degree",
        label="plane sample coordinate polynomial",
    )
    if isinstance(polynomial, (list, tuple)):
        if any(not isinstance(coefficient, str) for coefficient in polynomial):
            raise _validation_error(
                "sample_coordinate_shape",
                "plane sample coordinate coefficients must be decimal strings",
            )
        if any(
            len(coefficient.lstrip("-")) > MAX_PLANE_COMPONENT_SAMPLE_COEFFICIENT_DIGITS
            for coefficient in polynomial
        ):
            raise _validation_error(
                "coefficient_digits",
                "plane algebraic sample coordinate exceeds the coefficient-height bound",
            )
    root_index = coordinate.get("real_root_index")
    if root_index is not None and (
        not isinstance(root_index, int) or isinstance(root_index, bool)
    ):
        raise _validation_error(
            "sample_coordinate_shape",
            "plane sample coordinate root index must be an integer",
        )


def _raw_sample_envelope(value: object) -> None:
    if isinstance(value, IsolatedRealPlanePoint):
        return
    if not isinstance(value, Mapping):
        raise _validation_error("sample_shape", "every plane sample must be an object")
    _raw_mapping_keys(
        value,
        allowed=frozenset({"axis", "coordinates", "isolating_box"}),
        reason="sample_shape",
        label="plane sample",
    )
    _raw_collection_limit(
        value.get("axis"),
        maximum=2,
        reason="sample_axis",
        label="plane sample axis",
    )
    _raw_string_collection(value.get("axis"), label="plane sample axis")
    coordinates = value.get("coordinates")
    _raw_collection_limit(
        coordinates,
        maximum=2,
        reason="sample_coordinate_count",
        label="plane sample coordinate family",
    )
    if isinstance(coordinates, (list, tuple)):
        for coordinate in coordinates:
            _raw_algebraic_coordinate_envelope(coordinate)
    isolating_box = value.get("isolating_box")
    if isolating_box is None:
        return
    if isinstance(isolating_box, RationalBox):
        return
    if not isinstance(isolating_box, Mapping):
        raise _validation_error(
            "sample_box_shape", "plane sample isolating box must be an object"
        )
    _raw_mapping_keys(
        isolating_box,
        allowed=frozenset({"domain", "variables", "intervals"}),
        reason="sample_box_shape",
        label="plane sample isolating box",
    )
    _raw_optional_string(
        isolating_box.get("domain"), label="plane sample isolating-box domain"
    )
    variables = isolating_box.get("variables")
    _raw_collection_limit(
        variables,
        maximum=2,
        reason="sample_box_axis",
        label="plane sample isolating-box axis",
    )
    _raw_string_collection(variables, label="plane sample isolating-box axis")
    intervals = isolating_box.get("intervals")
    _raw_collection_limit(
        intervals,
        maximum=2,
        reason="sample_interval_count",
        label="plane sample isolating box",
    )
    if isinstance(intervals, (list, tuple)):
        for interval in intervals:
            if isinstance(interval, ClosedRationalInterval):
                continue
            if not isinstance(interval, Mapping):
                raise _validation_error(
                    "sample_interval_shape",
                    "every plane sample isolating interval must be an object",
                )
            _raw_mapping_keys(
                interval,
                allowed=frozenset({"lower", "upper"}),
                reason="sample_interval_shape",
                label="plane sample isolating interval",
            )
            for endpoint in ("lower", "upper"):
                _raw_rational_limit(
                    interval.get(endpoint),
                    maximum_digits=MAX_PLANE_COMPONENT_POINT_ISOLATOR_DIGITS,
                    label="plane sample isolating endpoint",
                )


class PlaneComponentProfileRequest(StrictModel):
    """Compute a complete bounded profile and locate optional exact samples.

    Supplied coordinate polynomials use the same degree-sixteen and 512-digit
    carrier as returned representatives. Nondegenerate requests also bound the
    complete CAD projection family before backend execution.
    """

    semialgebraic_set: PlaneSemialgebraicSet
    samples: tuple[IsolatedRealPlanePoint, ...] = Field(
        default=(),
        max_length=MAX_PLANE_COMPONENT_SAMPLES,
        description=(
            "Up to eight exact algebraic points to classify as outside or bind "
            "to the returned component IDs."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def require_raw_envelope(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data
        _raw_mapping_keys(
            data,
            allowed=frozenset({"semialgebraic_set", "samples"}),
            reason="request_shape",
            label="plane component request",
        )
        _raw_semialgebraic_envelope(data.get("semialgebraic_set"))
        samples = data.get("samples")
        _raw_collection_limit(
            samples,
            maximum=MAX_PLANE_COMPONENT_SAMPLES,
            reason="sample_count",
            label="plane component samples",
        )
        if isinstance(samples, (list, tuple)):
            for sample in samples:
                _raw_sample_envelope(sample)
        return canonicalize_json_containers(data)

    @model_validator(mode="after")
    def require_sample_axes(self) -> Self:
        if any(sample.axis != self.semialgebraic_set.axis for sample in self.samples):
            raise _validation_error(
                "sample_axis", "every sample must use the semialgebraic set axis"
            )
        return self


class PlaneComponentProfileResult(StrictModel):
    """The retained source and either an exact profile or no conclusion."""

    semialgebraic_set: PlaneSemialgebraicSet
    samples: tuple[IsolatedRealPlanePoint, ...] = Field(
        max_length=MAX_PLANE_COMPONENT_SAMPLES
    )
    outcome: PlaneComponentProfileOutcome

    @model_validator(mode="before")
    @classmethod
    def require_raw_source_envelope(cls, data: Any) -> Any:
        if isinstance(data, Mapping):
            _raw_semialgebraic_envelope(
                data.get("semialgebraic_set"), validate_model=True
            )
            return canonicalize_json_containers(data)
        return data

    @model_validator(mode="after")
    def bind_computed_sample_count(self) -> Self:
        if any(sample.axis != self.semialgebraic_set.axis for sample in self.samples):
            raise _validation_error(
                "sample_axis", "every retained sample must use the source axis"
            )
        if isinstance(self.outcome, PlaneComponentProfileComputed) and len(
            self.outcome.sample_dispositions
        ) != len(self.samples):
            raise _validation_error(
                "sample_result_count",
                "a computed profile needs one disposition per supplied sample",
            )
        if isinstance(self.outcome, PlaneComponentProfileComputed) and any(
            component.representative.axis != self.semialgebraic_set.axis
            for component in self.outcome.components
        ):
            raise _validation_error(
                "component_axis",
                "every component representative must use the source axis",
            )
        if (
            isinstance(self.outcome, PlaneComponentProfileNoncompletion)
            and self.outcome.status == "TIMEOUT"
        ):
            retained_request = {
                "semialgebraic_set": self.semialgebraic_set.model_dump(mode="json"),
                "samples": [sample.model_dump(mode="json") for sample in self.samples],
            }
            expected_digest = sha256_digest(encode_strict_json(retained_request))
            if self.outcome.request_digest != expected_digest:
                raise _validation_error(
                    "timeout_request_digest",
                    "timeout metadata must bind to the retained source and samples",
                )
        return self


__all__ = [
    "MAX_PLANE_COMPONENT_COEFFICIENT_DIGITS",
    "MAX_PLANE_COMPONENT_POINT_COEFFICIENT_DIGITS",
    "MAX_PLANE_COMPONENT_POINT_DEGREE",
    "MAX_PLANE_COMPONENT_POINT_ISOLATOR_DIGITS",
    "MAX_PLANE_COMPONENT_POINT_TERMS",
    "MAX_PLANE_COMPONENT_POLYNOMIALS",
    "MAX_PLANE_COMPONENT_RESULT_BYTES",
    "MAX_PLANE_COMPONENT_SAMPLES",
    "MAX_PLANE_COMPONENT_SAMPLE_COEFFICIENT_DIGITS",
    "MAX_PLANE_COMPONENT_SAMPLE_DEGREE",
    "MAX_PLANE_COMPONENT_SIGN_CONDITIONS",
    "MAX_PLANE_COMPONENT_TERMS_PER_POLYNOMIAL",
    "MAX_PLANE_COMPONENT_TOTAL_DEGREE",
    "MAX_PLANE_COMPONENT_TOTAL_TERMS",
    "IsolatedRealPlanePoint",
    "PlaneComponentProfileComputed",
    "PlaneComponentProfileNoncompletion",
    "PlaneComponentProfileOutcome",
    "PlaneComponentProfileRequest",
    "PlaneComponentProfileResult",
    "PlaneSampleDisposition",
    "PlaneSemialgebraicComponent",
    "PlaneSemialgebraicSet",
    "PlaneSign",
    "PlaneSignCondition",
]
