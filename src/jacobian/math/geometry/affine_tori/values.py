"""Canonical exact values for affine maps and cosets on standard real tori."""

from __future__ import annotations

from typing import Annotated, Self

from pydantic import Field, StrictInt, WithJsonSchema, model_validator
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import PydanticCustomError

from jacobian._exact import (
    CanonicalRational,
    ExactInteger,
    require_bounded_rational,
)
from jacobian._models import StrictModel, canonicalize_json_containers
from jacobian.canonical import format_canonical_integer
from jacobian.math.matrices.values import IntegerMatrix

# Conservative preflight fallback: the derived admission in
# build_affine_torus_plan is the actual gate, checking mathematical work and
# point height as well as the private worker channel. These caps only reject
# absurdly large raw input before the plan runs.
#
# The dimension envelope is capped by the reused integer-matrix carrier
# (`IntegerMatrix`, whose rows and columns are bounded at 32), so the
# exposed affine-matrix schema and preflight stay aligned with the actual
# parse rather than advertising a range the linear-part carrier cannot hold.
MAX_AFFINE_TORUS_DIMENSION = 32
MAX_AFFINE_TORUS_INPUT_DIGITS = 500
MAX_AFFINE_TORUS_POINT_DIGITS = 1_050
_AFFINE_SIGNED_INTEGER_PATTERN = (
    rf"^(?:0|-?[1-9][0-9]{{0,{MAX_AFFINE_TORUS_INPUT_DIGITS - 1}}})$"
)
_AFFINE_POSITIVE_INTEGER_PATTERN = (
    rf"^[1-9][0-9]{{0,{MAX_AFFINE_TORUS_INPUT_DIGITS - 1}}}$"
)
_POINT_SIGNED_INTEGER_PATTERN = (
    rf"^(?:0|-?[1-9][0-9]{{0,{MAX_AFFINE_TORUS_POINT_DIGITS - 1}}})$"
)
_POINT_POSITIVE_INTEGER_PATTERN = (
    rf"^[1-9][0-9]{{0,{MAX_AFFINE_TORUS_POINT_DIGITS - 1}}}$"
)


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"affine_torus.{reason}", message)


def _integer_digits(value: int | str) -> int:
    text = value if isinstance(value, str) else format_canonical_integer(value)
    return len(text.lstrip("-"))


def _preflight_sequence(value: object, *, label: str, maximum: int) -> None:
    if not isinstance(value, (list, tuple)):
        raise _validation_error("raw_type", f"{label} must be a JSON array")
    if len(value) > maximum:
        raise _validation_error(
            "raw_size", f"{label} exceeds the raw length bound of {maximum}"
        )


def _preflight_fields(
    value: dict[object, object], *, allowed: frozenset[str], label: str
) -> None:
    if any(not isinstance(key, str) or key not in allowed for key in value):
        raise _validation_error("raw_fields", f"{label} contains an unknown raw field")


def _preflight_torus(value: object) -> None:
    if isinstance(value, StandardRealTorus):
        return
    if not isinstance(value, dict):
        raise _validation_error("raw_type", "torus must be a JSON object")
    _preflight_fields(value, allowed=frozenset({"dimension"}), label="torus")
    dimension = value.get("dimension")
    if not isinstance(dimension, int) or isinstance(dimension, bool):
        raise _validation_error("raw_type", "torus dimension must be an integer")


def _preflight_affine_linear_part(value: object) -> None:
    """Reject an oversized raw linear part before nested canonical parsing."""

    if isinstance(value, IntegerMatrix):
        return
    if not isinstance(value, dict):
        raise _validation_error("raw_type", "affine linear part must be a JSON object")
    _preflight_fields(
        value,
        allowed=frozenset({"domain", "row_count", "column_count", "entries"}),
        label="affine linear part",
    )
    rows = value.get("row_count")
    columns = value.get("column_count")
    domain = value.get("domain", "ZZ")
    if not isinstance(domain, str):
        raise _validation_error("raw_type", "integer matrix domain must be a string")
    if not isinstance(rows, int) or isinstance(rows, bool):
        raise _validation_error(
            "raw_type", "integer matrix row_count must be an integer"
        )
    if not isinstance(columns, int) or isinstance(columns, bool):
        raise _validation_error(
            "raw_type", "integer matrix column_count must be an integer"
        )
    if (
        isinstance(rows, int)
        and not isinstance(rows, bool)
        and rows > MAX_AFFINE_TORUS_DIMENSION
    ):
        raise _validation_error(
            "raw_shape",
            f"integer matrix exceeds dimension {MAX_AFFINE_TORUS_DIMENSION}",
        )
    if (
        isinstance(columns, int)
        and not isinstance(columns, bool)
        and columns > MAX_AFFINE_TORUS_DIMENSION
    ):
        raise _validation_error(
            "raw_shape",
            f"integer matrix exceeds dimension {MAX_AFFINE_TORUS_DIMENSION}",
        )
    if (
        isinstance(rows, int)
        and not isinstance(rows, bool)
        and isinstance(columns, int)
        and not isinstance(columns, bool)
        and rows != columns
    ):
        raise _validation_error("matrix_shape", "affine linear part must be square")
    entries = value.get("entries")
    _preflight_sequence(
        entries,
        label="integer matrix rows",
        maximum=MAX_AFFINE_TORUS_DIMENSION,
    )
    assert isinstance(entries, (list, tuple))
    for row in entries:
        _preflight_sequence(
            row,
            label="integer matrix columns",
            maximum=MAX_AFFINE_TORUS_DIMENSION,
        )
        assert isinstance(row, (list, tuple))
        for entry in row:
            if not (
                isinstance(entry, str)
                or (isinstance(entry, int) and not isinstance(entry, bool))
            ):
                raise _validation_error(
                    "raw_type", "integer matrix entries must be exact integers"
                )
            if _integer_digits(entry) > MAX_AFFINE_TORUS_INPUT_DIGITS:
                raise _validation_error(
                    "raw_digits",
                    f"integer matrix entries may contain at most {MAX_AFFINE_TORUS_INPUT_DIGITS} digits",
                )


def _preflight_rational_coordinates(
    value: object, *, maximum_length: int, maximum_digits: int
) -> None:
    _preflight_sequence(value, label="torus point coordinates", maximum=maximum_length)
    assert isinstance(value, (list, tuple))
    for coordinate in value:
        if isinstance(coordinate, CanonicalRational):
            continue
        if not isinstance(coordinate, dict):
            raise _validation_error(
                "raw_type", "torus coordinates must be rational JSON objects"
            )
        _preflight_fields(
            coordinate,
            allowed=frozenset({"num", "den"}),
            label="rational coordinate",
        )
        for component in (coordinate.get("num"), coordinate.get("den")):
            if not (
                isinstance(component, str)
                or (isinstance(component, int) and not isinstance(component, bool))
            ):
                raise _validation_error(
                    "raw_type", "rational components must be exact integers"
                )
            if _integer_digits(component) > maximum_digits:
                raise _validation_error(
                    "raw_digits",
                    f"torus coordinates may contain at most {maximum_digits} digits",
                )


def _bounded_rational_schema(
    *,
    maximum_digits: int,
    signed_pattern: str,
    positive_pattern: str,
) -> JsonSchemaValue:
    """Expose one canonical rational's sign-aware component bounds."""

    return {
        "type": "object",
        "title": "CanonicalRational",
        "description": (
            "A reduced rational with a positive denominator and canonical zero. "
            f"Each component has at most {maximum_digits} decimal digits."
        ),
        "additionalProperties": False,
        "properties": {
            "num": {
                "type": "string",
                "maxLength": maximum_digits + 1,
                "pattern": signed_pattern,
                "description": "Canonical reduced numerator.",
            },
            "den": {
                "type": "string",
                "maxLength": maximum_digits,
                "pattern": positive_pattern,
                "description": "Positive canonical reduced denominator.",
            },
        },
        "required": ["num", "den"],
    }


_AffineTorusPointCoordinate = Annotated[
    CanonicalRational,
    WithJsonSchema(
        _bounded_rational_schema(
            maximum_digits=MAX_AFFINE_TORUS_POINT_DIGITS,
            signed_pattern=_POINT_SIGNED_INTEGER_PATTERN,
            positive_pattern=_POINT_POSITIVE_INTEGER_PATTERN,
        )
    ),
]


class StandardRealTorus(StrictModel):
    """The standard real torus ``T^n = R^n / Z^n`` with its ordered axis."""

    dimension: StrictInt = Field(ge=0, le=MAX_AFFINE_TORUS_DIMENSION)


class RationalTorusPoint(StrictModel):
    """One rational point in the canonical half-open cube ``[0,1)^n``."""

    torus: StandardRealTorus
    coordinates: tuple[_AffineTorusPointCoordinate, ...] = Field(
        max_length=MAX_AFFINE_TORUS_DIMENSION
    )

    @model_validator(mode="before")
    @classmethod
    def preflight_raw_coordinates(cls, value: object) -> object:
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            _preflight_fields(
                value,
                allowed=frozenset({"torus", "coordinates"}),
                label="torus point",
            )
            _preflight_torus(value.get("torus"))
            _preflight_rational_coordinates(
                value.get("coordinates"),
                maximum_length=MAX_AFFINE_TORUS_DIMENSION,
                maximum_digits=MAX_AFFINE_TORUS_POINT_DIGITS,
            )
        else:
            raise _validation_error("raw_type", "torus point must be a JSON object")
        return canonicalize_json_containers(value)

    @model_validator(mode="after")
    def require_canonical_torus_coordinates(self) -> Self:
        if len(self.coordinates) != self.torus.dimension:
            raise _validation_error(
                "point_shape", "torus point coordinates must match its dimension"
            )
        for coordinate in self.coordinates:
            try:
                require_bounded_rational(
                    coordinate,
                    max_digits=MAX_AFFINE_TORUS_POINT_DIGITS,
                    label="torus point coordinate",
                )
            except ValueError as exc:
                raise _validation_error("point_digits", str(exc)) from exc
            if not 0 <= coordinate.as_fraction() < 1:
                raise _validation_error(
                    "point_representative",
                    "torus point coordinates must use representatives in [0,1)",
                )
        return self


def _affine_linear_part_schema() -> JsonSchemaValue:
    """Project the 16-axis map envelope onto the reused matrix carrier."""

    schema = IntegerMatrix.model_json_schema()
    for field_name in ("row_count", "column_count"):
        schema["properties"][field_name].update(
            minimum=0,
            maximum=MAX_AFFINE_TORUS_DIMENSION,
        )
    entries = schema["properties"]["entries"]
    entries["maxItems"] = MAX_AFFINE_TORUS_DIMENSION
    entries["items"]["maxItems"] = MAX_AFFINE_TORUS_DIMENSION
    entries["items"]["items"].update(
        maxLength=MAX_AFFINE_TORUS_INPUT_DIGITS + 1,
        pattern=_AFFINE_SIGNED_INTEGER_PATTERN,
    )
    entries["description"] = (
        f"Exactly n rows of n canonical integers, each with at most {MAX_AFFINE_TORUS_INPUT_DIGITS} decimal digits."
    )
    return schema


def _affine_translation_schema() -> JsonSchemaValue:
    """Expose the source-only rational grammar without forking its type."""

    return {
        "type": "object",
        "title": "RationalTorusPoint",
        "description": (
            "A point of the same torus in [0,1)^n. Rational components are "
            "reduced, denominators are positive, and each component has at most "
            f"{MAX_AFFINE_TORUS_INPUT_DIGITS} decimal digits."
        ),
        "additionalProperties": False,
        "properties": {
            "torus": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "dimension": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": MAX_AFFINE_TORUS_DIMENSION,
                    }
                },
                "required": ["dimension"],
            },
            "coordinates": {
                "type": "array",
                "maxItems": MAX_AFFINE_TORUS_DIMENSION,
                "items": _bounded_rational_schema(
                    maximum_digits=MAX_AFFINE_TORUS_INPUT_DIGITS,
                    signed_pattern=_AFFINE_SIGNED_INTEGER_PATTERN,
                    positive_pattern=_AFFINE_POSITIVE_INTEGER_PATTERN,
                ),
            },
        },
        "required": ["torus", "coordinates"],
    }


class RationalAffineTorusMap(StrictModel):
    """The affine endomorphism ``x |-> A x + b`` of one standard torus."""

    torus: StandardRealTorus
    linear_part: Annotated[
        IntegerMatrix,
        WithJsonSchema(_affine_linear_part_schema()),
    ]
    translation: Annotated[
        RationalTorusPoint,
        WithJsonSchema(_affine_translation_schema()),
    ] = Field(
        description=(
            "A point of the same torus in [0,1)^n; numerator and denominator "
            f"components have at most {MAX_AFFINE_TORUS_INPUT_DIGITS} decimal digits for affine-map sources."
        )
    )

    @model_validator(mode="before")
    @classmethod
    def preflight_raw_source(cls, value: object) -> object:
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            _preflight_fields(
                value,
                allowed=frozenset({"torus", "linear_part", "translation"}),
                label="affine torus map",
            )
            _preflight_torus(value.get("torus"))
            _preflight_affine_linear_part(value.get("linear_part"))
            translation = value.get("translation")
            if isinstance(translation, RationalTorusPoint):
                pass
            elif isinstance(translation, dict):
                _preflight_fields(
                    translation,
                    allowed=frozenset({"torus", "coordinates"}),
                    label="affine translation",
                )
                _preflight_torus(translation.get("torus"))
                _preflight_rational_coordinates(
                    translation.get("coordinates"),
                    maximum_length=MAX_AFFINE_TORUS_DIMENSION,
                    maximum_digits=MAX_AFFINE_TORUS_INPUT_DIGITS,
                )
            else:
                raise _validation_error(
                    "raw_type", "affine translation must be a torus-point object"
                )
        else:
            raise _validation_error(
                "raw_type", "affine torus map must be a JSON object"
            )
        return canonicalize_json_containers(value)

    @model_validator(mode="after")
    def require_one_bounded_torus_map(self) -> Self:
        dimension = self.torus.dimension
        if (self.linear_part.row_count, self.linear_part.column_count) != (
            dimension,
            dimension,
        ):
            raise _validation_error(
                "map_shape", "affine linear part must be n by n on its torus"
            )
        if self.translation.torus != self.torus:
            raise _validation_error(
                "ambient_mismatch", "affine translation must belong to the source torus"
            )
        if any(
            _integer_digits(entry) > MAX_AFFINE_TORUS_INPUT_DIGITS
            for row in self.linear_part.entries
            for entry in row
        ):
            raise _validation_error(
                "map_digits",
                f"affine linear-part entries may contain at most {MAX_AFFINE_TORUS_INPUT_DIGITS} digits",
            )
        for coordinate in self.translation.coordinates:
            try:
                require_bounded_rational(
                    coordinate,
                    max_digits=MAX_AFFINE_TORUS_INPUT_DIGITS,
                    label="affine translation coordinate",
                )
            except ValueError as exc:
                raise _validation_error("map_digits", str(exc)) from exc
        return self


class IntegralTorusCharacter(StrictModel):
    """An integral character of ``T^n`` with a source-bound coefficient axis.

    Nonzero primitivity is a witness claim of the fixed-locus operation and
    is checked there (or by :func:`verify_integral_torus_character`).
    """

    torus: StandardRealTorus
    coefficients: tuple[ExactInteger, ...] = Field(
        min_length=1, max_length=MAX_AFFINE_TORUS_DIMENSION
    )

    @model_validator(mode="after")
    def require_character_shape(self) -> Self:
        if len(self.coefficients) != self.torus.dimension:
            raise _validation_error(
                "character_shape", "character coefficients must match torus dimension"
            )
        return self


class ConnectedSubtorusParameterization(StrictModel):
    """A primitive column-lattice map ``T^k -> T^n`` onto a connected subtorus."""

    ambient_torus: StandardRealTorus
    parameter_dimension: StrictInt = Field(ge=0, le=MAX_AFFINE_TORUS_DIMENSION)
    embedding: IntegerMatrix

    @model_validator(mode="after")
    def require_embedding_shape(self) -> Self:
        if (self.embedding.row_count, self.embedding.column_count) != (
            self.ambient_torus.dimension,
            self.parameter_dimension,
        ):
            raise _validation_error(
                "subtorus_shape", "subtorus embedding must have shape n by k"
            )
        if self.parameter_dimension > self.ambient_torus.dimension:
            raise _validation_error(
                "subtorus_dimension",
                "subtorus parameter dimension must not exceed ambient torus dimension",
            )
        return self


class FiniteTorusComponentPresentation(StrictModel):
    """A presentation ``Z^r / C Z^r`` of the finite component group."""

    generator_count: StrictInt = Field(ge=0, le=MAX_AFFINE_TORUS_DIMENSION)
    relation_matrix: IntegerMatrix
    generator_orders: tuple[ExactInteger, ...] = Field(
        max_length=MAX_AFFINE_TORUS_DIMENSION
    )
    invariant_factors: tuple[ExactInteger, ...] = Field(
        max_length=MAX_AFFINE_TORUS_DIMENSION
    )
    component_count: ExactInteger

    @model_validator(mode="after")
    def require_finite_presentation_metadata(self) -> Self:
        rank = self.generator_count
        if (self.relation_matrix.row_count, self.relation_matrix.column_count) != (
            rank,
            rank,
        ):
            raise _validation_error(
                "presentation_shape", "component relation matrix must be r by r"
            )
        if len(self.generator_orders) != rank:
            raise _validation_error(
                "presentation_orders", "one order is required for each generator"
            )
        if len(self.invariant_factors) > rank:
            raise _validation_error(
                "presentation_invariants",
                "invariant factor count must not exceed generator count",
            )
        return self


class RationalTorusCosetFamily(StrictModel):
    """A finite union of rational translates of one connected subtorus."""

    ambient_torus: StandardRealTorus
    base_point: RationalTorusPoint
    identity_component: ConnectedSubtorusParameterization
    component_generators: tuple[RationalTorusPoint, ...] = Field(
        max_length=MAX_AFFINE_TORUS_DIMENSION
    )
    finite_components: FiniteTorusComponentPresentation

    @model_validator(mode="after")
    def require_one_ambient_and_generator_axis(self) -> Self:
        if (
            self.base_point.torus != self.ambient_torus
            or self.identity_component.ambient_torus != self.ambient_torus
            or any(
                point.torus != self.ambient_torus for point in self.component_generators
            )
        ):
            raise _validation_error(
                "ambient_mismatch", "all coset-family values must use one ambient torus"
            )
        if len(self.component_generators) != self.finite_components.generator_count:
            raise _validation_error(
                "generator_count",
                "component generators must match the finite presentation",
            )
        return self


__all__ = [
    "MAX_AFFINE_TORUS_DIMENSION",
    "MAX_AFFINE_TORUS_INPUT_DIGITS",
    "MAX_AFFINE_TORUS_POINT_DIGITS",
    "ConnectedSubtorusParameterization",
    "FiniteTorusComponentPresentation",
    "IntegralTorusCharacter",
    "RationalAffineTorusMap",
    "RationalTorusCosetFamily",
    "RationalTorusPoint",
    "StandardRealTorus",
]
