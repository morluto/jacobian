"""Canonical bounded values for simple number fields and their embeddings."""

from __future__ import annotations

from collections.abc import Mapping
from math import gcd
from typing import Annotated, Any, Literal, Self

from pydantic import Field, StrictInt, ValidateAs, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalInteger, CanonicalRational
from jacobian._models import StrictModel, canonicalize_json_containers
from jacobian.canonical import parse_canonical_integer
from jacobian.math.number_theory.algebraic_numbers.complex import (
    ComplexAlgebraicValue,
    RationalComplexIsolatingRectangle,
    _UnrecognizedComplexAlgebraicValue,
)
from jacobian.math.number_theory.algebraic_numbers.real import (
    RationalIsolatingInterval,
    _UnrecognizedRealAlgebraicValue,
)

MAX_SIMPLE_NUMBER_FIELD_DEGREE = 31
MAX_NUMBER_FIELD_EMBEDDING_DEGREE = 8
MAX_SIMPLE_NUMBER_FIELD_COEFFICIENT_DIGITS = 256
MAX_SIMPLE_NUMBER_FIELD_ELEMENT_DIGITS = 256
MAX_NUMBER_FIELD_ISOLATOR_COMPONENT_DIGITS = 4_096


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"simple_number_field.{reason}", message)


def _raw_rational_component_bound(
    components: object,
    *,
    max_digits: int,
    label: str,
) -> None:
    if not isinstance(components, (list, tuple)):
        return
    for component in components:
        if not isinstance(component, Mapping):
            continue
        for part in ("num", "den"):
            raw = component.get(part)
            if isinstance(raw, str) and len(raw.lstrip("-")) > max_digits:
                raise _validation_error(
                    "rational_component_bound",
                    f"{label} components may contain at most {max_digits} digits",
                )


class SimpleNumberFieldPresentation(StrictModel):
    """The canonical bounded presentation ``QQ(alpha) = QQ[x]/(f)``.

    The defining polynomial is the unique primitive positive-leading integer
    representative of its rational scalar class. Mathematical consumers
    recognize irreducibility before treating the presented quotient as a
    field; ordinary construction and deserialization establish only this
    bounded canonical representation. ``alpha`` is the fixed indeterminate,
    not caller-selected syntax. Degree one presents ``QQ``.
    """

    domain: Literal["QQ"] = "QQ"
    coefficients_descending: tuple[CanonicalInteger, ...] = Field(
        min_length=2,
        max_length=MAX_SIMPLE_NUMBER_FIELD_DEGREE + 1,
        description=(
            "Primitive irreducible ZZ[x] coefficients in descending degree "
            "with positive leading coefficient."
        ),
        examples=[["1", "0", "-2"]],
    )

    @model_validator(mode="before")
    @classmethod
    def require_raw_polynomial_bound(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data
        coefficients = data.get("coefficients_descending")
        if isinstance(coefficients, (list, tuple)):
            if len(coefficients) > MAX_SIMPLE_NUMBER_FIELD_DEGREE + 1:
                raise _validation_error(
                    "degree_bound",
                    "simple number-field degree exceeds the bounded envelope",
                )
            if any(
                isinstance(coefficient, str)
                and len(coefficient.lstrip("-"))
                > MAX_SIMPLE_NUMBER_FIELD_COEFFICIENT_DIGITS
                for coefficient in coefficients
            ):
                raise _validation_error(
                    "coefficient_bound",
                    "simple number-field coefficients may contain at most "
                    f"{MAX_SIMPLE_NUMBER_FIELD_COEFFICIENT_DIGITS} digits",
                )
        return canonicalize_json_containers(data)

    @model_validator(mode="after")
    def require_canonical_polynomial(self) -> Self:
        coefficients = tuple(
            parse_canonical_integer(coefficient)
            for coefficient in self.coefficients_descending
        )
        if coefficients[0] <= 0:
            raise _validation_error(
                "leading_sign",
                "simple number-field polynomial must have positive leading coefficient",
            )
        content = 0
        for coefficient in coefficients:
            content = gcd(content, abs(coefficient))
        if content != 1:
            raise _validation_error(
                "not_primitive",
                "simple number-field polynomial must be primitive over ZZ",
            )
        return self

    @property
    def degree(self) -> int:
        return len(self.coefficients_descending) - 1


class SimpleNumberFieldElement(StrictModel):
    """One reduced exact element in the presentation's ascending power basis."""

    presentation: SimpleNumberFieldPresentation
    coefficients_ascending: tuple[CanonicalRational, ...] = Field(
        min_length=1,
        max_length=MAX_SIMPLE_NUMBER_FIELD_DEGREE,
        description=("Exactly degree-many coefficients of 1, alpha, ..., alpha^(n-1)."),
    )

    @model_validator(mode="before")
    @classmethod
    def require_raw_coordinate_bound(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data
        _raw_rational_component_bound(
            data.get("coefficients_ascending"),
            max_digits=MAX_SIMPLE_NUMBER_FIELD_ELEMENT_DIGITS,
            label="simple number-field element",
        )
        return canonicalize_json_containers(data)

    @model_validator(mode="after")
    def require_reduced_power_basis_coordinates(self) -> Self:
        if len(self.coefficients_ascending) != self.presentation.degree:
            raise _validation_error(
                "element_coordinate_count",
                "a reduced field element needs exactly one coefficient per power-basis vector",
            )
        for coefficient in self.coefficients_ascending:
            if (
                len(coefficient.num.lstrip("-"))
                > MAX_SIMPLE_NUMBER_FIELD_ELEMENT_DIGITS
                or len(coefficient.den) > MAX_SIMPLE_NUMBER_FIELD_ELEMENT_DIGITS
            ):
                raise _validation_error(
                    "element_coordinate_bound",
                    "simple number-field element coefficients exceed the "
                    f"{MAX_SIMPLE_NUMBER_FIELD_ELEMENT_DIGITS}-digit bound",
                )
        return self


class RealNumberFieldEmbedding(StrictModel):
    """A simple-field presentation with one structurally selected real root.

    Embedding producers establish that the indexed root exists. Deserialization
    preserves that canonical identity without replaying a backend; mathematical
    consumers recognize the root before using this value as a field homomorphism.
    """

    kind: Literal["REAL"]
    presentation: SimpleNumberFieldPresentation
    root: _UnrecognizedRealAlgebraicValue

    @model_validator(mode="after")
    def bind_root_to_presentation(self) -> Self:
        if self.root.polynomial != self.presentation.coefficients_descending:
            raise _validation_error(
                "embedding_polynomial",
                "embedding root must use the presentation's defining polynomial",
            )
        return self


class _ComplexNumberFieldEmbeddingShape(StrictModel):
    """Structural view of a complex embedding recognized by its result owner."""

    kind: Literal["COMPLEX"]
    presentation: SimpleNumberFieldPresentation
    root: _UnrecognizedComplexAlgebraicValue

    @model_validator(mode="after")
    def bind_root_to_presentation(self) -> Self:
        if self.root.polynomial != self.presentation.coefficients_descending:
            raise _validation_error(
                "embedding_polynomial",
                "embedding root must use the presentation's defining polynomial",
            )
        return self


class ComplexNumberFieldEmbedding(_ComplexNumberFieldEmbeddingShape):
    """A field homomorphism selected by one recognized exact nonreal root."""

    root: ComplexAlgebraicValue


def _unrecognized_complex_embedding_from_shape(
    shape: _ComplexNumberFieldEmbeddingShape,
) -> ComplexNumberFieldEmbedding:
    if isinstance(shape, ComplexNumberFieldEmbedding):
        return shape
    return ComplexNumberFieldEmbedding.model_construct(
        kind=shape.kind,
        presentation=shape.presentation,
        root=shape.root,
    )


_UnrecognizedComplexNumberFieldEmbedding = Annotated[
    ComplexNumberFieldEmbedding,
    ValidateAs(
        _ComplexNumberFieldEmbeddingShape,
        _unrecognized_complex_embedding_from_shape,
    ),
]


NumberFieldEmbedding = Annotated[
    RealNumberFieldEmbedding | ComplexNumberFieldEmbedding,
    Field(discriminator="kind"),
]


class RealNumberFieldEmbeddingRecord(StrictModel):
    """A real embedding together with exact rational isolation evidence."""

    kind: Literal["REAL"]
    embedding: RealNumberFieldEmbedding
    isolating_interval: RationalIsolatingInterval

    @model_validator(mode="before")
    @classmethod
    def require_raw_isolator_bound(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data
        interval = data.get("isolating_interval")
        if isinstance(interval, Mapping):
            for endpoint_name in ("lower", "upper"):
                endpoint = interval.get(endpoint_name)
                if not isinstance(endpoint, Mapping):
                    continue
                for part in ("num", "den"):
                    raw = endpoint.get(part)
                    if isinstance(raw, str) and len(raw.lstrip("-")) > (
                        MAX_NUMBER_FIELD_ISOLATOR_COMPONENT_DIGITS
                    ):
                        raise _validation_error(
                            "isolator_component_bound",
                            "real isolator components exceed the "
                            f"{MAX_NUMBER_FIELD_ISOLATOR_COMPONENT_DIGITS:,}-digit bound",
                        )
        return canonicalize_json_containers(data)

    @model_validator(mode="after")
    def require_canonical_isolator_bound(self) -> Self:
        if any(
            len(endpoint.num.lstrip("-")) > MAX_NUMBER_FIELD_ISOLATOR_COMPONENT_DIGITS
            or len(endpoint.den) > MAX_NUMBER_FIELD_ISOLATOR_COMPONENT_DIGITS
            for endpoint in (
                self.isolating_interval.lower,
                self.isolating_interval.upper,
            )
        ):
            raise _validation_error(
                "isolator_component_bound",
                "real isolator components exceed the "
                f"{MAX_NUMBER_FIELD_ISOLATOR_COMPONENT_DIGITS:,}-digit bound",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        embedding: RealNumberFieldEmbedding,
        isolating_interval: RationalIsolatingInterval,
    ) -> Self:
        return cls.model_construct(
            kind="REAL",
            embedding=embedding,
            isolating_interval=isolating_interval,
        )


class ComplexNumberFieldEmbeddingRecord(StrictModel):
    """A nonreal embedding together with exact rational isolation evidence."""

    kind: Literal["COMPLEX"]
    embedding: _UnrecognizedComplexNumberFieldEmbedding
    isolating_rectangle: RationalComplexIsolatingRectangle
    half_plane: Literal["NEGATIVE_IMAGINARY", "POSITIVE_IMAGINARY"]

    @model_validator(mode="after")
    def bind_declared_half_plane(self) -> Self:
        imaginary_lower = self.isolating_rectangle.imaginary_lower.as_fraction()
        imaginary_upper = self.isolating_rectangle.imaginary_upper.as_fraction()
        expected_half_plane: Literal["NEGATIVE_IMAGINARY", "POSITIVE_IMAGINARY"]
        if imaginary_upper < 0:
            expected_half_plane = "NEGATIVE_IMAGINARY"
        elif imaginary_lower > 0:
            expected_half_plane = "POSITIVE_IMAGINARY"
        else:
            raise _validation_error(
                "complex_half_plane",
                "a nonreal embedding isolator must lie wholly in one open half-plane",
            )
        if self.half_plane != expected_half_plane:
            raise _validation_error(
                "complex_half_plane",
                "complex embedding half-plane must agree with its exact isolator",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        embedding: ComplexNumberFieldEmbedding,
        isolating_rectangle: RationalComplexIsolatingRectangle,
        half_plane: Literal["NEGATIVE_IMAGINARY", "POSITIVE_IMAGINARY"],
    ) -> Self:
        return cls.model_construct(
            kind="COMPLEX",
            embedding=embedding,
            isolating_rectangle=isolating_rectangle,
            half_plane=half_plane,
        )


NumberFieldEmbeddingRecord = Annotated[
    RealNumberFieldEmbeddingRecord | ComplexNumberFieldEmbeddingRecord,
    Field(discriminator="kind"),
]


class NumberFieldSignature(StrictModel):
    real_embedding_count: StrictInt = Field(ge=0, le=MAX_NUMBER_FIELD_EMBEDDING_DEGREE)
    complex_conjugate_pair_count: StrictInt = Field(
        ge=0, le=MAX_NUMBER_FIELD_EMBEDDING_DEGREE // 2
    )


class NumberFieldConjugatePair(StrictModel):
    negative_embedding_index: StrictInt = Field(
        ge=0, le=MAX_NUMBER_FIELD_EMBEDDING_DEGREE - 1
    )
    positive_embedding_index: StrictInt = Field(
        ge=0, le=MAX_NUMBER_FIELD_EMBEDDING_DEGREE - 1
    )


class NumberFieldEmbeddingProfile(StrictModel):
    """Every Archimedean embedding of one presented simple number field.

    Exact embedding identity is carried by ``presentation + indexed root``.
    Isolation data is evidence for that identity and is deliberately kept in
    records rather than in the canonical embedding value itself.
    """

    field: SimpleNumberFieldPresentation
    records: tuple[NumberFieldEmbeddingRecord, ...] = Field(
        min_length=1, max_length=MAX_NUMBER_FIELD_EMBEDDING_DEGREE
    )
    signature: NumberFieldSignature
    complex_conjugate_pairs: tuple[NumberFieldConjugatePair, ...] = Field(
        max_length=MAX_NUMBER_FIELD_EMBEDDING_DEGREE // 2
    )
    defining_polynomial_discriminant: CanonicalInteger
    ordering: Literal["REAL_INCREASING_THEN_POSITIVE_REPRESENTATIVE_PAIRS_V1"] = (
        "REAL_INCREASING_THEN_POSITIVE_REPRESENTATIVE_PAIRS_V1"
    )

    @model_validator(mode="after")
    def bind_complete_profile(self) -> Self:
        if len(self.records) != self.field.degree:
            raise _validation_error(
                "embedding_count",
                "a complete profile needs exactly degree-many embedding records",
            )
        if any(record.embedding.presentation != self.field for record in self.records):
            raise _validation_error(
                "embedding_field",
                "every embedding record must belong to the profile field",
            )

        real_count = self.signature.real_embedding_count
        pair_count = self.signature.complex_conjugate_pair_count
        if real_count + 2 * pair_count != self.field.degree:
            raise _validation_error(
                "signature_degree",
                "signature must satisfy r1 + 2*r2 = field degree",
            )
        for index, record in enumerate(self.records[:real_count]):
            if not isinstance(record, RealNumberFieldEmbeddingRecord):
                raise _validation_error(
                    "real_embedding_prefix",
                    "all real embeddings must precede nonreal embeddings",
                )
            if record.embedding.root.real_root_index != index:
                raise _validation_error(
                    "real_root_order",
                    "real embedding records must follow increasing root order",
                )

        expected_pairs: list[NumberFieldConjugatePair] = []
        for pair_offset in range(pair_count):
            negative_index = real_count + 2 * pair_offset
            positive_index = negative_index + 1
            negative_record = self.records[negative_index]
            positive_record = self.records[positive_index]
            if not isinstance(
                negative_record, ComplexNumberFieldEmbeddingRecord
            ) or not isinstance(positive_record, ComplexNumberFieldEmbeddingRecord):
                raise _validation_error(
                    "complex_embedding_suffix",
                    "the nonreal suffix must consist of complex embedding pairs",
                )
            if (
                negative_record.embedding.root.root_index != negative_index
                or positive_record.embedding.root.root_index != positive_index
                or negative_record.half_plane != "NEGATIVE_IMAGINARY"
                or positive_record.half_plane != "POSITIVE_IMAGINARY"
            ):
                raise _validation_error(
                    "complex_root_order",
                    "each complex pair must list its negative then positive root",
                )
            negative_rectangle = negative_record.isolating_rectangle
            positive_rectangle = positive_record.isolating_rectangle
            if negative_rectangle.conjugate() != positive_rectangle:
                raise _validation_error(
                    "complex_conjugacy",
                    "each grouped pair must contain exact conjugate roots and evidence",
                )
            expected_pairs.append(
                NumberFieldConjugatePair(
                    negative_embedding_index=negative_index,
                    positive_embedding_index=positive_index,
                )
            )
        if self.complex_conjugate_pairs != tuple(expected_pairs):
            raise _validation_error(
                "complex_pair_grouping",
                "complex conjugate-pair grouping must cover the ordered nonreal suffix",
            )

        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        field: SimpleNumberFieldPresentation,
        records: tuple[
            RealNumberFieldEmbeddingRecord | ComplexNumberFieldEmbeddingRecord, ...
        ],
        signature: NumberFieldSignature,
        complex_conjugate_pairs: tuple[NumberFieldConjugatePair, ...],
        defining_polynomial_discriminant: CanonicalInteger,
    ) -> Self:
        return cls.model_construct(
            field=field,
            records=records,
            signature=signature,
            complex_conjugate_pairs=complex_conjugate_pairs,
            defining_polynomial_discriminant=defining_polynomial_discriminant,
            ordering="REAL_INCREASING_THEN_POSITIVE_REPRESENTATIVE_PAIRS_V1",
        )


__all__ = [
    "MAX_NUMBER_FIELD_EMBEDDING_DEGREE",
    "MAX_NUMBER_FIELD_ISOLATOR_COMPONENT_DIGITS",
    "MAX_SIMPLE_NUMBER_FIELD_COEFFICIENT_DIGITS",
    "MAX_SIMPLE_NUMBER_FIELD_DEGREE",
    "MAX_SIMPLE_NUMBER_FIELD_ELEMENT_DIGITS",
    "ComplexNumberFieldEmbeddingRecord",
    "NumberFieldConjugatePair",
    "NumberFieldEmbedding",
    "NumberFieldEmbeddingProfile",
    "NumberFieldEmbeddingRecord",
    "NumberFieldSignature",
    "RealNumberFieldEmbedding",
    "RealNumberFieldEmbeddingRecord",
    "SimpleNumberFieldElement",
    "SimpleNumberFieldPresentation",
]
