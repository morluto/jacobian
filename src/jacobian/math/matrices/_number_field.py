"""Private exact arithmetic for matrices over one embedded simple number field."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from sympy import QQ, CRootOf, Poly, Symbol
from sympy.polys.domains import AlgebraicField
from sympy.polys.matrices import DomainMatrix

from jacobian._exact import CanonicalRational
from jacobian.math._root_isolation import strict_root_count
from jacobian.math.matrices.values import EmbeddedRealSimpleNumberFieldMatrix
from jacobian.math.number_theory.number_fields.values import (
    RealNumberFieldEmbedding,
    SimpleNumberFieldElement,
)


class EmbeddedNumberFieldRecognitionError(ValueError):
    """A structurally valid embedding failed mathematical recognition."""

    def __init__(self, reason: str, message: str) -> None:
        self.reason = reason
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class RecognizedRealSimpleNumberField:
    """One admitted abstract field together with its selected real root."""

    embedding: RealNumberFieldEmbedding
    field: AlgebraicField

    @property
    def degree(self) -> int:
        return self.embedding.presentation.degree


def recognize_real_simple_number_field(
    embedding: RealNumberFieldEmbedding,
) -> RecognizedRealSimpleNumberField:
    """Recognize irreducibility and the indexed real root exactly once."""

    x = Symbol("x")
    polynomial = Poly.from_list(
        list(embedding.presentation.coefficients_descending),
        gens=x,
        domain=QQ,
    )
    if polynomial.is_irreducible is not True or (
        embedding.root.real_root_index >= len(polynomial.intervals())
    ):
        raise EmbeddedNumberFieldRecognitionError(
            "invalid_embedding",
            "the selected number-field embedding is not a recognized real root",
        )
    embedded_root = CRootOf(polynomial, embedding.root.real_root_index)
    field = QQ.algebraic_field(embedded_root, alias="alpha")
    return RecognizedRealSimpleNumberField(
        embedding=embedding,
        field=field,
    )


def field_element_from_value(
    value: SimpleNumberFieldElement,
    recognized: RecognizedRealSimpleNumberField,
) -> Any:
    """Convert canonical ascending coordinates to SymPy's exact ANP value."""

    coefficients_descending = [
        QQ(
            coefficient.num,
            coefficient.den,
        )
        for coefficient in reversed(value.coefficients_ascending)
    ]
    return recognized.field.new(coefficients_descending)


def field_element_coordinates(
    value: Any,
    recognized: RecognizedRealSimpleNumberField,
) -> tuple[Fraction, ...]:
    """Return one ANP value in the canonical ascending rational power basis."""

    descending = list(value.to_list())
    descending = [recognized.field.domain.zero] * (
        recognized.degree - len(descending)
    ) + descending
    return tuple(
        Fraction(int(coefficient.numerator), int(coefficient.denominator))
        for coefficient in reversed(descending)
    )


def field_element_sign(
    value: Any,
    recognized: RecognizedRealSimpleNumberField,
) -> int:
    """Return the exact sign of one field element at the selected real root.

    If ``f`` is the irreducible degree-``d`` defining polynomial and ``p`` is
    the nonzero reduced degree-``< d`` representative of the element, then
    ``f`` and ``p`` are coprime.  SymPy's exact rational isolation of ``f*p``
    therefore puts the selected root of ``f`` in an interval containing no
    root of ``p``.  The sign at any rational point in that interval is the
    required embedded sign.  The admitted degree and height envelope charges
    isolation of the degree-at-most-``2d-1`` product before this call.
    """

    coordinates = field_element_coordinates(value, recognized)
    if not any(coordinates):
        return 0
    x = Symbol("x")
    defining = Poly.from_list(
        list(recognized.embedding.presentation.coefficients_descending),
        gens=x,
        domain=QQ,
    )
    representative = Poly.from_list(
        [
            QQ(coordinate.numerator, coordinate.denominator)
            for coordinate in reversed(coordinates)
        ],
        gens=x,
        domain=QQ,
    )
    selected_index = recognized.embedding.root.real_root_index
    defining_roots_seen = 0
    for (lower, upper), _multiplicity in (defining * representative).intervals():
        if not strict_root_count(defining, lower, upper):
            continue
        if defining_roots_seen != selected_index:
            defining_roots_seen += 1
            continue
        sample = lower if lower == upper else (lower + upper) / 2
        exact_value = representative.eval(sample)
        if exact_value > 0:
            return 1
        if exact_value < 0:
            return -1
        # Coprimality makes this unreachable for an exact isolating interval.
        break
    raise EmbeddedNumberFieldRecognitionError(
        "sign_isolation",
        "exact root isolation did not retain the selected real embedding",
    )


def domain_matrix_from_embedded(
    matrix: EmbeddedRealSimpleNumberFieldMatrix,
    recognized: RecognizedRealSimpleNumberField,
) -> DomainMatrix:
    """Convert a canonical embedded matrix to a dense exact DomainMatrix."""

    if matrix.embedding != recognized.embedding:
        raise EmbeddedNumberFieldRecognitionError(
            "embedding_mismatch",
            "the recognized field must use the matrix's selected embedding",
        )
    rows = [
        [field_element_from_value(value, recognized) for value in row]
        for row in matrix.entries
    ]
    return DomainMatrix(
        rows,
        (matrix.row_count, matrix.column_count),
        recognized.field,
    )


def simple_number_field_element_from_field(
    value: Any,
    recognized: RecognizedRealSimpleNumberField,
) -> SimpleNumberFieldElement:
    """Convert one admitted ANP value back to the canonical field element."""

    return SimpleNumberFieldElement(
        presentation=recognized.embedding.presentation,
        coefficients_ascending=tuple(
            CanonicalRational.from_fraction(coordinate)
            for coordinate in field_element_coordinates(value, recognized)
        ),
    )


def embedded_matrix_from_domain(
    matrix: DomainMatrix,
    recognized: RecognizedRealSimpleNumberField,
) -> EmbeddedRealSimpleNumberFieldMatrix:
    """Convert one admitted dense DomainMatrix to the canonical public value."""

    dense = matrix.to_dense().rep.to_ddm()
    return EmbeddedRealSimpleNumberFieldMatrix(
        embedding=recognized.embedding,
        row_count=matrix.shape[0],
        column_count=matrix.shape[1],
        entries=tuple(
            tuple(
                simple_number_field_element_from_field(value, recognized)
                for value in row
            )
            for row in dense
        ),
    )


__all__ = [
    "EmbeddedNumberFieldRecognitionError",
    "RecognizedRealSimpleNumberField",
    "domain_matrix_from_embedded",
    "embedded_matrix_from_domain",
    "field_element_coordinates",
    "field_element_from_value",
    "field_element_sign",
    "recognize_real_simple_number_field",
]
