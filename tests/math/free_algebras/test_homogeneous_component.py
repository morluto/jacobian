"""Exact graded projections of sparse noncommutative polynomials."""

import json
from collections import Counter
from itertools import islice, product

import pytest

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.free_algebras._models import FreeAlgebraPolynomial
from jacobian.math.free_algebras.homogeneous_component._models import (
    FreeAlgebraHomogeneousComponent,
)
from jacobian.math.free_algebras.homogeneous_component.operations import (
    homogeneous_component,
)


def _term(word: list[str], coefficient: int = 1) -> dict[str, object]:
    return {
        "coefficient": {"num": coefficient, "den": 1},
        "word": word,
    }


def _polynomial() -> FreeAlgebraPolynomial:
    return FreeAlgebraPolynomial.model_validate(
        {
            "alphabet": ["x", "y"],
            "terms": [
                _term(["x", "y"], 3),
                _term(["y"], 2),
                _term(["x"], 1),
                _term([], 5),
            ],
        }
    )


def test_projection_preserves_exact_terms_parent_and_degree_after_round_trip() -> None:
    result = homogeneous_component(_polynomial(), 1)
    restored = FreeAlgebraHomogeneousComponent.model_validate(result.model_dump())
    assert restored.degree == 1
    assert restored.polynomial.alphabet == ("x", "y")
    assert [term.word for term in restored.polynomial.terms] == [("y",), ("x",)]
    assert [term.coefficient.num for term in restored.polynomial.terms] == [2, 1]


def test_degree_family_reconstructs_source_and_preserves_zero_parent() -> None:
    source = _polynomial()
    components = [homogeneous_component(source, degree) for degree in range(4)]
    projected = Counter(
        (term.word, term.coefficient.num, term.coefficient.den)
        for component in components
        for term in component.polynomial.terms
    )
    original = Counter(
        (term.word, term.coefficient.num, term.coefficient.den) for term in source.terms
    )
    assert projected == original

    missing = homogeneous_component(source, 3)
    assert missing.degree == 3
    assert missing.polynomial.is_zero
    assert missing.polynomial.alphabet == source.alphabet


def test_native_projection_composes_with_polynomial_multiplication() -> None:
    component = FreeAlgebraHomogeneousComponent.model_validate_json(
        json.dumps(homogeneous_component(_polynomial(), 1).model_dump(mode="json"))
    )
    from jacobian.math.free_algebras.operations import multiply

    product_result = multiply(
        component.polynomial,
        FreeAlgebraPolynomial.model_validate(
            {"alphabet": ["x", "y"], "terms": [_term(["x"])]}
        ),
    )
    words = [term.word for term in product_result.product.terms]
    assert words == [("y", "x"), ("x", "x")]


def test_large_selected_support_is_admitted_by_mathematical_bounds() -> None:
    alphabet = tuple(chr(ord("A") + index) * 64 for index in range(26))
    words = list(islice(product(alphabet, repeat=8), 4096))
    polynomial = FreeAlgebraPolynomial.model_validate(
        {
            "alphabet": alphabet,
            "terms": [_term(list(word)) for word in reversed(words)],
        }
    )
    result = homogeneous_component(polynomial, 8)
    assert len(result.polynomial.terms) == len(words)


def test_native_operation_rejects_invalid_or_over_bound_degree() -> None:
    polynomial = _polynomial()
    with pytest.raises(OperationDomainValidationError, match="nonnegative integer"):
        homogeneous_component(polynomial, -1)
    assert homogeneous_component(polynomial, 65).polynomial.is_zero
    with pytest.raises(OperationDomainValidationError, match="nonnegative integer"):
        homogeneous_component(polynomial, True)
    forged = FreeAlgebraPolynomial.model_construct(alphabet=("x", "x"), terms=())
    with pytest.raises(OperationDomainValidationError, match="not canonical"):
        homogeneous_component(forged, 0)
    oversized = FreeAlgebraPolynomial.model_construct(
        alphabet=(), terms=(None,) * 600001
    )
    with pytest.raises(OperationResourceAdmissionError, match="scan bound"):
        homogeneous_component(oversized, 0)
