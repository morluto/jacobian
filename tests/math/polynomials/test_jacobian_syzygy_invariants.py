from __future__ import annotations

import json
from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.math.geometry.projective.values import RationalProjectiveLine
from jacobian.math.polynomials._jacobian_syzygy import (
    compute_graded_jacobian_syzygy,
    compute_graded_jacobian_syzygy_coefficients,
)
from jacobian.math.polynomials._syzygy_models import (
    GradedJacobianSyzygyCoefficientRequest,
    GradedJacobianSyzygyRequest,
    GradedJacobianSyzygyResult,
)
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)


def _rational(value: Fraction) -> CanonicalRational:
    return CanonicalRational.from_fraction(value)


def _four_variable_counterexample() -> RationalPolynomial:
    """Expand the nine specified linear forms without using the producer."""

    factors = (
        ((1, 0, 0, 0),),
        ((0, 1, 0, 0),),
        ((0, 0, 1, 0),),
        ((0, 0, 0, 1),),
        ((0, 1, 0, 1),),
        ((0, 0, 1, 1),),
        ((-1, 1, 0, 1),),
        ((-1, 0, 1, 1),),
        ((-1, 1, 1, 1),),
    )
    terms: dict[tuple[int, int, int, int], Fraction] = {(0, 0, 0, 0): Fraction(1)}
    for factor in factors:
        product: dict[tuple[int, int, int, int], Fraction] = {}
        for exponents, coefficient in terms.items():
            for factor_exponents in factor:
                # Every listed factor has coefficients in {-1, 0, 1} encoded by
                # its nonzero variable terms below.
                for variable, factor_coefficient in enumerate(factor_exponents):
                    if not factor_coefficient:
                        continue
                    target = list(exponents)
                    target[variable] += 1
                    target_tuple = tuple(target)
                    product[target_tuple] = product.get(target_tuple, Fraction(0)) + (
                        coefficient * factor_coefficient
                    )
        terms = {
            exponents: coefficient
            for exponents, coefficient in product.items()
            if coefficient
        }
    return RationalPolynomial(
        variables=("x", "y", "z", "w"),
        polynomial=SparseRationalPolynomial(
            terms=tuple(
                RationalPolynomialTerm(
                    coefficient=_rational(coefficient), exponents=exponents
                )
                for exponents, coefficient in sorted(terms.items(), reverse=True)
            )
        ),
    )


def _sparse_polynomial(
    variables: tuple[str, ...], terms: dict[tuple[int, ...], int]
) -> RationalPolynomial:
    return RationalPolynomial(
        variables=variables,
        polynomial=SparseRationalPolynomial(
            terms=tuple(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational(num=str(coefficient), den="1"),
                    exponents=exponents,
                )
                for exponents, coefficient in sorted(terms.items(), reverse=True)
            )
        ),
    )


LEGACY_THREE_VARIABLE_CERTIFICATE_PAYLOAD = {
    "polynomial": {
        "variables": ["x", "y", "z"],
        "polynomial": {
            "terms": [
                {
                    "coefficient": {"num": "1", "den": "1"},
                    "exponents": [1, 1, 1],
                }
            ]
        },
    },
    "max_degree": 1,
    "coefficient_map_detail": "CERTIFICATES",
}


def test_four_variable_rank_fixture_has_the_published_first_kernel() -> None:
    result = compute_graded_jacobian_syzygy(
        GradedJacobianSyzygyRequest(
            polynomial=_four_variable_counterexample(), max_degree=3
        )
    )

    assert result.variables == ("x", "y", "z", "w")
    assert result.first_syzygy_degree == 3
    assert result.status == "FOUND"
    assert result.kernel_witness is not None
    assert len(result.kernel_witness.multipliers) == 4
    assert [
        (item.row_count, item.column_count, item.rank, item.nullity)
        for item in result.degree_maps
    ] == [
        (165, 4, 4, 0),
        (220, 16, 16, 0),
        (286, 40, 40, 0),
        (364, 80, 76, 4),
    ]

    # Revalidation replays every returned partial and the exact witness equation.
    assert (
        GradedJacobianSyzygyResult.model_validate(result.model_dump(mode="python"))
        == result
    )


def test_three_variable_sparse_and_labelled_inputs_remain_compatible() -> None:
    sparse_result = compute_graded_jacobian_syzygy(
        GradedJacobianSyzygyRequest(
            polynomial=_sparse_polynomial(
                ("x", "y", "z"),
                {(1, 1, 1): 1},
            ),
            max_degree=1,
        )
    )
    labelled_result = compute_graded_jacobian_syzygy(
        GradedJacobianSyzygyRequest(
            linear_factors=(
                RationalProjectiveLine(
                    label="x",
                    coefficients=(
                        _rational(Fraction(1)),
                        _rational(Fraction(0)),
                        _rational(Fraction(0)),
                    ),
                ),
                RationalProjectiveLine(
                    label="y",
                    coefficients=(
                        _rational(Fraction(0)),
                        _rational(Fraction(1)),
                        _rational(Fraction(0)),
                    ),
                ),
                RationalProjectiveLine(
                    label="z",
                    coefficients=(
                        _rational(Fraction(0)),
                        _rational(Fraction(0)),
                        _rational(Fraction(1)),
                    ),
                ),
            ),
            linear_factor_variables=("x", "y", "z"),
            max_degree=1,
        )
    )

    sparse_payload = sparse_result.model_dump()
    labelled_payload = labelled_result.model_dump()
    assert sparse_payload.pop("source_kind") == "EXPANDED_POLYNOMIAL"
    assert labelled_payload.pop("source_kind") == "LABELLED_LINEAR_FACTOR_PRODUCT"
    assert sparse_payload == labelled_payload
    assert sparse_result.first_syzygy_degree == 1
    assert [item.column_count for item in sparse_result.degree_maps] == [3, 9]


def test_frozen_legacy_three_variable_certificate_payload_remains_version_one() -> None:
    result = compute_graded_jacobian_syzygy(
        GradedJacobianSyzygyRequest.model_validate(
            LEGACY_THREE_VARIABLE_CERTIFICATE_PAYLOAD
        )
    )

    assert result.result_schema_version == "1"
    assert result.coefficient_map_detail == "CERTIFICATES"
    assert result.first_syzygy_degree == 1
    assert [
        (item.row_count, item.column_count, item.rank, item.nullity)
        for item in result.degree_maps
    ] == [(6, 3, 3, 0), (10, 9, 7, 2)]


def test_detail_mode_is_part_of_each_operation_request_contract() -> None:
    sparse_payload = {
        **LEGACY_THREE_VARIABLE_CERTIFICATE_PAYLOAD,
        "coefficient_map_detail": "SPARSE_ENTRIES",
    }
    with pytest.raises(ValidationError, match="CERTIFICATES"):
        GradedJacobianSyzygyRequest.model_validate(sparse_payload)

    with pytest.raises(ValidationError, match="SPARSE_ENTRIES"):
        GradedJacobianSyzygyCoefficientRequest.model_validate(
            LEGACY_THREE_VARIABLE_CERTIFICATE_PAYLOAD
        )


def test_sparse_coefficient_ledger_uses_four_variable_exponent_vectors() -> None:
    result = compute_graded_jacobian_syzygy_coefficients(
        GradedJacobianSyzygyCoefficientRequest(
            polynomial=_sparse_polynomial(("x", "y", "z", "w"), {(1, 1, 1, 1): 1}),
            max_degree=0,
        )
    )

    coefficient_map = result.degree_maps[0]
    assert coefficient_map.row_count == 20
    assert coefficient_map.column_count == 4
    assert coefficient_map.sparse_entries
    assert all(
        len(exponents) == 4
        for exponents in (
            *coefficient_map.source_monomial_basis,
            *coefficient_map.target_monomial_basis,
        )
    )


def test_result_rejects_a_mutated_witness_or_partial_derivative() -> None:
    result = compute_graded_jacobian_syzygy(
        GradedJacobianSyzygyRequest(
            polynomial=_sparse_polynomial(("x", "y", "z"), {(1, 1, 1): 1}),
            max_degree=1,
        )
    )
    witness = result.kernel_witness
    assert witness is not None
    first_nonzero = next(
        index
        for index, coefficient in enumerate(witness.coefficient_vector)
        if coefficient.as_fraction()
    )
    corrupted_witness = json.loads(result.model_dump_json())
    corrupted_witness["kernel_witness"]["coefficient_vector"][first_nonzero] = {
        "num": "2",
        "den": "1",
    }
    with pytest.raises(ValidationError, match="kernel vector must reconstruct"):
        GradedJacobianSyzygyResult.model_validate(corrupted_witness)

    corrupted_partial = json.loads(result.model_dump_json())
    corrupted_partial["partial_derivatives"][0]["polynomial"]["terms"][0][
        "coefficient"
    ] = {"num": "2", "den": "1"}
    with pytest.raises(ValidationError, match="partial derivatives must reconstruct"):
        GradedJacobianSyzygyResult.model_validate(corrupted_partial)

    corrupted_map = json.loads(result.model_dump_json())
    corrupted_map["degree_maps"][0]["matrix_digest"] = "sha256:" + "0" * 64
    with pytest.raises(ValidationError, match="digest must bind"):
        GradedJacobianSyzygyResult.model_validate(corrupted_map)


def test_dimension_specific_basis_boundary_rejects_before_backend_execution() -> None:
    with pytest.raises(ValidationError, match="512-monomial"):
        GradedJacobianSyzygyRequest(
            polynomial=_sparse_polynomial(
                ("x", "y", "z", "w"),
                {
                    (11, 0, 0, 0): 1,
                    (0, 11, 0, 0): 1,
                    (0, 0, 11, 0): 1,
                    (0, 0, 0, 11): 1,
                },
            ),
            max_degree=3,
        )

    with pytest.raises(ValidationError, match="15000000-update"):
        GradedJacobianSyzygyRequest(
            polynomial=_sparse_polynomial(
                ("x", "y", "z", "w"),
                {(9, 0, 0, 0): 1, (0, 9, 0, 0): 1, (0, 0, 9, 0): 1, (0, 0, 0, 9): 1},
            ),
            max_degree=4,
        )


def test_syzygy_kernel_rejects_an_incomplete_linear_factor_request() -> None:
    request = GradedJacobianSyzygyRequest.model_construct(
        polynomial=None,
        linear_factors=None,
        linear_factor_variables=None,
        max_degree=0,
        coefficient_map_detail="CERTIFICATES",
    )

    with pytest.raises(ValueError, match="linear-factor input is incomplete"):
        compute_graded_jacobian_syzygy(request)


def test_zero_partial_derivatives_admit_the_forced_degree_zero_kernel() -> None:
    request = GradedJacobianSyzygyRequest(
        polynomial=_sparse_polynomial(("a", "b", "c", "d", "e"), {(1, 0, 0, 0, 0): 1}),
    )

    assert request.max_degree == 6
    result = compute_graded_jacobian_syzygy(request)

    assert result.status == "FOUND"
    assert result.first_syzygy_degree == 0
    assert result.searched_through_degree == 0
    assert [
        (item.row_count, item.column_count, item.rank, item.nullity)
        for item in result.degree_maps
    ] == [(1, 5, 1, 4)]
    assert (
        GradedJacobianSyzygyResult.model_validate(result.model_dump(mode="python"))
        == result
    )


def test_default_bound_admits_eight_variables_when_the_kernel_is_forced() -> None:
    request = GradedJacobianSyzygyRequest(
        polynomial=_sparse_polynomial(
            ("a", "b", "c", "d", "e", "f", "g", "h"),
            {(1, 0, 0, 0, 0, 0, 0, 0): 1},
        ),
        max_degree=8,
    )
    result = compute_graded_jacobian_syzygy(request)

    assert result.first_syzygy_degree == 0
    assert [item.column_count for item in result.degree_maps] == [8]


def test_dependent_gradient_without_zero_partials_stops_at_degree_zero() -> None:
    result = compute_graded_jacobian_syzygy(
        GradedJacobianSyzygyRequest(
            polynomial=_sparse_polynomial(
                ("x", "y", "z"),
                {(2, 0, 2): 1, (1, 1, 2): -2, (0, 2, 2): 1},
            ),
            max_degree=6,
        )
    )

    assert result.status == "FOUND"
    assert result.first_syzygy_degree == 0
    assert len(result.degree_maps) == 1
    witness = result.kernel_witness
    assert witness is not None
    assert len(witness.multipliers) == 3


def test_result_rejects_labelled_provenance_off_three_variables() -> None:
    for variables in (("x",), ("x", "y"), ("x", "y", "z", "w")):
        source = _sparse_polynomial(
            variables,
            {tuple(1 if index == 0 else 0 for index in range(len(variables))): 1},
        )
        produced = compute_graded_jacobian_syzygy_coefficients(
            GradedJacobianSyzygyCoefficientRequest(
                polynomial=source,
                max_degree=1,
                coefficient_map_detail="SPARSE_ENTRIES",
            )
        )
        payload = json.loads(produced.model_dump_json())
        assert GradedJacobianSyzygyResult.model_validate(payload) == produced

        payload["source_kind"] = "LABELLED_LINEAR_FACTOR_PRODUCT"
        with pytest.raises(ValidationError, match="requires exactly three variables"):
            GradedJacobianSyzygyResult.model_validate(payload)
