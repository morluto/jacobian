from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from jacobian.math.polynomials._jacobian_syzygy import (
    compute_graded_jacobian_syzygy,
    compute_graded_jacobian_syzygy_coefficients,
)
from jacobian.math.polynomials._syzygy_models import (
    GradedJacobianSyzygyRequest,
    GradedJacobianSyzygyResult,
)

ZERO_POLYNOMIAL = {"variables": ["x", "y", "z"], "polynomial": {"terms": []}}

FERMAT_QUADRATIC_TERMS = [
    {"coefficient": {"num": "1", "den": "1"}, "exponents": [2, 0, 0]},
    {"coefficient": {"num": "1", "den": "1"}, "exponents": [0, 2, 0]},
    {"coefficient": {"num": "1", "den": "1"}, "exponents": [0, 0, 2]},
]

EXPANDED_XYZ_TERMS = [
    {"coefficient": {"num": "1", "den": "1"}, "exponents": [1, 1, 0]},
    {"coefficient": {"num": "1", "den": "1"}, "exponents": [1, 0, 1]},
    {"coefficient": {"num": "1", "den": "1"}, "exponents": [0, 1, 1]},
]


def _sparse_request(terms: list[dict[str, Any]], max_degree: int) -> dict[str, Any]:
    return {
        "polynomial": {
            "variables": ["x", "y", "z"],
            "polynomial": {"terms": terms},
        },
        "max_degree": max_degree,
    }


def _none_payload() -> dict[str, Any]:
    request = GradedJacobianSyzygyRequest.model_validate(
        _sparse_request(FERMAT_QUADRATIC_TERMS, 0)
    )
    return compute_graded_jacobian_syzygy(request).model_dump(mode="json")


def _found_payload() -> dict[str, Any]:
    request = GradedJacobianSyzygyRequest.model_validate(
        _sparse_request(EXPANDED_XYZ_TERMS, 1)
    )
    return compute_graded_jacobian_syzygy(request).model_dump(mode="json")


def _ledger_payload() -> dict[str, Any]:
    request = GradedJacobianSyzygyRequest.model_validate(
        _sparse_request(FERMAT_QUADRATIC_TERMS, 0)
    )
    return compute_graded_jacobian_syzygy_coefficients(request).model_dump(mode="json")


def _swap_in_foreign_source(payload: dict[str, Any]) -> None:
    payload["expanded_polynomial"] = _none_payload()["expanded_polynomial"]


def _scale_first_partial_derivative(payload: dict[str, Any]) -> None:
    payload["partial_derivatives"][0]["polynomial"]["terms"][0]["coefficient"][
        "num"
    ] = "3"


def _zero_second_partial_derivative(payload: dict[str, Any]) -> None:
    payload["partial_derivatives"][1] = ZERO_POLYNOMIAL


def _replace_coefficient_map_digest(payload: dict[str, Any]) -> None:
    payload["degree_maps"][0]["matrix_digest"] = "sha256:" + "0" * 64


def _shrink_rank_bookkeeping_coherently(payload: dict[str, Any]) -> None:
    degree_map = payload["degree_maps"][-1]
    degree_map["rank"] = degree_map["rank"] - 1
    degree_map["nullity"] = degree_map["nullity"] + 1
    degree_map["pivot_columns"] = degree_map["pivot_columns"][:-1]
    minor = degree_map["rank_minor"]
    minor["row_indices"] = minor["row_indices"][:-1]
    minor["column_indices"] = minor["column_indices"][:-1]


def _mutate_kernel_coefficient_vector(payload: dict[str, Any]) -> None:
    payload["kernel_witness"]["coefficient_vector"][0]["num"] = "5"


def _mutate_kernel_multiplier_encoding(payload: dict[str, Any]) -> None:
    payload["kernel_witness"]["multipliers"][0]["polynomial"]["terms"][0][
        "coefficient"
    ]["num"] = "2"


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


def test_syzygy_result_retains_exact_none_through_bound_evidence() -> None:
    request = GradedJacobianSyzygyRequest.model_validate(
        _sparse_request(FERMAT_QUADRATIC_TERMS, 0)
    )

    result = compute_graded_jacobian_syzygy(request)

    assert result.status == "NONE_THROUGH_BOUND"
    assert result.kernel_witness is None
    terms = result.partial_derivatives[0].polynomial.terms
    assert [term.exponents for term in terms] == [(1, 0, 0)]
    assert terms[0].coefficient.num == "2"
    degree_map = result.degree_maps[0]
    assert degree_map.rank == 3
    assert degree_map.nullity == 0
    assert degree_map.injective
    assert degree_map.pivot_columns == (0, 1, 2)
    assert degree_map.rank_minor is not None
    assert degree_map.rank_minor.determinant.num == "8"
    GradedJacobianSyzygyResult.model_validate(result.model_dump(mode="json"))


def test_syzygy_none_through_bound_backs_every_degree_with_a_certificate() -> None:
    request = GradedJacobianSyzygyRequest.model_validate(
        _sparse_request(
            [
                {"coefficient": {"num": "1", "den": "1"}, "exponents": [3, 0, 0]},
                {"coefficient": {"num": "1", "den": "1"}, "exponents": [0, 3, 0]},
                {"coefficient": {"num": "1", "den": "1"}, "exponents": [0, 0, 3]},
            ],
            1,
        )
    )

    result = compute_graded_jacobian_syzygy(request)

    assert result.status == "NONE_THROUGH_BOUND"
    assert result.searched_through_degree == 1
    assert all(item.injective for item in result.degree_maps)
    assert all(item.rank_minor is not None for item in result.degree_maps)
    GradedJacobianSyzygyResult.model_validate(result.model_dump(mode="json"))


def test_syzygy_found_result_binds_the_first_kernel_witness() -> None:
    request = GradedJacobianSyzygyRequest.model_validate(
        _sparse_request(EXPANDED_XYZ_TERMS, 1)
    )

    result = compute_graded_jacobian_syzygy(request)

    assert result.status == "FOUND"
    assert result.first_syzygy_degree == 1
    assert result.searched_through_degree == 1
    witness = result.kernel_witness
    assert witness is not None
    assert witness.multiplier_degree == 1
    assert len(witness.coefficient_vector) == result.degree_maps[1].column_count
    GradedJacobianSyzygyResult.model_validate(result.model_dump(mode="json"))


def test_syzygy_result_rejects_mutated_derivatives_and_minor_determinant() -> None:
    payload = _none_payload()
    payload["partial_derivatives"] = [ZERO_POLYNOMIAL] * 3
    payload["degree_maps"][0]["rank_minor"]["determinant"] = {
        "num": "999",
        "den": "1",
    }

    with pytest.raises(ValueError, match="partial derivative"):
        GradedJacobianSyzygyResult.model_validate(payload)


def test_syzygy_result_rejects_an_altered_rank_minor_determinant() -> None:
    payload = _none_payload()
    payload["degree_maps"][0]["rank_minor"]["determinant"] = {
        "num": "999",
        "den": "1",
    }

    with pytest.raises(ValueError, match="rank minor"):
        GradedJacobianSyzygyResult.model_validate(payload)


def test_syzygy_result_rejects_incoherent_cross_result_splicing() -> None:
    payload = _found_payload()
    _swap_in_foreign_source(payload)

    with pytest.raises(ValueError, match="partial derivative"):
        GradedJacobianSyzygyResult.model_validate(payload)


@pytest.mark.parametrize(
    ("label", "mutate"),
    [
        ("source_polynomial", _swap_in_foreign_source),
        ("scaled_partial_derivative", _scale_first_partial_derivative),
        ("zeroed_partial_derivative", _zero_second_partial_derivative),
        ("coefficient_map_digest", _replace_coefficient_map_digest),
        ("rank_bookkeeping_with_coherent_minor", _shrink_rank_bookkeeping_coherently),
        ("kernel_coefficient_vector", _mutate_kernel_coefficient_vector),
        ("kernel_multiplier_encoding", _mutate_kernel_multiplier_encoding),
    ],
)
def test_syzygy_result_rejects_authored_mutations(
    label: str,
    mutate: Callable[[dict[str, Any]], None],
) -> None:
    payload = _found_payload()
    mutate(payload)

    with pytest.raises(ValueError):
        GradedJacobianSyzygyResult.model_validate(payload)


def test_syzygy_ledger_result_rejects_mutated_sparse_entries() -> None:
    payload = _ledger_payload()
    payload["degree_maps"][0]["sparse_entries"][0]["coefficient"]["num"] = "7"

    with pytest.raises(ValueError, match="sparse entries"):
        GradedJacobianSyzygyResult.model_validate(payload)
