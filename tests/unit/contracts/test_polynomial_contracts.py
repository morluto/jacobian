from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.contracts.exact import (
    bounded_rational_grid_size,
    bounded_rational_scalars,
)
from jacobian.contracts.polynomial_systems import PolynomialSystemRationalSearchRequest
from jacobian.contracts.polynomials import (
    PolynomialCollisionOutput,
    PolynomialCollisionPayload,
    PolynomialCollisionRequest,
    PolynomialCollisionSearchRequest,
    PolynomialCollisionVerifyRequest,
    PolynomialEvaluationRequest,
    PolynomialJacobianRequest,
    PolynomialMapEvaluation,
    PolynomialMapInverseCollisionVerifyRequest,
    RationalFunctionArtifact,
    RationalFunctionIdentityRequest,
    RationalPolynomialMap,
    SparseRationalFunction,
    SparseRationalPolynomial,
)


def _rational(value: int = 0) -> dict[str, str]:
    return {"num": str(value), "den": "1"}


def _identity_map(dimension: int = 1) -> dict[str, object]:
    return {
        "map_schema_version": "1",
        "domain": "QQ",
        "variables": [f"x{index}" for index in range(dimension)],
        "coordinates": [
            {
                "terms": [
                    {
                        "coefficient": _rational(1),
                        "exponents": [
                            int(coordinate == exponent) for exponent in range(dimension)
                        ],
                    }
                ]
            }
            for coordinate in range(dimension)
        ],
    }


def _linear_system(dimension: int = 1) -> dict[str, object]:
    return {
        "system_schema_version": "1",
        "domain": "QQ",
        "variables": [f"x{index}" for index in range(dimension)],
        "equations": [
            {
                "terms": [
                    {
                        "coefficient": _rational(1),
                        "exponents": [
                            int(variable == 0) for variable in range(dimension)
                        ],
                    }
                ]
            }
        ],
        "inequations": [],
    }


def test_evaluation_request_enforces_map_point_dimension() -> None:
    with pytest.raises(ValidationError, match="point dimension"):
        PolynomialEvaluationRequest.model_validate(
            {"map": _identity_map(), "point": [_rational(), _rational()]}
        )


def test_evaluation_artifact_accepts_rectangular_map_image() -> None:
    evaluation = PolynomialMapEvaluation.model_validate(
        {
            "map_uri": "artifact://sha256/" + "a" * 64,
            "point": {"values": [_rational()]},
            "image": [_rational(), _rational()],
            "backend": "sympy",
            "backend_version": "1.14.0",
        }
    )

    assert len(evaluation.point.values) == 1
    assert len(evaluation.image) == 2


def test_polynomial_map_point_contracts_accept_five_dimensions() -> None:
    point = [_rational()] * 5
    second_point = [_rational(1), *([_rational()] * 4)]
    image = [_rational()] * 5
    polynomial_map = _identity_map(5)

    request = PolynomialEvaluationRequest.model_validate(
        {"map": polynomial_map, "point": point}
    )
    evaluation = PolynomialMapEvaluation.model_validate(
        {
            "map_uri": "artifact://sha256/" + "a" * 64,
            "point": {"values": point},
            "image": image,
            "backend": "sympy",
            "backend_version": "1.14.0",
        }
    )
    collision = PolynomialCollisionVerifyRequest.model_validate(
        {
            "map": polynomial_map,
            "first_point": point,
            "second_point": second_point,
            "claimed_image": image,
        }
    )
    inverse_collision = PolynomialMapInverseCollisionVerifyRequest.model_validate(
        {
            "map": polynomial_map,
            "first_point": point,
            "second_point": second_point,
            "claimed_image": image,
        }
    )
    payload = PolynomialCollisionPayload.model_validate(
        {
            "first_point": point,
            "second_point": second_point,
            "image": image,
        }
    )

    assert len(request.point) == 5
    assert len(evaluation.image) == 5
    assert len(collision.claimed_image) == 5
    assert len(inverse_collision.claimed_image) == 5
    assert len(payload.image) == 5


def test_collision_payload_enforces_all_dimensions() -> None:
    with pytest.raises(ValidationError, match="points must have matching dimensions"):
        PolynomialCollisionPayload.model_validate(
            {
                "first_point": [_rational()],
                "second_point": [_rational(), _rational()],
                "image": [_rational()],
            }
        )


def test_collision_request_requires_two_distinct_evaluation_artifacts() -> None:
    artifact_uri = "artifact://sha256/" + "a" * 64

    with pytest.raises(ValidationError, match="distinct evaluation artifacts"):
        PolynomialCollisionRequest.model_validate(
            {
                "first_evaluation_uri": artifact_uri,
                "second_evaluation_uri": artifact_uri,
            }
        )


def test_collision_output_enforces_distinct_points_and_equal_images() -> None:
    artifact_uri = "artifact://sha256/" + "a" * 64
    second_artifact_uri = "artifact://sha256/" + "c" * 64
    with pytest.raises(ValidationError, match="collision status"):
        PolynomialCollisionOutput.model_validate(
            {
                "claim_uri": artifact_uri,
                "candidate_uri": artifact_uri,
                "first_evaluation_uri": artifact_uri,
                "second_evaluation_uri": second_artifact_uri,
                "first_point": [_rational(0)],
                "second_point": [_rational(0)],
                "first_image": [_rational(1)],
                "second_image": [_rational(1)],
                "candidate_collision": True,
                "witness_uri": artifact_uri,
            }
        )


def test_jacobian_request_rejects_excessive_symbolic_expansion() -> None:
    dimension = 4
    exponents = [[degree, 1, 1, 1] for degree in range(32, 12, -1)]
    polynomial = {
        "terms": [
            {"coefficient": _rational(1), "exponents": monomial}
            for monomial in exponents
        ]
    }
    polynomial_map = {
        "map_schema_version": "1",
        "domain": "QQ",
        "variables": ["w", "x", "y", "z"],
        "coordinates": [polynomial] * dimension,
    }

    with pytest.raises(ValidationError, match="operation budget"):
        PolynomialJacobianRequest.model_validate({"map": polynomial_map})


def test_shared_map_accepts_operation_expensive_exponents() -> None:
    polynomial_map = _identity_map()
    polynomial_map["coordinates"][0]["terms"][0]["exponents"] = [33]

    polynomial_map_value = RationalPolynomialMap.model_validate(polynomial_map)
    assert polynomial_map_value.coordinates[0].terms[0].exponents == (33,)

    with pytest.raises(ValidationError, match="32-degree operation budget"):
        PolynomialJacobianRequest.model_validate({"map": polynomial_map})


def test_shared_polynomial_map_is_not_limited_to_square_maps() -> None:
    polynomial_map = _identity_map()
    polynomial_map["coordinates"].append({"terms": []})

    value = RationalPolynomialMap.model_validate(polynomial_map)
    assert len(value.coordinates) == 2


def test_canonical_sparse_value_still_rejects_duplicate_exponents() -> None:
    with pytest.raises(ValidationError, match="exponent tuples must be unique"):
        SparseRationalPolynomial.model_validate(
            {
                "terms": [
                    {"coefficient": _rational(1), "exponents": [1]},
                    {"coefficient": _rational(2), "exponents": [1]},
                ]
            }
        )


def test_canonical_sparse_value_still_rejects_zero_terms() -> None:
    with pytest.raises(ValidationError, match="zero polynomial terms must be omitted"):
        SparseRationalPolynomial.model_validate(
            {
                "terms": [
                    {"coefficient": _rational(0), "exponents": [0]},
                ]
            }
        )


def test_shared_polynomial_representation_exceeds_gcd_input_budget() -> None:
    from jacobian.contracts.polynomial_operations import PolynomialGcdRequest
    from jacobian.contracts.polynomials import RationalPolynomial

    coefficient = "9" * 257
    polynomial = RationalPolynomial.model_validate(
        {
            "variables": ["x"],
            "polynomial": {
                "terms": [
                    {
                        "coefficient": {"num": coefficient, "den": "1"},
                        "exponents": [1],
                    }
                ]
            },
        }
    )
    assert polynomial.polynomial.terms[0].coefficient.num == coefficient

    with pytest.raises(ValidationError, match="256-digit"):
        PolynomialGcdRequest(left=polynomial, right=polynomial)


def test_bounded_rational_scalars_deduplicate_equivalents() -> None:
    assert len(bounded_rational_scalars(8, 8)) == 87
    assert bounded_rational_grid_size(8, 8, 2) == 7569


def test_collision_search_accepts_exact_grid_within_limit() -> None:
    # Loose upper bound is 136**2 = 18496; exact deduplicated grid is 87**2 = 7569.
    request = PolynomialCollisionSearchRequest.model_validate(
        {
            "map": _identity_map(2),
            "max_abs_numerator": 8,
            "max_denominator": 8,
        }
    )
    assert request.max_abs_numerator == 8
    assert request.max_denominator == 8
    assert len(request.map.variables) == 2


def test_collision_search_rejects_exact_grid_over_limit() -> None:
    with pytest.raises(ValidationError, match="10,000"):
        PolynomialCollisionSearchRequest.model_validate(
            {
                "map": _identity_map(4),
                "max_abs_numerator": 8,
                "max_denominator": 8,
            }
        )


def test_system_search_accepts_exact_grid_within_limit() -> None:
    request = PolynomialSystemRationalSearchRequest.model_validate(
        {
            "system": _linear_system(2),
            "max_abs_numerator": 8,
            "max_denominator": 8,
        }
    )
    assert request.max_abs_numerator == 8
    assert request.max_denominator == 8
    assert len(request.system.variables) == 2


def test_system_search_rejects_exact_grid_over_limit() -> None:
    with pytest.raises(ValidationError, match="10,000"):
        PolynomialSystemRationalSearchRequest.model_validate(
            {
                "system": _linear_system(4),
                "max_abs_numerator": 8,
                "max_denominator": 8,
            }
        )


def _dense_sparse_polynomial(term_count: int) -> SparseRationalPolynomial:
    """Build a univariate polynomial with term_count terms in descending order."""
    return SparseRationalPolynomial(
        terms=tuple(
            {
                "coefficient": _rational(1),
                "exponents": [term_count - 1 - index],
            }
            for index in range(term_count)
        )
    )


def test_rational_function_artifact_accepts_dense_self_product_fraction() -> None:
    """Thread PRRT_kwDOThEfjc6VuwhR: a single fraction with 65 numerator terms
    and 65 denominator terms has a 4,225-pair self-product, but the artifact
    must not apply the two-function cross-product bound to itself.
    """
    dense = _dense_sparse_polynomial(65)
    artifact = RationalFunctionArtifact(
        variables=("x",),
        numerator=dense,
        denominator=dense,
    )
    assert artifact.numerator is dense
    assert artifact.denominator is dense


def test_rational_function_identity_request_still_enforces_cross_product_bound() -> (
    None
):
    """The identity-request cross-product bound must still reject a pair of
    fractions whose cross product exceeds 4096 term pairs.
    """
    dense = _dense_sparse_polynomial(65)
    with pytest.raises(ValidationError, match="cross product exceeds 4096"):
        RationalFunctionIdentityRequest(
            variables=("x",),
            left=SparseRationalFunction(numerator=dense, denominator=dense),
            right=SparseRationalFunction(numerator=dense, denominator=dense),
        )


def test_rational_function_artifact_rejects_duplicate_variables() -> None:
    with pytest.raises(ValidationError, match="variables must be unique"):
        RationalFunctionArtifact(
            variables=("x", "x"),
            numerator=SparseRationalPolynomial(
                terms=({"coefficient": _rational(1), "exponents": [1, 0]},)
            ),
            denominator=SparseRationalPolynomial(
                terms=({"coefficient": _rational(1), "exponents": [0, 0]},)
            ),
        )


def test_rational_function_artifact_rejects_zero_denominator() -> None:
    with pytest.raises(ValidationError, match="denominator must be nonzero"):
        RationalFunctionArtifact(
            variables=("x",),
            numerator=SparseRationalPolynomial(
                terms=({"coefficient": _rational(1), "exponents": [0]},)
            ),
            denominator=SparseRationalPolynomial(terms=()),
        )


def test_rational_function_artifact_rejects_exponent_dimension_mismatch() -> None:
    with pytest.raises(ValidationError, match="variable order"):
        RationalFunctionArtifact(
            variables=("x", "y"),
            numerator=SparseRationalPolynomial(
                terms=({"coefficient": _rational(1), "exponents": [1, 0]},)
            ),
            denominator=SparseRationalPolynomial(
                terms=({"coefficient": _rational(1), "exponents": [0]},)
            ),
        )
