"""Tests for RSK permutation operation."""

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.algebraic._models import (
    RSKPermutationRequest,
    RSKResult,
)
from jacobian.math.combinatorics.algebraic._operations import compute_rsk_permutation


class TestRSK:
    def test_request_and_result_publish_the_row_insertion_convention(self) -> None:
        request_schema = RSKPermutationRequest.model_json_schema()
        result_schema = RSKResult.model_json_schema()
        assert (
            request_schema["properties"]["convention"]["const"]
            == "ROW_INSERTION_RSK_V1"
        )
        assert (
            result_schema["properties"]["convention"]["const"] == "ROW_INSERTION_RSK_V1"
        )

    def test_empty(self) -> None:
        result = compute_rsk_permutation(RSKPermutationRequest(permutation=()))
        assert result.permutation == ()
        assert result.shape.parts == ()
        assert result.p_tableau.rows == ()
        assert result.q_tableau.rows == ()
        assert result.lis_length == 0
        assert result.lds_length == 0
        assert result.convention == "ROW_INSERTION_RSK_V1"

    def test_identity(self) -> None:
        result = compute_rsk_permutation(
            RSKPermutationRequest(permutation=(1, 2, 3, 4, 5))
        )
        assert result.shape.parts == (5,)
        assert result.lis_length == 5
        assert result.lds_length == 1
        assert result.p_tableau.rows == ((1, 2, 3, 4, 5),)
        assert result.q_tableau.rows == ((1, 2, 3, 4, 5),)

    def test_reverse(self) -> None:
        result = compute_rsk_permutation(
            RSKPermutationRequest(permutation=(5, 4, 3, 2, 1))
        )
        assert result.shape.parts == (1, 1, 1, 1, 1)
        assert result.lis_length == 1
        assert result.lds_length == 5

    def test_132(self) -> None:
        result = compute_rsk_permutation(RSKPermutationRequest(permutation=(1, 3, 2)))
        assert result.shape.parts == (2, 1)
        assert result.lis_length == 2
        assert result.lds_length == 2

    def test_312(self) -> None:
        result = compute_rsk_permutation(RSKPermutationRequest(permutation=(3, 1, 2)))
        assert result.shape.parts == (2, 1)
        assert result.lis_length == 2
        assert result.lds_length == 2

    def test_invalid_not_permutation(self) -> None:
        request = RSKPermutationRequest(permutation=(1, 2, 2))
        with pytest.raises(OperationDomainValidationError) as error:
            compute_rsk_permutation(request)
        assert (
            error.value.errors()[0]["type"]
            == "algebraic_combinatorics.permutation_invalid"
        )

    @pytest.mark.parametrize("value", [True, "1", 1.0])
    def test_permutation_entries_are_strict_integers(self, value: object) -> None:
        with pytest.raises(ValidationError):
            RSKPermutationRequest.model_validate({"permutation": [value]})

    def test_p_and_q_same_shape(self) -> None:
        for perm in [(1, 2, 3), (3, 2, 1), (1, 3, 2), (2, 1, 3), (2, 3, 1), (3, 1, 2)]:
            result = compute_rsk_permutation(RSKPermutationRequest(permutation=perm))
            assert result.p_tableau.shape == result.q_tableau.shape == result.shape

    def test_single_element(self) -> None:
        result = compute_rsk_permutation(RSKPermutationRequest(permutation=(1,)))
        assert result.shape.parts == (1,)
        assert result.p_tableau.rows == ((1,),)
        assert result.q_tableau.rows == ((1,),)

    def test_result_parsing_retains_only_structural_tableau_checks(self) -> None:
        result = compute_rsk_permutation(RSKPermutationRequest(permutation=(1, 3, 2)))
        payload = result.model_dump(mode="json")
        payload["permutation"] = [2, 1, 3]
        assert RSKResult.model_validate(payload).permutation == (2, 1, 3)

        payload = result.model_dump(mode="json")
        payload["permutation"] = [1, 2]
        with pytest.raises(ValidationError, match="size must equal permutation"):
            RSKResult.model_validate(payload)

        payload = result.model_dump(mode="json")
        payload["shape"] = {"parts": [3]}
        with pytest.raises(ValidationError) as error:
            RSKResult.model_validate(payload)
        assert error.value.errors()[0]["type"] in {
            "algebraic_combinatorics.rsk_shape_mismatch",
            "algebraic_combinatorics.rsk_lengths_mismatch",
        }
