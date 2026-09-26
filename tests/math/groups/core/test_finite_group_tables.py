"""Contracts for bounded indexed finite-group tables."""

import json

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.groups._table_models import (
    FiniteGroupTable,
    FiniteGroupTableElement,
    FiniteGroupTableRequest,
)
from jacobian.math.groups._tools import construct_finite_group_table

S3 = (
    (0, 1, 2, 3, 4, 5),
    (1, 0, 4, 5, 2, 3),
    (2, 5, 0, 4, 3, 1),
    (3, 4, 5, 0, 1, 2),
    (4, 3, 1, 2, 5, 0),
    (5, 2, 3, 1, 0, 4),
)


def test_construct_nonabelian_s3_and_retain_parent_bound_elements() -> None:
    result = construct_finite_group_table(
        FiniteGroupTableRequest(identity=0, multiplication=S3)
    )

    assert result.group.multiplication == S3
    assert result.group.identity == 0
    assert result.group.inverse == (0, 1, 2, 3, 5, 4)
    assert result.identity_element == FiniteGroupTableElement(
        group=result.group, index=0
    )
    assert S3[1][2] == 4
    assert S3[2][1] == 5
    assert result.model_validate_json(result.model_dump_json()) == result


def test_table_rejects_non_square_shape_and_wrong_identity() -> None:
    with pytest.raises(OperationDomainValidationError, match="square"):
        construct_finite_group_table(
            FiniteGroupTableRequest(identity=0, multiplication=((0, 1),))
        )
    with pytest.raises(OperationDomainValidationError, match="two-sided"):
        construct_finite_group_table(
            FiniteGroupTableRequest(identity=1, multiplication=S3)
        )


def test_table_rejects_nonassociative_loop() -> None:
    nonassociative = [list(row) for row in S3]
    nonassociative[1][2] = 0
    with pytest.raises(OperationDomainValidationError, match="associative"):
        construct_finite_group_table(
            FiniteGroupTableRequest(
                identity=0,
                multiplication=tuple(tuple(row) for row in nonassociative),
            )
        )


def test_element_index_is_checked_against_its_group_parent() -> None:
    group = FiniteGroupTable(
        identity=0,
        multiplication=S3,
        inverse=(0, 1, 2, 3, 5, 4),
    )
    with pytest.raises(ValidationError, match="belong"):
        FiniteGroupTableElement(group=group, index=6)


def test_table_rejects_forged_inverse_map_from_construct_and_json() -> None:
    forged = FiniteGroupTable.model_construct(
        identity=0,
        multiplication=S3,
        inverse=(0, 0, 2, 3, 5, 4),
    )

    with pytest.raises(ValidationError, match="two-sided inverse"):
        FiniteGroupTable.model_validate_json(forged.model_dump_json())
    with pytest.raises(ValidationError, match="two-sided inverse"):
        FiniteGroupTable.model_validate(
            {
                "identity": 0,
                "multiplication": S3,
                "inverse": (0, 0, 2, 3, 5, 4),
            }
        )
    with pytest.raises(ValidationError, match="two-sided inverse"):
        FiniteGroupTable.model_validate_json(
            json.dumps(
                {
                    "identity": 0,
                    "multiplication": S3,
                    "inverse": (0, 0, 2, 3, 5, 4),
                }
            )
        )


def test_construct_readmits_bypass_constructed_request_before_group_laws() -> None:
    # ``model_construct`` skips the request validators, so a boolean leaf can
    # compare equal to 0 and slip past the numeric group-law checks. The
    # producer must re-admit the request before computing, otherwise it
    # publishes a table whose boolean leaves the holonomy consumer rejects.
    forged = FiniteGroupTableRequest.model_construct(
        multiplication=((False,),),
        identity=0,
    )
    with pytest.raises(OperationDomainValidationError, match="request"):
        construct_finite_group_table(forged)

    missing = FiniteGroupTableRequest.model_construct(identity=0)
    with pytest.raises(OperationDomainValidationError, match="request"):
        construct_finite_group_table(missing)
