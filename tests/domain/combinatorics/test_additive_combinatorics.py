from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from tests.support.services import DomainTestServices, open_domain_services

from jacobian.contracts.combinatorics import (
    MAX_ADDITIVE_DIFFERENCE_INTEGER_LENGTH,
    MAX_ADDITIVE_INTEGER_LENGTH,
)
from jacobian.contracts.operations import OperationRequest
from jacobian.contracts.results import ExecutionStatus
from jacobian.domains.combinatorics import combinatorics_operations

_BASE = ["1", "2", "4", "8", "13"]


@pytest.fixture
def domain_services(tmp_path: Path) -> Iterator[DomainTestServices]:
    with open_domain_services(
        tmp_path / "state", combinatorics_operations()
    ) as services:
        yield services


def test_integer_sidon_materializes_every_ordered_difference(
    domain_services: DomainTestServices,
) -> None:
    computed = domain_services.core.operations.invoke(
        OperationRequest(
            operation_id="combinatorics.integer_set.sidon.decide",
            input={"elements": _BASE},
        )
    )

    result = computed.output["result"]
    assert computed.execution.status is ExecutionStatus.COMPLETED
    assert result["is_sidon"] is True
    assert len(result["ordered_differences"]) == 20


def test_perfect_difference_set_reports_complete_residue_profile(
    domain_services: DomainTestServices,
) -> None:
    computed = domain_services.core.operations.invoke(
        OperationRequest(
            operation_id="combinatorics.cyclic_difference_set.perfect.decide",
            input={"modulus": 7, "residues": [0, 1, 3]},
        )
    )

    result = computed.output["result"]
    assert result["is_perfect"] is True
    assert result["missing_residues"] == []
    assert result["repeated_residues"] == []
    assert len(result["difference_multiplicities"]) == 6


@pytest.mark.parametrize(
    ("order", "candidate_count"),
    ((5, 1), (6, 26), (7, 703)),
)
def test_fixed_order_extension_returns_complete_negative_decisions(
    domain_services: DomainTestServices,
    order: int,
    candidate_count: int,
) -> None:
    computed = domain_services.core.operations.invoke(
        OperationRequest(
            operation_id="combinatorics.cyclic_difference_set.extension.decide",
            input={"base_elements": _BASE, "target_order": order},
        )
    )

    assert computed.execution.status is ExecutionStatus.COMPLETED
    assert computed.artifact_uris == ()
    assert computed.output["result"]["decision"] == "DOES_NOT_EXTEND"
    assert computed.output["result"]["coverage"] == "ALL_CANDIDATES"
    assert computed.output["result"]["candidate_space_size"] == candidate_count


def test_fixed_order_extension_returns_a_complete_positive_witness(
    domain_services: DomainTestServices,
) -> None:
    computed = domain_services.core.operations.invoke(
        OperationRequest(
            operation_id="combinatorics.cyclic_difference_set.extension.decide",
            input={"base_elements": ["0", "1"], "target_order": 3},
        )
    )

    assert computed.artifact_uris == ()
    assert computed.output["result"]["decision"] == "EXTENDS"
    assert computed.output["result"]["coverage"] == "WITNESS"
    assert computed.output["result"]["extension"] == [0, 1, 3]


def test_integer_sidon_accepts_the_widest_canonical_difference(
    domain_services: DomainTestServices,
) -> None:
    """The result bound must hold every ordered difference of accepted inputs.

    The largest accepted positive value uses every ``AdditiveInteger``
    character for digits, while the most-negative accepted value spends one
    character on the sign. Their ordered difference reaches one extra digit,
    and the negative direction adds the sign back, so the canonical difference
    string is exactly ``MAX_ADDITIVE_INTEGER_LENGTH + 2`` characters. A bound
    of ``+1`` (the previous value) rejects this valid public request.
    """
    largest_positive = "9" * MAX_ADDITIVE_INTEGER_LENGTH
    most_negative = "-" + "9" * (MAX_ADDITIVE_INTEGER_LENGTH - 1)
    assert len(largest_positive) == MAX_ADDITIVE_INTEGER_LENGTH
    assert len(most_negative) == MAX_ADDITIVE_INTEGER_LENGTH

    computed = domain_services.core.operations.invoke(
        OperationRequest(
            operation_id="combinatorics.integer_set.sidon.decide",
            input={"elements": [largest_positive, most_negative]},
        )
    )

    result = computed.output["result"]
    assert computed.execution.status is ExecutionStatus.COMPLETED
    assert result["is_sidon"] is True
    difference_strings = {
        record["difference"] for record in result["ordered_differences"]
    }
    # The negative direction is the tight case: sign + (L+1) digits = L+2 chars.
    widest_negative = "-" + "1" + "0" + "9" * (MAX_ADDITIVE_INTEGER_LENGTH - 2) + "8"
    assert len(widest_negative) == MAX_ADDITIVE_DIFFERENCE_INTEGER_LENGTH
    assert widest_negative in difference_strings
    assert max(len(value) for value in difference_strings) == (
        MAX_ADDITIVE_DIFFERENCE_INTEGER_LENGTH
    )
