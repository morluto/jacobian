"""Public-bound execution checks for the link-extension operations."""

from __future__ import annotations

import json
from typing import Literal

import pytest

from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.dispatch import invoke_operation
from jacobian.math.topology.links import (
    BraidLetter,
    BraidWord,
    braid_closure,
)
from jacobian.math.topology.links._extensions_models import (
    BraidProductRequest,
    LinkDeterminantRequest,
    LinkDeterminantResult,
)


def _two_braid(*exponents: Literal[-1, 1]) -> BraidWord:
    return BraidWord(
        strand_count=2,
        letters=tuple(
            BraidLetter(generator=1, exponent=exponent) for exponent in exponents
        ),
    )


class TestAlexanderPolynomial:
    def test_determinant_is_a_public_source_bound_operation(self) -> None:
        tool = next(
            tool
            for tool in BUILTIN_TOOLS
            if tool.operation_id == "link_diagram.determinant.compute"
        )
        diagram = braid_closure(_two_braid(1, 1, 1)).diagram
        request = LinkDeterminantRequest(diagram=diagram)
        result = tool.run(request)

        assert isinstance(result, LinkDeterminantResult)
        assert result.alexander.diagram == diagram
        assert result.determinant == 3
        assert (
            LinkDeterminantResult.model_validate_json(result.model_dump_json())
            == result
        )
        # Dispatch owns the public boundary: operation-ID lookup, strict wire
        # parsing, and the serialized output envelope must all accept it.
        dispatched = invoke_operation(
            tool.operation_id, request.model_dump(mode="json"), Catalog.open()
        )
        assert (
            LinkDeterminantResult.model_validate_json(json.dumps(dispatched.output))
            == result
        )


class TestLinkExtensionTools:
    def test_braid_product_public_bounds_are_exact(self) -> None:
        mismatched = BraidProductRequest(
            left=_two_braid(1), right=BraidWord(strand_count=3)
        )
        with pytest.raises(OperationDomainValidationError) as mismatch:
            invoke_operation(
                "braid.word.multiply.compute",
                mismatched.model_dump(mode="json"),
                Catalog.open(),
            )
        assert (
            mismatch.value.errors()[0]["type"] == "link_diagram.braid_parent_mismatch"
        )

        overbound = BraidProductRequest(
            left=BraidWord(
                strand_count=2,
                letters=tuple(BraidLetter(generator=1, exponent=1) for _ in range(32)),
            ),
            right=BraidWord(
                strand_count=2,
                letters=tuple(BraidLetter(generator=1, exponent=1) for _ in range(33)),
            ),
        )
        with pytest.raises(OperationResourceAdmissionError) as growth:
            invoke_operation(
                "braid.word.multiply.compute",
                overbound.model_dump(mode="json"),
                Catalog.open(),
            )
        assert (
            growth.value.errors()[0]["type"]
            == "link_diagram.braid_product_length_bound"
        )
