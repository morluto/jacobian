"""Public-bound execution checks for the link-extension operations."""

from __future__ import annotations

from typing import Literal

import pytest

from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
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
        result = tool.run(LinkDeterminantRequest(diagram=diagram))

        assert isinstance(result, LinkDeterminantResult)
        assert result.alexander.diagram == diagram
        assert result.determinant == 3
        assert (
            LinkDeterminantResult.model_validate_json(result.model_dump_json())
            == result
        )


class TestLinkExtensionTools:
    def test_braid_product_public_bounds_are_exact(self) -> None:
        catalog = {tool.operation_id: tool for tool in BUILTIN_TOOLS}
        multiply = catalog["braid.word.multiply.compute"]
        with pytest.raises(OperationDomainValidationError, match="same strand count"):
            multiply.run(
                BraidProductRequest(left=_two_braid(1), right=BraidWord(strand_count=3))
            )
        with pytest.raises(OperationResourceAdmissionError, match="64-letter"):
            multiply.run(
                BraidProductRequest(
                    left=BraidWord(
                        strand_count=2,
                        letters=tuple(
                            BraidLetter(generator=1, exponent=1) for _ in range(32)
                        ),
                    ),
                    right=BraidWord(
                        strand_count=2,
                        letters=tuple(
                            BraidLetter(generator=1, exponent=1) for _ in range(33)
                        ),
                    ),
                )
            )
