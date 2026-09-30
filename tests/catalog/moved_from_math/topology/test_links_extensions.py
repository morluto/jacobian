"""Catalog-publication checks moved out of the math lane.

This module boots the built-in catalog, which discovers and compiles every
owner-local ``._tools`` manifest. It lives under ``tests/catalog`` because
``tests/math`` must drive kernels directly and must not start the catalog;
see ``docs/reference/testing-strategy.md`` (test lane ownership).

Moved from ``tests/math/topology/test_links_extensions.py``. The preamble is carried over verbatim so the moved tests
resolve every name they use.
"""

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
    braid_permutation,
    link_blackboard_graph,
)
from jacobian.math.topology.links._extensions_models import (
    AlexanderPolynomialRequest,
    BraidProductRequest,
    BraidWordRequest,
    GoeritzDataRequest,
    LinkDeterminantRequest,
    LinkDeterminantResult,
    SeifertCircleRequest,
    WirtingerPresentationRequest,
)
from jacobian.math.topology.links._models import OrientedLinkDiagram


def _two_braid(*exponents: Literal[-1, 1]) -> BraidWord:
    return BraidWord(
        strand_count=2,
        letters=tuple(
            BraidLetter(generator=1, exponent=exponent) for exponent in exponents
        ),
    )


class TestBraidWords:
    def test_catalog_exposes_exact_group_word_operations(self) -> None:
        catalog = {tool.operation_id: tool for tool in BUILTIN_TOOLS}
        word = _two_braid(1, -1, 1)
        inverse = catalog["braid.word.inverse.compute"].run(BraidWordRequest(word=word))
        product = catalog["braid.word.multiply.compute"].run(
            BraidProductRequest(left=word, right=inverse)
        )

        assert product == BraidWord(
            strand_count=2, letters=word.letters + inverse.letters
        )
        assert braid_permutation(product).permutation == (0, 1)


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
    def test_catalog_exposes_bounded_link_extension_operations(self) -> None:
        catalog = {tool.operation_id: tool for tool in BUILTIN_TOOLS}
        assert (
            catalog["link_diagram.goeritz_matrix.compute"]
            .run(
                GoeritzDataRequest(
                    blackboard_graph=link_blackboard_graph(
                        braid_closure(_two_braid(1, 1)).diagram
                    )
                )
            )
            .absolute_determinant
            == 2
        )
        assert (
            catalog["link_diagram.seifert_circles.compute"]
            .run(SeifertCircleRequest(diagram=OrientedLinkDiagram(free_loops=1)))
            .genus
            == 0
        )
        assert (
            catalog["link_diagram.alexander_polynomial.compute"]
            .run(AlexanderPolynomialRequest(diagram=OrientedLinkDiagram(free_loops=1)))
            .polynomial.terms[0]
            .coefficient.num
            == 1
        )
        assert (
            catalog["braid.word.permutation.compute"]
            .run(BraidWordRequest(word=_two_braid(1, 1, 1)))
            .closure_component_count
            == 1
        )
        assert (
            len(
                catalog["braid.word.closure.compute"]
                .run(BraidWordRequest(word=_two_braid(1, 1)))
                .diagram.crossings
            )
            == 2
        )
        inverse = catalog["braid.word.inverse.compute"].run(
            BraidWordRequest(word=_two_braid(1, -1))
        )
        assert tuple(letter.exponent for letter in inverse.letters) == (1, -1)
        product = catalog["braid.word.multiply.compute"].run(
            BraidProductRequest(left=_two_braid(1, -1), right=_two_braid(-1, 1))
        )
        assert tuple(letter.exponent for letter in product.letters) == (1, -1, -1, 1)
        assert product.strand_count == 2
        assert catalog["link_diagram.wirtinger_presentation.compute"].run(
            WirtingerPresentationRequest(diagram=OrientedLinkDiagram(free_loops=1))
        ).presentation.generators == ("meridian_000",)

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
