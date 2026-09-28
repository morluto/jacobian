from jacobian.math.topology.edge_paths._models import (
    AbelianizationRequest,
    FiniteGroupPresentation,
    FiniteGroupWord,
    WordLetter,
)
from jacobian.math.topology.edge_paths._tools import TOOLS


def _presentation(
    relator_exponents: tuple[tuple[int, ...], ...],
) -> FiniteGroupPresentation:
    generators = ("a", "b")
    return FiniteGroupPresentation(
        generators=generators,
        relators=tuple(
            FiniteGroupWord(
                letters=tuple(
                    WordLetter(generator=index, exponent=1)
                    for index, multiplicity in enumerate(exponents)
                    for _ in range(multiplicity)
                )
            )
            for exponents in relator_exponents
        ),
    )


def test_arbitrary_presentation_abelianization_and_catalog_composition() -> None:
    presentation = _presentation(((2, 0), (0, 6)))
    tool = next(
        tool
        for tool in TOOLS
        if tool.operation_id == "topology.group_presentation.abelianization.compute"
    )
    result = tool.run(AbelianizationRequest(presentation=presentation))
    assert result.presentation == presentation
    assert result.abelianization.relation_matrix.entries == ((2, 0), (0, 6))
    assert result.abelianization.rank == 2
    assert result.abelianization.free_rank == 0
    assert result.abelianization.torsion_invariant_factors == (2, 6)
    assert type(result).model_validate_json(result.model_dump_json()) == result


def test_empty_presentation_has_free_abelianization() -> None:
    presentation = FiniteGroupPresentation(generators=("a", "b"), relators=())
    tool = next(
        tool
        for tool in TOOLS
        if tool.operation_id == "topology.group_presentation.abelianization.compute"
    )
    result = tool.run(AbelianizationRequest(presentation=presentation))
    assert result.abelianization.relation_matrix.entries == ()
    assert result.abelianization.relation_matrix.row_count == 0
    assert result.abelianization.relation_matrix.column_count == 2
    assert result.abelianization.free_rank == 2
    assert result.abelianization.torsion_invariant_factors == ()
