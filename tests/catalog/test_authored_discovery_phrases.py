"""Exact authored wording is strong retrieval evidence, not broad token matching."""

import json
from dataclasses import asdict

import pytest
from tools.discovery_phrase_recall import authored_phrase_recall

from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationDescriptor, OperationMatchRequest
from jacobian.catalog.search import OperationSearchIndex


def _descriptor(
    operation_id: str,
    *,
    title: str = "Inspect objects",
    phrase: str = "pp-defined relation",
    tags: tuple[str, ...] = (),
) -> OperationDescriptor:
    return OperationDescriptor(
        operation_id=operation_id,
        title=title,
        description="Inspect a finite mathematical object.",
        input_schema={"type": "object"},
        output_schema={"type": "object"},
        discovery_terms=(phrase,),
        tags=tags,
    )


@pytest.mark.parametrize(
    ("phrase", "owner"),
    [
        ("pp-defined relation", "pp_formula.evaluate_relation.compute"),
        ("H to V polyhedron conversion", "polytope.rational.h_to_v.compute"),
        (
            "characteristic p syzygies",
            "finite_field.jacobian_syzygy.generators.compute",
        ),
        ("solve Ax=b", "linear.rational_solution.compute"),
    ],
)
def test_exact_authored_phrase_retrieves_its_owner_first(
    phrase: str, owner: str
) -> None:
    catalog = Catalog.open()
    descriptor = catalog.operation(owner)
    assert descriptor is not None and phrase in descriptor.discovery_terms
    for query in (phrase, "  " + "   ".join(phrase.upper().split()) + "  "):
        matches = catalog.match(OperationMatchRequest(need=query, limit=10)).matches
        assert matches[0].operation_id == owner


def test_complete_phrase_eligibility_preserves_ranking_and_pagination() -> None:
    operations = (
        _descriptor("z.alias.inspect"),
        _descriptor("a.alias.inspect"),
        _descriptor(
            "b.neighbor.inspect", title="pp-defined relation", phrase="another topic"
        ),
    )
    index = OperationSearchIndex(tuple(reversed(operations)))
    cursor = None
    found: list[str] = []
    while True:
        result = index.match(
            OperationMatchRequest(need="pp-defined relation", limit=1, cursor=cursor)
        )
        found.extend(match.operation_id for match in result.matches)
        cursor = result.next_cursor
        if cursor is None:
            break
    assert found == ["b.neighbor.inspect", "a.alias.inspect", "z.alias.inspect"]
    broad = index.match(
        OperationMatchRequest(need="pp-defined relation", search_mode="broad")
    )
    assert found == [match.operation_id for match in broad.matches]
    assert len(found) == len(set(found))
    report = authored_phrase_recall(operations, page_limit=2)
    shared = [row for row in report.rows if row.phrase == "pp-defined relation"]
    assert [(row.owner, row.first_page_rank) for row in shared] == [
        ("a.alias.inspect", 2),
        ("z.alias.inspect", None),
    ]
    assert all(row.owners_for_phrase == 2 for row in shared)
    assert json.loads(json.dumps(asdict(report)))["owner_phrase_count"] == 3


@pytest.mark.parametrize(
    "query",
    [
        "normalize pp-defined relation",
        "enumerate pp-defined relation",
        "pp-defined unrelated relation",
        "relation pp-defined",
        "pp defined relation",
    ],
)
def test_partial_reordered_or_changed_postcondition_does_not_get_phrase_override(
    query: str,
) -> None:
    result = OperationSearchIndex((_descriptor("toy.inspect"),)).match(
        OperationMatchRequest(need=query)
    )
    assert not result.matches


def test_exact_phrase_does_not_override_namespace_or_explicit_domain() -> None:
    phrase = "check polynomial identity over integers"
    index = OperationSearchIndex(
        (
            _descriptor(
                "modular.polynomial.check",
                title="Check polynomial identity",
                phrase=phrase,
                tags=("modular",),
            ),
        )
    )
    assert not index.match(OperationMatchRequest(need=phrase)).matches
    assert (
        not Catalog.open()
        .match(OperationMatchRequest(need="solve Ax=b", namespace="smt"))
        .matches
    )


@pytest.mark.parametrize(
    "query",
    [
        "check polynomial identity over integers",
        "classify every finite simple group by proof",
        "normalize pp-defined relation",
    ],
)
def test_unsupported_or_neighboring_tasks_keep_existing_precise_behavior(
    query: str,
) -> None:
    matches = Catalog.open().match(OperationMatchRequest(need=query, limit=10)).matches
    assert "pp_formula.evaluate_relation.compute" not in {
        match.operation_id for match in matches
    }
    if query == "check polynomial identity over integers":
        assert not matches


@pytest.mark.parametrize(
    ("query", "owner", "excluded"),
    [
        (
            "rational vector field Lie bracket",
            "differential_geometry.rational_tensor.lie_derivative.compute",
            "Lie bracket",
        ),
        (
            "free associative algebra multiplication",
            "free_algebra.polynomial.multiply.compute",
            "free associative algebra",
        ),
        (
            "source-indexed edge-deletion family",
            "graph.deck.edge_deleted.compute",
            "graph deck",
        ),
        (
            "source-indexed vertex-deletion family",
            "graph.deck.vertex_deleted.compute",
            "vertex deck",
        ),
        (
            "orientable rotation-system embedding check",
            "graph.embedding.orientable.check",
            "combinatorial map",
        ),
        (
            "characteristic p differentiation",
            "finite_field.polynomial.jacobian.compute",
            None,
        ),
        (
            "characteristic p Jacobian syzygy verification",
            "finite_field.jacobian_syzygy.check",
            "characteristic p differentiation",
        ),
        (
            "sparse nc polynomial multiplication",
            "free_algebra.polynomial.multiply.compute",
            "sparse nc polynomial",
        ),
        ("sparse nc polynomial addition", "free_algebra.polynomial.add.compute", None),
        (
            "sparse nc polynomial subtraction",
            "free_algebra.polynomial.subtract.compute",
            None,
        ),
        ("GL n p", "finite_matrix_group.general_linear.construct", "GL n q"),
        ("SL n p", "finite_matrix_group.special_linear.construct", "SL n q"),
        (
            "GL n q extension field",
            "finite_matrix_group.extension.general_linear.construct",
            None,
        ),
        (
            "SL n q extension field",
            "finite_matrix_group.extension.special_linear.construct",
            None,
        ),
        (
            "GF(p)(x)[y] product reduction",
            "function_field.element.multiply.compute",
            "GF(p)(x)[y] reduction",
        ),
    ],
)
def test_curated_phrases_preserve_actual_scope(
    query: str, owner: str, excluded: str | None
) -> None:
    catalog = Catalog.open()
    operation = catalog.operation(owner)
    assert operation is not None and query in operation.discovery_terms
    assert excluded not in operation.discovery_terms
    matches = catalog.match(OperationMatchRequest(need=query)).matches
    assert owner in {match.operation_id for match in matches}


@pytest.mark.timeout(120)
def test_authored_multiword_catalog_conformance_reports_owner_recall() -> None:
    # This deliberate whole-catalog metadata sweep is larger than a normal query.
    report = authored_phrase_recall(BUILTIN_TOOLS, page_limit=5)
    missing = [
        asdict(row)
        for row in report.rows
        if row.first_page_rank is None and row.owners_for_phrase <= report.page_limit
    ]
    assert not missing, json.dumps(missing, indent=2)
    assert report.owner_phrase_count > 0


def test_recall_report_counts_normalized_ownership_collisions() -> None:
    operations = (
        _descriptor("a.alias.inspect", phrase="A B"),
        _descriptor("z.alias.inspect", phrase="a  b"),
    )
    report = authored_phrase_recall(operations, page_limit=1)
    assert report.distinct_phrases == 1
    assert report.owner_phrase_count == 2
    assert report.first_page_hits == 1
    assert report.owner_recall == 0.5
    assert all(
        row.phrase == "a b" and row.owners_for_phrase == 2 for row in report.rows
    )
    assert all(row.authored_spellings == ("A B", "a  b") for row in report.rows)


def test_complete_symbolic_alias_works_without_turning_other_symbols_into_matches() -> (
    None
):
    index = OperationSearchIndex((_descriptor("toy.symbol.inspect", phrase="Ω"),))
    for mode in ("precise", "broad"):
        result = index.match(OperationMatchRequest(need="Ω", search_mode=mode))
        assert [match.operation_id for match in result.matches] == [
            "toy.symbol.inspect"
        ]
        assert not index.match(
            OperationMatchRequest(need="Ψ", search_mode=mode)
        ).matches


@pytest.mark.parametrize("limit", [0, 21, True])
def test_recall_probe_rejects_invalid_page_limits(limit: int) -> None:
    with pytest.raises(ValueError, match="page_limit"):
        authored_phrase_recall((), page_limit=limit)


def test_empty_recall_probe_does_not_claim_success() -> None:
    report = authored_phrase_recall(())
    assert report.owner_phrase_count == report.first_page_hits == 0
    assert report.owner_recall is None
    assert report.rows == ()
