"""Exact carrier-transport checks with independent finite oracles."""

from collections.abc import Iterator
from itertools import permutations, product

import pytest

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.logic.relational_structures._models import (
    FiniteCspConstraint,
    FiniteCspInstance,
)
from jacobian.math.logic.relational_structures.operations import profile_csp_assignment
from jacobian.math.logic.relational_structures.relabeling._models import (
    CspTemplateCarrierRelabelingRequest,
    RelationalCarrierRelabelingRequest,
)
from jacobian.math.logic.relational_structures.relabeling._tools import TOOLS
from jacobian.math.logic.relational_structures.relabeling.operations import (
    relabel_csp_template_carrier,
    relabel_structure_carrier,
)
from jacobian.math.logic.relational_structures.values import (
    MAX_RELATIONAL_TABLE_ROWS,
    FiniteRelationalStructure,
    FiniteRelationSymbol,
)


def _structure(
    carrier_size: int,
    signature: tuple[FiniteRelationSymbol, ...],
    tables: tuple[tuple[tuple[int, ...], ...], ...],
) -> FiniteRelationalStructure:
    return FiniteRelationalStructure(
        carrier_size=carrier_size, signature=signature, relation_tables=tables
    )


def _oracle_transport(
    structure: FiniteRelationalStructure, old_to_new: tuple[int, ...]
) -> FiniteRelationalStructure:
    """Independent mathematical definition, without the production helper."""

    tables = []
    for table in structure.relation_tables:
        transported = {
            tuple(old_to_new[coordinate] for coordinate in row) for row in table
        }
        tables.append(tuple(sorted(transported)))
    return _structure(structure.carrier_size, structure.signature, tuple(tables))


def _oracle_isomorphism(
    source: FiniteRelationalStructure,
    target: FiniteRelationalStructure,
    old_to_new: tuple[int, ...],
) -> bool:
    """Check relation preservation and reflection directly, symbol by symbol."""

    if (
        source.signature != target.signature
        or source.carrier_size != target.carrier_size
    ):
        return False
    inverse = [0] * len(old_to_new)
    for old, new in enumerate(old_to_new):
        inverse[new] = old
    for source_table, target_table in zip(
        source.relation_tables, target.relation_tables, strict=True
    ):
        mapped_source = {
            tuple(old_to_new[coordinate] for coordinate in row) for row in source_table
        }
        mapped_target_back = {
            tuple(inverse[coordinate] for coordinate in row) for row in target_table
        }
        if mapped_source != set(target_table) or mapped_target_back != set(
            source_table
        ):
            return False
    return True


_Table = tuple[tuple[int, ...], ...]


def _all_subsets(items: _Table) -> tuple[_Table, ...]:
    return tuple(
        tuple(row for index, row in enumerate(items) if selected & (1 << index))
        for selected in range(1 << len(items))
    )


def test_exhaustive_small_structures_match_transport_and_isomorphism_oracle() -> None:
    """Exhaust all nullary/unary/binary tables through a two-element carrier."""

    signature = (
        FiniteRelationSymbol(symbol_id="N", arity=0),
        FiniteRelationSymbol(symbol_id="U", arity=1),
        FiniteRelationSymbol(symbol_id="E", arity=2),
    )
    for carrier_size in range(3):
        carriers = range(carrier_size)
        nullary_tables: tuple[_Table, ...] = (((),), ())
        unary_rows: _Table = tuple((element,) for element in carriers)
        binary_rows: _Table = tuple(product(carriers, repeat=2))
        choices: tuple[tuple[_Table, ...], ...] = (
            nullary_tables,
            _all_subsets(unary_rows),
            _all_subsets(binary_rows),
        )
        for tables in product(*choices):
            source = _structure(carrier_size, signature, tuple(tables))
            for mapping in permutations(range(carrier_size)):
                result = relabel_structure_carrier(source, mapping)
                expected = _oracle_transport(source, mapping)
                assert result.target == expected
                assert result.old_to_new == mapping
                assert tuple(result.new_to_old[new] for new in mapping) == tuple(
                    range(carrier_size)
                )

                # Isomorphism is checked independently by enumerating all maps.
                isomorphic = any(
                    _oracle_isomorphism(source, result.target, candidate)
                    for candidate in permutations(range(carrier_size))
                )
                assert isomorphic


def test_csp_template_relabeling_preserves_occurrences_and_solutions() -> None:
    template = _structure(
        3,
        (FiniteRelationSymbol(symbol_id="R", arity=2),),
        (((0, 1), (1, 2), (2, 0)),),
    )
    instance = FiniteCspInstance(
        template=template,
        variable_count=3,
        constraints=(
            FiniteCspConstraint(constraint_id="edge_a", symbol_id="R", scope=(0, 1)),
            FiniteCspConstraint(constraint_id="loop", symbol_id="R", scope=(1, 1)),
            FiniteCspConstraint(constraint_id="edge_b", symbol_id="R", scope=(0, 1)),
        ),
    )
    mapping = (2, 0, 1)
    result = relabel_csp_template_carrier(instance, mapping)
    assert result.target.constraints == instance.constraints
    assert result.target.variable_count == instance.variable_count
    assert result.target.template == _oracle_transport(template, mapping)

    for assignment in product(
        range(template.carrier_size), repeat=instance.variable_count
    ):
        old = profile_csp_assignment(instance, assignment)
        transported = tuple(mapping[value] for value in assignment)
        new = profile_csp_assignment(result.target, transported)
        assert old.status == new.status
        assert tuple(item.constraint_id for item in new.evaluations) == (
            "edge_a",
            "loop",
            "edge_b",
        )


@pytest.mark.parametrize(
    ("mapping", "error_type"),
    [
        ((), "relational.relabeling.map_axis"),
        ((0, 0), "relational.relabeling.map_bijection"),
        ((0, 2), "relational.relabeling.map_range"),
        ((0, True), "relational.relabeling.map_label"),
    ],
)
def test_rejects_non_bijections(mapping: tuple[object, ...], error_type: str) -> None:
    source = _structure(
        2,
        (FiniteRelationSymbol(symbol_id="E", arity=2),),
        (((0, 1),),),
    )
    with pytest.raises(OperationDomainValidationError) as exc_info:
        relabel_structure_carrier(source, mapping)  # type: ignore[arg-type]
    assert exc_info.value.errors()[0]["type"] == error_type


def test_preflights_aggregate_tuple_transport_before_expansion() -> None:
    signature = tuple(
        FiniteRelationSymbol(symbol_id=f"R{index}", arity=2) for index in range(5)
    )
    rows = tuple(product(range(64), repeat=2))[:4096]
    source = _structure(64, signature, (rows,) * len(signature))
    with pytest.raises(OperationResourceAdmissionError) as exc_info:
        relabel_structure_carrier(source, tuple(reversed(range(64))))
    assert exc_info.value.errors()[0]["type"] == "relational.relabeling.tuple_limit"


def test_bypassed_structure_is_admitted_before_table_expansion() -> None:
    """A ``model_construct`` payload cannot force expansion before admission."""

    class _UnexpandableTable(list[tuple[int, ...]]):
        def __len__(self) -> int:
            return MAX_RELATIONAL_TABLE_ROWS + 1

        def __iter__(self) -> Iterator[tuple[int, ...]]:
            raise AssertionError("relation table expanded before admission")

    source = FiniteRelationalStructure.model_construct(
        carrier_size=2,
        signature=(FiniteRelationSymbol(symbol_id="E", arity=2),),
        relation_tables=(_UnexpandableTable(),),
    )
    with pytest.raises(OperationResourceAdmissionError) as exc_info:
        relabel_structure_carrier(source, (1, 0))
    error = exc_info.value.errors()[0]
    assert error["type"] == "relational.relabeling.table_limit"
    assert error["loc"] == ("source", "relation_tables")


def test_csp_tuple_admission_points_at_the_instance_template() -> None:
    """CSP template admission reports the nested template path, not ``source``."""

    signature = tuple(
        FiniteRelationSymbol(symbol_id=f"R{index}", arity=2) for index in range(5)
    )
    rows = tuple(product(range(64), repeat=2))[:4096]
    template = _structure(64, signature, (rows,) * len(signature))
    instance = FiniteCspInstance(
        template=template,
        variable_count=64,
        constraints=(
            FiniteCspConstraint(constraint_id="edge", symbol_id="R0", scope=(0, 1)),
        ),
    )
    with pytest.raises(OperationResourceAdmissionError) as exc_info:
        relabel_csp_template_carrier(instance, tuple(reversed(range(64))))
    error = exc_info.value.errors()[0]
    assert error["type"] == "relational.relabeling.tuple_limit"
    assert error["loc"] == ("instance", "template", "relation_tables")


def test_request_schemas_publish_the_permutation_contract() -> None:
    for request_type in (
        RelationalCarrierRelabelingRequest,
        CspTemplateCarrierRelabelingRequest,
    ):
        description = request_type.model_json_schema()["properties"]["old_to_new"][
            "description"
        ]
        assert "new label of old" in description
        assert "permutation" in description


def test_examples_state_the_permutation_contract() -> None:
    for tool in TOOLS:
        for example in tool.examples:
            assert "old_to_new[i]" in example.description
            assert "permutation" in example.description
