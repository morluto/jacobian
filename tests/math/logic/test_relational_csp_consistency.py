"""Independent exhaustive checks for finite CSP domain consistency."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.logic.relational_structures._models import (
    FiniteCspConstraint,
    FiniteCspInstance,
)
from jacobian.math.logic.relational_structures.consistency import (
    CspDomainConsistency,
    CspDomainRequest,
    generalized_arc_consistency,
)
from jacobian.math.logic.relational_structures.consistency._tools import TOOLS
from jacobian.math.logic.relational_structures.values import (
    FiniteRelationalStructure,
    FiniteRelationSymbol,
)


def _instance(
    carrier: int,
    relations: tuple[tuple[int, ...], ...],
    relation_rows: tuple[tuple[tuple[int, ...], ...], ...],
    variable_count: int,
    scopes: tuple[tuple[int, tuple[int, ...]], ...],
) -> FiniteCspInstance:
    template = FiniteRelationalStructure(
        carrier_size=carrier,
        signature=tuple(
            FiniteRelationSymbol(symbol_id=f"R{i}", arity=arity)
            for i, arity in enumerate(relations)
        ),
        relation_tables=relation_rows,
    )
    return FiniteCspInstance(
        template=template,
        variable_count=variable_count,
        constraints=tuple(
            FiniteCspConstraint(
                constraint_id=f"c{i}", symbol_id=f"R{symbol}", scope=scope
            )
            for i, (symbol, scope) in enumerate(scopes)
        ),
    )


def _scan_oracle(
    instance: FiniteCspInstance, supplied: tuple[tuple[int, ...], ...]
) -> tuple[tuple[int, ...], ...]:
    """Slow repeated full-table scan, intentionally unlike the worklist kernel."""
    domains = [set(domain) for domain in supplied]
    symbols = {s.symbol_id: i for i, s in enumerate(instance.template.signature)}
    changed = True
    while changed:
        changed = False
        for constraint in instance.constraints:
            rows = instance.template.relation_tables[symbols[constraint.symbol_id]]
            for variable in set(constraint.scope):
                supported: set[int] = set()
                for row in rows:
                    if not all(
                        value in domains[var]
                        for var, value in zip(constraint.scope, row, strict=True)
                    ):
                        continue
                    if any(
                        row[i] != row[j]
                        for i, var in enumerate(constraint.scope)
                        for j in range(i)
                        if constraint.scope[j] == var
                    ):
                        continue
                    positions = [
                        i for i, scoped_variable in enumerate(constraint.scope)
                        if scoped_variable == variable
                    ]
                    if positions and all(row[i] == row[positions[0]] for i in positions):
                        supported.add(row[positions[0]])
                retained = domains[variable] & supported
                if retained != domains[variable]:
                    domains[variable] = retained
                    changed = True
    return tuple(tuple(sorted(domain)) for domain in domains)


def test_worklist_matches_independent_scan_on_cascading_and_degenerate_cases() -> None:
    # x!=y and y=z force z and then x after the initial restriction.
    instance = _instance(
        3,
        (2,),
        (((0, 1), (1, 2), (2, 0)),),
        3,
        ((0, (0, 1)), (0, (1, 2))),
    )
    cases = (
        ((0, 1, 2), (1,), (0, 1, 2)),
        ((2,), (0, 1, 2), (0, 1, 2)),
        ((), (0, 1, 2), (0, 1, 2)),
    )
    for domains in cases:
        request = CspDomainRequest(instance=instance, domains=domains)
        result = generalized_arc_consistency(request)
        assert result.domains == _scan_oracle(instance, domains)
        assert CspDomainConsistency.model_validate_json(result.model_dump_json()) == result


def test_repeated_variable_and_false_nullary_constraints_are_exact() -> None:
    instance = _instance(
        2,
        (2, 0),
        (((0, 1), (1, 1)), ()),
        2,
        ((0, (0, 0)), (1, ())),
    )
    result = generalized_arc_consistency(
        CspDomainRequest(instance=instance, domains=((0, 1), (0, 1)))
    )
    assert result.domains == ((1,), (0, 1))
    assert result.empty_domain_variables == ()
    assert result.false_nullary_constraint_ids == ("c1",)


def test_repeated_constraint_occurrences_share_exact_support_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jacobian.math.logic.relational_structures.consistency import operations

    instance = _instance(
        2,
        (2,),
        (((0, 1),),),
        2,
        ((0, (0, 1)), (0, (0, 1))),
    )
    monkeypatch.setattr(operations, "MAX_GAC_CANDIDATE_ROWS", 1)
    result = generalized_arc_consistency(
        CspDomainRequest(instance=instance, domains=((0, 1), (0, 1)))
    )
    assert result.domains == ((0,), (1,))


def test_arc_consistency_does_not_claim_global_satisfiability() -> None:
    # Odd disequality cycle on two labels is unsatisfiable but arc-consistent.
    instance = _instance(
        2,
        (2,),
        (((0, 1), (1, 0)),),
        3,
        ((0, (0, 1)), (0, (1, 2)), (0, (2, 0))),
    )
    result = generalized_arc_consistency(
        CspDomainRequest(instance=instance, domains=((0, 1),) * 3)
    )
    assert result.domains == ((0, 1),) * 3
    assert result.empty_domain_variables == ()


def test_zero_variable_true_nullary_and_domain_validation() -> None:
    instance = _instance(0, (0,), (((),),), 0, ((0, ()),))
    result = generalized_arc_consistency(CspDomainRequest(instance=instance, domains=()))
    assert result.domains == ()
    assert result.false_nullary_constraint_ids == ()
    with pytest.raises(ValidationError):
        CspDomainRequest(instance=instance, domains=((0,),))


def test_candidate_expansion_is_admitted_before_kernel_work(monkeypatch: pytest.MonkeyPatch) -> None:
    from jacobian.math.logic.relational_structures.consistency import operations

    monkeypatch.setattr(operations, "MAX_GAC_CANDIDATE_ROWS", 0)
    instance = _instance(2, (1,), (((0,),),), 1, ((0, (0,)),))
    with pytest.raises(OperationResourceAdmissionError):
        generalized_arc_consistency(
            CspDomainRequest(instance=instance, domains=((0, 1),))
        )


def test_operation_manifest_uses_typed_carriers() -> None:
    assert len(TOOLS) == 1
    declaration = TOOLS[0]
    assert declaration.operation_id == "relational.csp.generalized_arc_consistency.compute"
    assert declaration.request_type is CspDomainRequest
    assert declaration.result_type is CspDomainConsistency
