"""Behavioral invariants for the installed operation catalog."""

from __future__ import annotations

import pytest

from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationDiscoveryRequest
from jacobian.catalog.search import matches_namespace


def test_catalog_rejects_duplicate_tool_ids() -> None:
    catalog = Catalog.open()
    operation = catalog.operation("integer.compute.extended_gcd")
    assert operation is not None

    with pytest.raises(ValueError, match="duplicate built-in operation ID"):
        Catalog((operation, operation))


def test_each_tool_contract_and_function_have_one_math_owner() -> None:
    for operation in BUILTIN_TOOLS:
        modules = {
            operation.request_type.__module__,
            operation.result_type.__module__,
            operation.run.__module__,
        }
        non_math_modules = {
            module for module in modules if not module.startswith("jacobian.math.")
        }
        assert not non_math_modules, (
            f"{operation.operation_id} has non-math owners: {sorted(non_math_modules)}"
        )
        # The operation's home domain owns its request contract and run
        # function.  A result may be another math domain's canonical value:
        # producers return the domain-owned canonical type unchanged
        # (AGENTS.md) instead of recreating it per operation.
        home_modules = {
            operation.request_type.__module__,
            operation.run.__module__,
        }
        home_owners = {
            module.removeprefix("jacobian.math.").split(".", 1)[0]
            for module in home_modules
        }
        assert len(home_owners) == 1, (
            f"{operation.operation_id} spans mathematical owners: {sorted(modules)}"
        )


def test_geometry_incidence_search_stays_one_capability_family() -> None:
    """Exhaustive collinear-triple and concyclic-quadruple search has one
    public operation (``geometry.points.general_position.search``, which
    returns both complete witness sets with certified absence); per-kind
    projections of the same postcondition must not re-enter the catalog as
    near-duplicate discovery entries."""
    public_ids = {tool.operation_id for tool in BUILTIN_TOOLS}
    assert "geometry.points.general_position.search" in public_ids
    assert not public_ids & {
        "geometry.points.collinear_triples.find",
        "geometry.points.concyclic_quadruples.find",
    }


def test_prime_field_matrix_computations_have_one_operation_family() -> None:
    """Rank, RREF, and nullspace over GF(p) are owned by ``prime_field.matrix``.

    A second family exposing the same kernels under different IDs made
    agents discover six competing operations for three computations.
    """
    matrix_ids = sorted(
        tool.operation_id
        for tool in BUILTIN_TOOLS
        if tool.operation_id.startswith("prime_field")
    )
    assert matrix_ids == [
        "prime_field.matrix.nullspace.compute",
        "prime_field.matrix.rank.compute",
        "prime_field.matrix.rref.compute",
    ]


def test_linear_code_dual_and_syndrome_have_one_operation_family() -> None:
    """Dual codes and syndromes of canonical prime-field encoders are owned
    by ``code.linear``; duplicate IDs routing to the identical request,
    result, and callable must not re-enter public discovery."""
    public_ids = {tool.operation_id for tool in BUILTIN_TOOLS}
    assert {"code.linear.dual.compute", "code.linear.syndrome.compute"} <= public_ids
    assert not public_ids & {
        "code.dual_code.compute",
        "code.syndrome.compute",
    }


def test_search_browse_and_inspect_results_stay_within_the_public_catalog() -> None:
    catalog = Catalog.open()
    public_ids = {
        descriptor.operation_id for descriptor in catalog.snapshot().operations
    }
    search = catalog.search(
        OperationDiscoveryRequest(query="finite field factorization", limit=5)
    )
    browse = catalog.browse(namespace="graph", limit=5, cursor=None)
    inspected = catalog.inspect("integer.compute.extended_gcd")

    assert search.matches
    assert len(search.matches) <= 5
    assert {match.operation_id for match in search.matches} <= public_ids
    assert search.total_matches >= len(search.matches)

    assert len(browse.operations) <= 5
    assert {operation.operation_id for operation in browse.operations} <= public_ids
    assert browse.total_operations == sum(
        1 for tool in BUILTIN_TOOLS if matches_namespace(tool, "graph")
    )
    assert browse.total_operations >= len(browse.operations)

    assert inspected is not None
    assert inspected.operation_id == "integer.compute.extended_gcd"
    assert "version" not in inspected.model_dump(mode="json")
