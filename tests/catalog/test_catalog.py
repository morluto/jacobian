from __future__ import annotations

from collections.abc import Callable
from typing import cast

import pytest

from jacobian._models import StrictModel
from jacobian.catalog import catalog as catalog_module
from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import MathTool, OperationMatchRequest
from jacobian.catalog.search import browse_operations, match_operations
from jacobian.dispatch import invoke_operation


class _BindingRequest(StrictModel):
    value: int


class _BindingResult(StrictModel):
    doubled: int


class _WrongBindingResult(StrictModel):
    value: int


class _ExtraBindingResult(_BindingResult):
    leaked: str


def test_catalog_inspects_determinant_without_sqlite() -> None:
    catalog = Catalog.open()

    descriptor = catalog.inspect("matrix.determinant.compute")
    assert descriptor is not None
    assert descriptor.operation_id == "matrix.determinant.compute"


def test_output_schema_describes_serialized_exact_integers() -> None:
    descriptor = Catalog.open().inspect(
        "number_theory.euler_phi.preimage_power_sums.compute"
    )
    assert descriptor is not None
    assert descriptor.output_schema["properties"]["power_sum"]["type"] == "string"


def test_every_served_operation_publishes_request_valid_examples() -> None:
    catalog = Catalog.open()

    for descriptor in catalog.snapshot().operations:
        operation = catalog.operation(descriptor.operation_id)
        assert operation is not None
        assert operation.examples, (
            f"{descriptor.operation_id} must publish an invocation example"
        )
        for invocation_example in operation.examples:
            operation.request_type.model_validate(invocation_example.input)


def test_checked_catalog_binding_rejects_an_incorrect_declared_result() -> None:
    def wrong_result(_request: _BindingRequest) -> _WrongBindingResult:
        return _WrongBindingResult(value=1)

    operation = MathTool(
        operation_id="test.binding.result",
        title="Test checked result binding",
        description="Exercise the private heterogeneous catalog boundary.",
        request_type=_BindingRequest,
        result_type=_BindingResult,
        run=cast(Callable[[_BindingRequest], _BindingResult], wrong_result),
    )

    with pytest.raises(TypeError, match="returned a result outside its declared type"):
        invoke_operation("test.binding.result", {"value": 1}, Catalog((operation,)))


def test_checked_catalog_binding_passes_exact_declared_result_unchanged() -> None:
    def double(_request: _BindingRequest) -> _BindingResult:
        return _BindingResult(doubled=2)

    operation = MathTool(
        operation_id="test.binding.exact",
        title="Test exact result binding",
        description="Exercise the private heterogeneous catalog boundary.",
        request_type=_BindingRequest,
        result_type=_BindingResult,
        run=double,
    )

    result = invoke_operation("test.binding.exact", {"value": 1}, Catalog((operation,)))

    assert set(result.output) == {"doubled"}
    assert result.output["doubled"] == 2


def test_checked_catalog_binding_rejects_result_subclass_extra_fields() -> None:
    def leak_extra_field(_request: _BindingRequest) -> _ExtraBindingResult:
        return _ExtraBindingResult(doubled=2, leaked="private")

    operation = MathTool(
        operation_id="test.binding.subclass",
        title="Test subclass result binding",
        description="Exercise the private heterogeneous catalog boundary.",
        request_type=_BindingRequest,
        result_type=_BindingResult,
        run=cast(Callable[[_BindingRequest], _BindingResult], leak_extra_field),
    )

    with pytest.raises(TypeError, match="returned a result outside its declared type"):
        invoke_operation("test.binding.subclass", {"value": 1}, Catalog((operation,)))


@pytest.mark.parametrize(
    ("discovery_terms", "message"),
    (
        (("inverse totient", "inverse totient"), "must be unique"),
        ((" ",), "must not be empty"),
        (tuple(f"term_{index}" for index in range(9)), "at most 8"),
    ),
)
def test_math_tool_keeps_discovery_vocabulary_small_and_reviewable(
    discovery_terms: tuple[str, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        MathTool(
            operation_id="test.discovery.terms",
            title="Test discovery terms",
            description="Exercise immutable declaration terminology validation.",
            request_type=_BindingRequest,
            result_type=_BindingResult,
            run=lambda _request: _BindingResult(doubled=2),
            discovery_terms=discovery_terms,
        )


def test_explicit_binary_profile_remains_the_published_code_profile() -> None:
    operation = Catalog.open().operation("code.binary.explicit.profile.compute")

    assert operation is not None
    assert operation.operation_id == "code.binary.explicit.profile.compute"


def test_compact_discovery_matches_full_descriptor_discovery() -> None:
    catalog = Catalog.open()
    descriptors = catalog.snapshot().operations
    operations = tuple(
        operation
        for descriptor in descriptors
        if (operation := catalog.operation(descriptor.operation_id)) is not None
    )
    request = OperationMatchRequest(need="matrix determinant", limit=2)

    expected_search = match_operations(descriptors, request)
    assert match_operations(operations, request) == expected_search
    if expected_search.next_cursor is not None:
        next_request = request.model_copy(
            update={"cursor": expected_search.next_cursor}
        )
        assert match_operations(operations, next_request) == match_operations(
            descriptors, next_request
        )

    expected_browse = browse_operations(
        descriptors, namespace="matrix", limit=2, cursor=None
    )
    assert (
        browse_operations(operations, namespace="matrix", limit=2, cursor=None)
        == expected_browse
    )
    if expected_browse.next_cursor is not None:
        assert browse_operations(
            operations,
            namespace="matrix",
            limit=2,
            cursor=expected_browse.next_cursor,
        ) == browse_operations(
            descriptors,
            namespace="matrix",
            limit=2,
            cursor=expected_browse.next_cursor,
        )


def test_match_and_browse_do_not_materialize_descriptors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = Catalog.open()

    def fail_descriptor(*_args: object) -> None:
        raise AssertionError("compact discovery must not construct full descriptors")

    monkeypatch.setattr(catalog_module, "_descriptor", fail_descriptor)

    search = catalog.match(OperationMatchRequest(need="matrix", limit=2))
    browse = catalog.browse(namespace="matrix", limit=2, cursor=None)

    assert search.matches
    assert browse.operations


def test_namespace_filters_only_the_primary_operation_id_segment() -> None:
    catalog = Catalog.open()

    search = catalog.match(
        OperationMatchRequest(need="polynomial", namespace="polynomial", limit=20)
    )
    browse = catalog.browse(namespace="polynomial", limit=20, cursor=None)

    assert search.matches
    assert browse.operations
    assert all(match.operation_id.startswith("polynomial.") for match in search.matches)
    assert all(
        card.operation_id.startswith("polynomial.") for card in browse.operations
    )


def test_natural_prime_power_query_ranks_factorization_before_prime_navigation() -> (
    None
):
    catalog = Catalog.open()

    result = catalog.match(
        OperationMatchRequest(
            need="factor an integer into prime powers",
            # Keep the complete navigation comparison within the public
            # discovery limit; new factor-related operations may occupy the
            # first ten slots without changing the relative ordering.
            limit=20,
        )
    )
    assert result.matches[0].operation_id == "integer.compute.prime_factorization"


def test_natural_powerful_number_query_finds_bounded_decision() -> None:
    catalog = Catalog.open()

    result = catalog.match(
        OperationMatchRequest(
            need="decide whether an integer is 2-full or powerful",
            limit=5,
        )
    )

    assert "integer.decide.powerful" in {match.operation_id for match in result.matches}


def test_catalog_runs_source_bound_powerful_decision() -> None:
    catalog = Catalog.open()
    operation = catalog.operation("integer.decide.powerful")
    assert operation is not None

    result = invoke_operation(
        "integer.decide.powerful",
        {"value": "12168"},
        catalog,
    )

    assert result.output is not None
    assert result.output["is_powerful"] is True
    assert result.output["value"] == "12168"


def test_global_search_finds_lattice_hnf() -> None:
    catalog = Catalog.open()

    result = catalog.match(
        OperationMatchRequest(
            need="row Hermite normal form",
            limit=10,
        )
    )

    assert "lattice.hermite_normal_form.compute" in {
        match.operation_id for match in result.matches
    }


def test_search_finds_generalized_exact_cover() -> None:
    catalog = Catalog.open()

    result = catalog.match(
        OperationMatchRequest(need="generalized exact cover", limit=5)
    )

    assert result.matches[0].operation_id == (
        "combinatorics.generalized_exact_cover.find"
    )


def test_browse_includes_lattice_hnf_in_its_primary_namespace() -> None:
    catalog = Catalog.open()

    result = catalog.browse(namespace="lattice", limit=100, cursor=None)

    assert "lattice.hermite_normal_form.compute" in {
        operation.operation_id for operation in result.operations
    }
