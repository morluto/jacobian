"""Portfolio-level tests for the DomainBundle architecture."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from jacobian.artifacts import ArtifactService
from jacobian.capability_service import CapabilityService
from jacobian.contracts.arithmetic import (
    IntegerBaseDigitsRequest,
    IntegerNthRootRequest,
)
from jacobian.contracts.arithmetic import (
    IntegerValueRequest as ArithIntegerValueRequest,
)
from jacobian.contracts.capabilities import (
    CapabilityAssuranceLevel,
    CapabilityDescriptor,
    CapabilityRequest,
)
from jacobian.contracts.combinatorics import (
    BinomialRequest,
    CyclicDifferenceSetExtensionRequest,
    CyclicPerfectDifferenceSetRequest,
    FibonacciPairRequest,
    IntegerPartitionEnumerationRequest,
    IntegerSidonRequest,
    LinearRecurrenceEvaluationRequest,
    PolynomialCoefficientRecurrenceEvaluationRequest,
    RationalGeneratingFunctionCoefficientsRequest,
)
from jacobian.contracts.combinatorics import (
    IntegerListRequest as CombIntegerListRequest,
)
from jacobian.contracts.combinatorics import (
    NonnegativeIntegerRequest as CombNonnegIntRequest,
)
from jacobian.contracts.combinatorics import (
    NonnegativePairRequest as CombNonnegPairRequest,
)
from jacobian.contracts.finite_sets import FiniteSetPairRequest
from jacobian.contracts.number_theory import (
    ArithmeticFunctionRequest,
    ChineseRemainderRequest,
    DiscreteLogarithmRequest,
    DivisibilityRequest,
    FactorialValuationRequest,
    FactorizationRequest,
    FloorSquareRootRequest,
    JacobiSymbolRequest,
    LegendreSymbolRequest,
    ModularPolynomialResidueImageRequest,
    ModularValueRequest,
    ModulusRequest,
    PositiveIntegerRequest,
    PowerfulNumberRequest,
    ValuationRequest,
)
from jacobian.contracts.number_theory import (
    IntegerPairRequest as NTIntegerPairRequest,
)
from jacobian.contracts.number_theory import (
    IntegerValueRequest as NTIntegerValueRequest,
)
from jacobian.contracts.number_theory import (
    NonnegativeIntegerRequest as NTNonnegIntRequest,
)
from jacobian.contracts.rationals import RationalPairRequest, RationalValueRequest
from jacobian.contracts.results import ContractModel, ExecutionStatus
from jacobian.contracts.sequences import IntegerSequenceRequest
from jacobian.domains.arithmetic import build_arithmetic_bundle
from jacobian.domains.combinatorics import build_combinatorics_bundle
from jacobian.domains.finite_sets import build_finite_set_bundle
from jacobian.domains.number_theory import build_number_theory_bundle
from jacobian.domains.sequences import build_sequence_bundle
from jacobian.operation_installation import OperationInstaller
from jacobian.operations import (
    BoundedSearchOperation,
    ComputedOperation,
    MaterializedOperation,
)
from jacobian.schema_registry import SchemaRegistry
from jacobian.storage.repository import ArtifactRepository

ALL_BUNDLES = (
    build_arithmetic_bundle(),
    build_combinatorics_bundle(),
    build_finite_set_bundle(),
    build_number_theory_bundle(),
    build_sequence_bundle(),
)

_REPR: list[tuple[type[ContractModel], dict[str, object]]] = [
    (ArithIntegerValueRequest, {"value": "12"}),
    (IntegerBaseDigitsRequest, {"value": "12", "base": 2}),
    (IntegerNthRootRequest, {"value": "8", "degree": 3}),
    (CombNonnegIntRequest, {"n": 5}),
    (FibonacciPairRequest, {"n": 5}),
    (CombNonnegPairRequest, {"n": 5, "k": 2}),
    (BinomialRequest, {"n": 5, "k": 2}),
    (CombIntegerListRequest, {"values": ["2", "1", "1"]}),
    (IntegerSidonRequest, {"elements": ["1", "2", "4"]}),
    (CyclicPerfectDifferenceSetRequest, {"modulus": 7, "residues": [0, 1, 3]}),
    (
        CyclicDifferenceSetExtensionRequest,
        {"base_elements": ["0", "1"], "target_order": 3},
    ),
    (IntegerPartitionEnumerationRequest, {"n": 5, "max_parts": 3}),
    (
        LinearRecurrenceEvaluationRequest,
        {
            "coefficients": [
                {"num": "1", "den": "1"},
                {"num": "1", "den": "1"},
            ],
            "initial_values": [
                {"num": "0", "den": "1"},
                {"num": "1", "den": "1"},
            ],
            "coefficient_convention": (
                "A_N_EQUALS_SUM_C_J_TIMES_A_N_MINUS_J_FOR_J_FROM_1"
            ),
            "scope": "PREFIX",
            "term_count": 6,
            "indices": [],
        },
    ),
    (
        RationalGeneratingFunctionCoefficientsRequest,
        {
            "numerator": [{"num": "1", "den": "1"}],
            "denominator": [
                {"num": "1", "den": "1"},
                {"num": "-1", "den": "1"},
            ],
            "coefficient_convention": "ASCENDING_POWERS_OF_X",
            "expansion_point": "0",
            "truncation_order": 6,
        },
    ),
    (
        PolynomialCoefficientRecurrenceEvaluationRequest,
        {
            "coefficient_polynomials": [
                [{"num": "1", "den": "1"}],
                [{"num": "-1", "den": "1"}],
            ],
            "initial_values": [{"num": "1", "den": "1"}],
            "coefficient_convention": (
                "SUM_P_J_OF_N_TIMES_A_N_MINUS_J_EQUALS_ZERO_FOR_J_FROM_0"
            ),
            "polynomial_convention": "ASCENDING_POWERS_OF_N",
            "scope": "PREFIX",
            "term_count": 6,
            "indices": [],
        },
    ),
    (
        FiniteSetPairRequest,
        {"left": {"elements": ["1", "2"]}, "right": {"elements": ["2", "3"]}},
    ),
    (NTIntegerValueRequest, {"value": "12"}),
    (NTIntegerPairRequest, {"left": "12", "right": "8"}),
    (DivisibilityRequest, {"divisor": "3", "dividend": "12"}),
    (
        FactorizationRequest,
        {"value": "12", "resource_budget": {"wall_seconds": 5}},
    ),
    (
        PowerfulNumberRequest,
        {"value": "72", "resource_budget": {"wall_seconds": 5}},
    ),
    (
        ArithmeticFunctionRequest,
        {"n": 30, "resource_budget": {"wall_seconds": 5}},
    ),
    (ValuationRequest, {"value": "12", "prime": "2"}),
    (NTNonnegIntRequest, {"n": 10}),
    (PositiveIntegerRequest, {"n": 10}),
    (ModularValueRequest, {"value": "3", "modulus": 7}),
    (ModulusRequest, {"modulus": 7}),
    (
        ModularPolynomialResidueImageRequest,
        {
            "modulus": 7,
            "variables": [
                {"name": "x", "residues": [0, 1, 2, 3, 4, 5, 6]},
            ],
            "terms": [{"coefficient": "4", "exponents": [3]}],
        },
    ),
    (ChineseRemainderRequest, {"residues": [2, 3], "moduli": [3, 5]}),
    (JacobiSymbolRequest, {"a": "10", "n": 21}),
    (FloorSquareRootRequest, {"n": 12}),
    (LegendreSymbolRequest, {"a": 2, "prime": 7}),
    (FactorialValuationRequest, {"n": 10, "base": 2}),
    (
        DiscreteLogarithmRequest,
        {"base": 7, "target": 15, "modulus": 41},
    ),
    (RationalValueRequest, {"value": {"num": "1", "den": "2"}}),
    (
        RationalPairRequest,
        {"left": {"num": "1", "den": "2"}, "right": {"num": "1", "den": "3"}},
    ),
    (IntegerSequenceRequest, {"values": ["1", "2", "3"]}),
]
REPRESENTATIVE_PAYLOADS: dict[type[ContractModel], dict[str, object]] = dict(_REPR)


def _all_operation_ids() -> set[str]:
    ids: set[str] = set()
    for bundle in ALL_BUNDLES:
        for operation in bundle.capabilities:
            ids.add(operation.capability_id)
    return ids


def test_installed_bundles_expose_operations() -> None:
    actual = _all_operation_ids()
    assert actual, "expected at least one DomainBundle operation"
    assert len(actual) == sum(len(bundle.capabilities) for bundle in ALL_BUNDLES)


def test_unique_ids_within_each_bundle() -> None:
    for bundle in ALL_BUNDLES:
        ids = [op.capability_id for op in bundle.capabilities]
        assert len(ids) == len(set(ids)), (
            f"{bundle.domain_id}: duplicates {[i for i in ids if ids.count(i) > 1]}"
        )


def test_no_id_in_two_bundles() -> None:
    seen: dict[str, str] = {}
    for bundle in ALL_BUNDLES:
        for operation in bundle.capabilities:
            cap_id = operation.capability_id
            assert cap_id not in seen, (
                f"{cap_id!r} in both {seen[cap_id]!r} and {bundle.domain_id!r}"
            )
            seen[cap_id] = bundle.domain_id


def test_unique_domain_ids() -> None:
    domain_ids = [b.domain_id for b in ALL_BUNDLES]
    assert len(domain_ids) == len(set(domain_ids)), f"duplicates: {domain_ids}"


@pytest.fixture
def service(tmp_path: Path) -> Iterator[CapabilityService]:
    store = ArtifactRepository(tmp_path / "state")
    schemas = SchemaRegistry(store)
    artifacts = ArtifactService(store, schemas)
    service = CapabilityService(store)
    installer = OperationInstaller(store, schemas, artifacts)
    for bundle in ALL_BUNDLES:
        for adapter in installer.install(bundle).adapters:
            service.register(adapter)
    try:
        yield service
    finally:
        store.close()


def test_catalog_covers_all_operations(service: CapabilityService) -> None:
    catalog_ids = {d.capability_id for d in service.catalog().capabilities}
    expected = _all_operation_ids()
    assert catalog_ids == expected, (
        f"missing from catalog: {sorted(expected - catalog_ids)}\n"
        f"extra in catalog: {sorted(catalog_ids - expected)}"
    )


def test_catalog_descriptors_match_operations(service: CapabilityService) -> None:
    by_id: dict[str, CapabilityDescriptor] = {
        d.capability_id: d for d in service.catalog().capabilities
    }
    for bundle in ALL_BUNDLES:
        for operation in bundle.capabilities:
            desc = by_id[operation.capability_id]
            assert desc.version == operation.version
            assert desc.title == operation.title
            assert desc.description == operation.description
            assert desc.provider == bundle.provider_runtime.provider
            assert desc.tags == operation.tags


def test_representative_payloads_invoke_all_operations(
    service: CapabilityService,
) -> None:
    for bundle in ALL_BUNDLES:
        for operation in bundle.capabilities:
            payload = REPRESENTATIVE_PAYLOADS.get(operation.request_model)
            assert payload is not None, (
                f"{operation.capability_id}: no representative payload for "
                f"{operation.request_model}"
            )
            result = service.invoke(
                CapabilityRequest(capability_id=operation.capability_id, input=payload)
            )
            assert result.capability_id == operation.capability_id
            assert result.capability_version == operation.version
            assert result.execution.status is ExecutionStatus.COMPLETED, (
                operation.capability_id,
                result.diagnostics,
            )
            assert result.assurance.level is CapabilityAssuranceLevel.COMPUTED
            if isinstance(operation, ComputedOperation):
                assert result.artifact_uris == ()
                assert result.relationships == ()
                assert result.output["backend_version"] == bundle.backend_version
                continue
            if isinstance(operation, BoundedSearchOperation):
                assert len(result.artifact_uris) == 3
                assert len(result.obligations) == 1
            else:
                assert isinstance(operation, MaterializedOperation)
                assert len(result.artifact_uris) == 2
                assert result.output["backend_version"] == bundle.backend_version
            assert result.relationships[0].relation_id == operation.relation_id
            input_uri, result_uri, *_ = result.artifact_uris
            assert service.store.get(result_uri).manifest.parents == (input_uri,), (
                f"{operation.capability_id}: parent mismatch"
            )
            stored_result = service.store.get(result_uri)
            assert isinstance(stored_result.payload, dict)
            if isinstance(operation, BoundedSearchOperation):
                assert stored_result.payload == result.output, (
                    f"{operation.capability_id}: materialized payload mismatch"
                )
