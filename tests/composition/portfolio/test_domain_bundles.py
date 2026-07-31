"""Portfolio-level tests for the DomainBundle architecture."""

from __future__ import annotations

import pytest

from jacobian.artifacts import ArtifactService
from jacobian.capabilities import CapabilityService
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
    FibonacciPairRequest,
    IntegerPartitionEnumerationRequest,
    LinearRecurrenceEvaluationRequest,
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
from jacobian.memory import ResearchMemory
from jacobian.operation_installation import OperationInstaller
from jacobian.operations import BoundedSearchOperation
from jacobian.schema_registry import SchemaRegistry
from jacobian.store import ArtifactStore

EXPECTED_IDS: frozenset[str] = frozenset(
    {
        "combinatorics.compute.bell",
        "combinatorics.compute.bernoulli",
        "combinatorics.compute.binomial",
        "combinatorics.compute.catalan",
        "combinatorics.compute.central_binomial",
        "combinatorics.compute.compositions",
        "combinatorics.compute.derangements",
        "combinatorics.compute.double_factorial",
        "combinatorics.compute.factorial",
        "combinatorics.compute.fibonacci",
        "combinatorics.compute.fibonacci_pair",
        "combinatorics.compute.lucas",
        "combinatorics.generating_function.coefficients.compute",
        "combinatorics.recurrence.linear.evaluate",
        "combinatorics.compute.motzkin",
        "combinatorics.compute.multinomial",
        "combinatorics.compute.partition_number",
        "combinatorics.compute.permutations",
        "combinatorics.compute.stirling_first",
        "combinatorics.compute.stirling_second",
        "combinatorics.enumerate.integer_partitions",
        "finite_set.compute.difference",
        "finite_set.compute.intersection",
        "finite_set.compute.intersection_cardinality",
        "finite_set.compute.left_cardinality",
        "finite_set.compute.symmetric_difference",
        "finite_set.compute.union",
        "finite_set.compute.union_cardinality",
        "finite_set.decide.disjoint",
        "finite_set.decide.proper_subset",
        "finite_set.decide.subset",
        "integer.compute.absolute_value",
        "integer.compute.aliquot_sum",
        "integer.compute.decimal_digit_count",
        "integer.compute.decimal_digit_sum",
        "integer.compute.divisor_count",
        "integer.compute.divisor_sum",
        "integer.compute.divisors",
        "integer.compute.euler_totient",
        "integer.compute.extended_gcd",
        "integer.compute.floor_square_root",
        "integer.compute.gcd",
        "integer.compute.lcm",
        "integer.compute.mobius",
        "integer.compute.next_prime",
        "integer.compute.nth_prime",
        "integer.compute.nth_root",
        "integer.compute.previous_prime",
        "integer.compute.prime_count",
        "integer.compute.prime_factorization",
        "integer.compute.primorial",
        "integer.compute.proper_divisors",
        "integer.compute.radical",
        "integer.compute.sign",
        "integer.compute.valuation",
        "integer.decide.abundant",
        "integer.decide.coprime",
        "integer.decide.deficient",
        "integer.decide.divides",
        "integer.decide.even",
        "integer.decide.odd",
        "integer.decide.perfect",
        "integer.decide.prime",
        "integer.decide.powerful",
        "integer.decide.square",
        "integer.decide.squarefree",
        "integer.transform.base_digits",
        "modular.compute.inverse",
        "modular.compute.discrete_logarithm",
        "modular.compute.multiplicative_order",
        "modular.enumerate.quadratic_residues",
        "modular.polynomial_residue_image.compute",
        "modular.solve.chinese_remainder",
        "number_theory.compute.jacobi_symbol",
        "number_theory.compute.factorial_valuation",
        "number_theory.compute.legendre_symbol",
        "rational.compute.absolute_value",
        "rational.compute.ceiling",
        "rational.compute.continued_fraction",
        "rational.compute.difference",
        "rational.compute.floor",
        "rational.compute.maximum",
        "rational.compute.minimum",
        "rational.compute.negation",
        "rational.compute.product",
        "rational.compute.quotient",
        "rational.compute.reciprocal",
        "rational.compute.sum",
        "rational.decide.equal",
        "rational.decide.less_than",
        "sequence.compute.distinct_count",
        "sequence.compute.first_differences",
        "sequence.compute.frequencies",
        "sequence.compute.gcd",
        "sequence.compute.lcm",
        "sequence.compute.maximum",
        "sequence.compute.mean",
        "sequence.compute.median",
        "sequence.compute.minimum",
        "sequence.compute.prefix_gcds",
        "sequence.compute.prefix_lcms",
        "sequence.compute.prefix_maxima",
        "sequence.compute.prefix_minima",
        "sequence.compute.prefix_products",
        "sequence.compute.prefix_sums",
        "sequence.compute.product",
        "sequence.compute.range",
        "sequence.compute.second_differences",
        "sequence.compute.sum",
        "sequence.compute.zero_indices",
        "sequence.decide.arithmetic",
        "sequence.decide.geometric",
        "sequence.decide.nondecreasing",
        "sequence.decide.strictly_increasing",
        "sequence.transform.parities",
        "sequence.transform.reverse",
        "sequence.transform.signs",
        "sequence.transform.sort",
        "sequence.transform.sorted_unique",
    }
)

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
    (IntegerNthRootRequest, {"value": 8, "degree": 3}),
    (CombNonnegIntRequest, {"n": 5}),
    (FibonacciPairRequest, {"n": 5}),
    (CombNonnegPairRequest, {"n": 5, "k": 2}),
    (CombIntegerListRequest, {"values": ["2", "1", "1"]}),
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


def test_union_equals_expected_ids() -> None:
    actual = _all_operation_ids()
    assert actual == EXPECTED_IDS, (
        f"missing: {sorted(EXPECTED_IDS - actual)}\n"
        f"extra: {sorted(actual - EXPECTED_IDS)}"
    )


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


@pytest.fixture(scope="module")
def service(tmp_path_factory: pytest.TempPathFactory) -> CapabilityService:
    store = ArtifactStore(tmp_path_factory.mktemp("domain-bundles"))
    schemas = SchemaRegistry(store)
    artifacts = ArtifactService(store, schemas)
    service = CapabilityService(store, ResearchMemory(store, schemas))
    installer = OperationInstaller(store, schemas, artifacts)
    for bundle in ALL_BUNDLES:
        for adapter in installer.install(bundle).adapters:
            service.register(adapter)
    return service


def test_catalog_covers_all_operations(service: CapabilityService) -> None:
    catalog_ids = {d.capability_id for d in service.catalog().capabilities}
    assert catalog_ids == EXPECTED_IDS, (
        f"missing from catalog: {sorted(EXPECTED_IDS - catalog_ids)}\n"
        f"extra in catalog: {sorted(catalog_ids - EXPECTED_IDS)}"
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
            if isinstance(operation, BoundedSearchOperation):
                assert len(result.artifact_uris) == 3
                assert len(result.obligations) == 1
            else:
                assert len(result.artifact_uris) == 2
                assert result.output["backend_version"] == bundle.backend_version
            assert result.relationships[0].relation_id == operation.relation_id
            input_uri, result_uri, *_ = result.artifact_uris
            assert service.store.get(result_uri).manifest.parents == (input_uri,), (
                f"{operation.capability_id}: parent mismatch"
            )
            expected_output = (
                result.output
                if isinstance(operation, BoundedSearchOperation)
                else result.output["result"]
            )
            assert service.store.get(result_uri).payload == expected_output, (
                f"{operation.capability_id}: materialized payload mismatch"
            )
