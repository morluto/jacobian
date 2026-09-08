"""Native-only discrete certificate consumers preserve operational noncompletion."""

from importlib import import_module
from types import ModuleType

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.combinatorics.additive.zero_sum_atoms._models import (
    ZeroSumAtomSource,
)
from jacobian.math.combinatorics.discrepancy._models import FiniteSetSystem
from jacobian.math.combinatorics.finite_structures.sets._models import FiniteIntegerSet
from jacobian.math.groups.finite_abelian import FiniteAbelianProductGroup

_RATIONALS = (CanonicalRational(num=1, den=1), CanonicalRational(num=2, den=1))
_SET = FiniteIntegerSet(elements=(1, 2))
_ATOMS = ZeroSumAtomSource(
    group=FiniteAbelianProductGroup(moduli=(3,)), elements=((1,), (2,))
)
_CASES = (
    (
        "arithmetic_progression_hypergraph",
        "construct_arithmetic_progression_hypergraph",
        "verify_arithmetic_progression_hypergraph",
        (1, 5, 3),
        "_models",
        "MAX_EDGES",
    ),
    (
        "finite_structures.axis_aligned_square_grid",
        "construct_axis_aligned_square_grid",
        "verify_axis_aligned_square_grid",
        (3,),
        "operations",
        "MAX_EDGES",
    ),
    (
        "finite_structures.divisibility_sum_triples",
        "construct_divisibility_sum_triples_hypergraph",
        "verify_divisibility_sum_triples",
        (1, 5),
        "operations",
        "MAX_TRIPLE_ENUMERATION",
    ),
    (
        "additive.cyclic_sumset_profile",
        "compute_cyclic_sumset_profile",
        "verify_cyclic_sumset_profile",
        (3, (0, 1), (0, 1)),
        "operations",
        "MAX_CYCLIC_SUMSET_PAIRS",
    ),
    (
        "additive.product_representation",
        "compute_product_representation_profile",
        "verify_product_representation_profile",
        (_SET, _SET),
        "operations",
        "MAX_PRODUCT_REPRESENTATION_PAIRS",
    ),
    (
        "additive.gowers_cube_profile",
        "compute_gowers_cube_profile",
        "verify_gowers_cube_profile",
        (3, (0, 1), 2),
        "operations",
        "MAX_GOWERS_CUBE_VERTEX_CHECKS",
    ),
    (
        "additive.rational_fixed_arity",
        "compute_rational_fixed_arity_sum_profile",
        "verify_rational_fixed_arity_sum_profile",
        (_RATIONALS, 1),
        "operations",
        "MAX_ENUMERATION_WORK",
    ),
    (
        "additive.rational_subset_sum",
        "compute_rational_subset_sum_profile",
        "verify_rational_subset_sum_profile",
        (_RATIONALS,),
        "operations",
        "MAX_SEQUENCE_LENGTH",
    ),
    (
        "additive.zero_sum_atoms",
        "construct_zero_sum_atom_hypergraph",
        "verify_zero_sum_atom_hypergraph",
        (_ATOMS,),
        "operations",
        "MAX_ATOM_SUBSET_CHECKS",
    ),
)


def _module(owner: str, component: str = "operations") -> ModuleType:
    return import_module("jacobian.math.combinatorics." + owner + "." + component)


@pytest.mark.parametrize("owner,producer,verifier,args,budget_module,budget", _CASES)
@pytest.mark.parametrize(
    "error_type", [RuntimeError, ValueError, TypeError, MemoryError, TimeoutError]
)
def test_native_verifier_preserves_backend_failure(
    monkeypatch: pytest.MonkeyPatch,
    owner: str,
    producer: str,
    verifier: str,
    args: tuple[object, ...],
    budget_module: str,
    budget: str,
    error_type: type[Exception],
) -> None:
    module = _module(owner)
    claim = getattr(module, producer)(*args)
    verify = getattr(module, verifier)
    assert verify(claim)

    def fail(*args: object, **kwargs: object) -> None:
        raise error_type("backend failure")

    monkeypatch.setattr(module, producer, fail)
    with pytest.raises(error_type, match="backend failure"):
        verify(claim)


@pytest.mark.parametrize("owner,producer,verifier,args,budget_module,budget", _CASES)
def test_native_verifier_preserves_actual_resource_refusal(
    monkeypatch: pytest.MonkeyPatch,
    owner: str,
    producer: str,
    verifier: str,
    args: tuple[object, ...],
    budget_module: str,
    budget: str,
) -> None:
    module = _module(owner)
    claim = getattr(module, producer)(*args)
    monkeypatch.setattr(_module(owner, budget_module), budget, 0)
    with pytest.raises(OperationResourceAdmissionError):
        getattr(module, verifier)(claim)


def test_single_atom_verifier_preserves_resource_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module("additive.zero_sum_atoms")
    assert module.verify_zero_sum_atom(_ATOMS, (0, 1))
    monkeypatch.setattr(module, "MAX_ATOM_SUBSET_CHECKS", 0)
    with pytest.raises(OperationResourceAdmissionError):
        module.verify_zero_sum_atom(_ATOMS, (0, 1))


def test_discrepancy_verifier_preserves_backend_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module("discrepancy", "_tools")
    source = FiniteSetSystem(ground_set_size=2, sets=((0, 1),))
    claim = module._compute_discrepancy_native(source, (1, -1))
    assert module.verify_discrepancy(claim)

    def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError("backend failure")

    monkeypatch.setattr(module, "_compute_discrepancy_native", fail)
    with pytest.raises(RuntimeError, match="backend failure"):
        module.verify_discrepancy(claim)
