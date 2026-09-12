from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.posets.core._closure_models import LowerClosureRequest
from jacobian.math.combinatorics.posets.core._closure_tools import lower_closure
from jacobian.math.combinatorics.posets.core._models import (
    ElementRank,
    FinitePoset,
    FinitePosetRequest,
    IncomparablePair,
    LinearExtensionRequest,
    MobiusFunctionRequest,
    OrderedPair,
)
from jacobian.math.combinatorics.posets.core.operations import (
    finite_poset_digest,
    linear_extension_count,
    materialize_finite_poset,
    verify_finite_poset,
    width,
)


def _materialize(elements: list[str], relation: list[tuple[str, str]]) -> FinitePoset:
    request = FinitePosetRequest.model_validate(
        {
            "elements": elements,
            "relation": [{"lower": lower, "upper": upper} for lower, upper in relation],
            "interpretation": "COVER_EDGES",
        }
    )
    return materialize_finite_poset(
        request.elements,
        request.relation,
        request.interpretation,
        request.reflexive_pairs,
    )


def _materialize_request(request: FinitePosetRequest) -> FinitePoset:
    return materialize_finite_poset(
        request.elements,
        request.relation,
        request.interpretation,
        request.reflexive_pairs,
    )


def _assert_code(exc: pytest.ExceptionInfo[ValidationError], code: str) -> None:
    assert exc.value.errors()[0]["type"] == code


def _assert_operation_code(
    exc: pytest.ExceptionInfo[OperationDomainValidationError], code: str
) -> None:
    assert exc.value.errors()[0]["type"] == code


def test_cover_relation_rejects_cycles_and_redundant_edges() -> None:
    with pytest.raises(OperationDomainValidationError) as exc:
        _materialize_request(
            FinitePosetRequest.model_validate(
                {
                    "elements": ["a", "b"],
                    "relation": [
                        {"lower": "a", "upper": "b"},
                        {"lower": "b", "upper": "a"},
                    ],
                    "interpretation": "COVER_EDGES",
                }
            )
        )
    _assert_operation_code(exc, "poset.relation_antisymmetric")
    with pytest.raises(OperationDomainValidationError) as exc:
        _materialize_request(
            FinitePosetRequest.model_validate(
                {
                    "elements": ["a", "b", "c"],
                    "relation": [
                        {"lower": "a", "upper": "b"},
                        {"lower": "b", "upper": "c"},
                        {"lower": "a", "upper": "c"},
                    ],
                    "interpretation": "COVER_EDGES",
                }
            )
        )
    _assert_operation_code(exc, "poset.cover_edges_transitive_redundancy")


def test_serialized_poset_keeps_order_profile_as_a_claim() -> None:
    poset = FinitePoset.model_validate(
        {
            "elements": ["a", "b"],
            "strict_order_pairs": [
                {"lower": "a", "upper": "b"},
                {"lower": "b", "upper": "a"},
            ],
            "cover_relations": [],
            "incomparable_pairs": [],
            "minimal_elements": [],
            "maximal_elements": [],
            "graded": False,
            "ranks": None,
            "poset_digest": "sha256:" + "0" * 64,
        }
    )
    assert verify_finite_poset(poset) is False
    with pytest.raises(OperationDomainValidationError, match="antisymmetric"):
        width(poset)
    with pytest.raises(OperationDomainValidationError, match="antisymmetric"):
        lower_closure(LowerClosureRequest.model_construct(poset=poset, subset=("a",)))


def test_consumers_reject_forged_noncanonical_carrier_order() -> None:
    poset = _materialize(["a", "b"], [("a", "b")])
    assert verify_finite_poset(poset) is True
    strict_order_pairs = (OrderedPair(lower="a", upper="b"),)
    cover_relations = (OrderedPair(lower="a", upper="b"),)
    ranks = (ElementRank(element="b", rank=1), ElementRank(element="a", rank=0))
    forged = FinitePoset.model_construct(
        elements=("b", "a"),
        strict_order_pairs=strict_order_pairs,
        cover_relations=cover_relations,
        incomparable_pairs=(),
        minimal_elements=("a",),
        maximal_elements=("b",),
        graded=True,
        ranks=ranks,
        poset_digest=finite_poset_digest(
            elements=("b", "a"),
            strict_order_pairs=strict_order_pairs,
            cover_relations=cover_relations,
            incomparable_pairs=(),
            minimal_elements=("a",),
            maximal_elements=("b",),
            graded=True,
            ranks=ranks,
        ),
    )

    assert verify_finite_poset(forged) is False
    with pytest.raises(OperationDomainValidationError, match="canonical finite poset"):
        lower_closure(LowerClosureRequest.model_construct(poset=forged, subset=("b",)))


def test_consumers_reject_model_bypassed_nested_claims_as_domain_errors() -> None:
    poset = _materialize(["a", "b"], [])
    malformed_claims = (
        poset.model_copy(update={"incomparable_pairs": (object(),)}),
        poset.model_copy(update={"strict_order_pairs": (object(),)}),
        poset.model_copy(
            update={
                "strict_order_pairs": (
                    OrderedPair.model_construct(lower=object(), upper="b"),
                )
            }
        ),
    )

    for malformed in malformed_claims:
        assert verify_finite_poset(malformed) is False
        with pytest.raises(
            OperationDomainValidationError, match="canonical finite poset"
        ):
            width(malformed)


def test_consumers_reject_mixed_carrier_elements_as_domain_errors() -> None:
    poset = _materialize(["a"], [])
    malformed = poset.model_copy(update={"elements": ("a", object())})

    assert verify_finite_poset(malformed) is False
    with pytest.raises(OperationDomainValidationError, match="canonical finite poset"):
        width(malformed)


def test_consumers_reject_integer_graded_flag_as_domain_errors() -> None:
    poset = _materialize(["a", "b"], [("a", "b")])
    forged = poset.model_copy(
        update={
            "graded": 1,
            "poset_digest": finite_poset_digest(
                elements=poset.elements,
                strict_order_pairs=poset.strict_order_pairs,
                cover_relations=poset.cover_relations,
                incomparable_pairs=poset.incomparable_pairs,
                minimal_elements=poset.minimal_elements,
                maximal_elements=poset.maximal_elements,
                graded=1,  # type: ignore[arg-type]
                ranks=poset.ranks,
            ),
        }
    )

    assert type(forged.graded) is int
    assert forged.graded == 1
    assert verify_finite_poset(forged) is False
    with pytest.raises(OperationDomainValidationError, match="canonical finite poset"):
        width(forged)


def test_consumers_reject_foreign_ordered_pair_models_as_domain_errors() -> None:
    class ForeignOrderedPair(OrderedPair):
        extra: str = "foreign"

    poset = _materialize(["a", "b"], [("a", "b")])
    forged_pairs = tuple(
        ForeignOrderedPair(lower=pair.lower, upper=pair.upper)
        for pair in poset.strict_order_pairs
    )
    forged = poset.model_copy(
        update={
            "strict_order_pairs": forged_pairs,
            "poset_digest": finite_poset_digest(
                elements=poset.elements,
                strict_order_pairs=forged_pairs,
                cover_relations=poset.cover_relations,
                incomparable_pairs=poset.incomparable_pairs,
                minimal_elements=poset.minimal_elements,
                maximal_elements=poset.maximal_elements,
                graded=poset.graded,
                ranks=poset.ranks,
            ),
        }
    )

    assert all(isinstance(pair, OrderedPair) for pair in forged.strict_order_pairs)
    assert any(type(pair) is not OrderedPair for pair in forged.strict_order_pairs)
    assert verify_finite_poset(forged) is False
    with pytest.raises(OperationDomainValidationError, match="canonical finite poset"):
        width(forged)


def test_consumers_reject_foreign_rank_models_as_domain_errors() -> None:
    class ForeignRank(ElementRank):
        extra: str = "foreign"

    poset = _materialize(["a", "b"], [("a", "b")])
    forged_ranks = tuple(
        ForeignRank(element=entry.element, rank=entry.rank)
        for entry in poset.ranks or ()
    )
    forged = poset.model_copy(
        update={
            "ranks": forged_ranks,
            "poset_digest": finite_poset_digest(
                elements=poset.elements,
                strict_order_pairs=poset.strict_order_pairs,
                cover_relations=poset.cover_relations,
                incomparable_pairs=poset.incomparable_pairs,
                minimal_elements=poset.minimal_elements,
                maximal_elements=poset.maximal_elements,
                graded=poset.graded,
                ranks=forged_ranks,
            ),
        }
    )

    assert verify_finite_poset(forged) is False
    with pytest.raises(OperationDomainValidationError, match="canonical finite poset"):
        width(forged)


def test_consumers_reject_foreign_incomparable_pair_models_as_domain_errors() -> None:
    class ForeignPair(IncomparablePair):
        extra: str = "foreign"

    poset = _materialize(["a", "b"], [])
    forged_pairs = tuple(
        ForeignPair(left=pair.left, right=pair.right)
        for pair in poset.incomparable_pairs
    )
    forged = poset.model_copy(
        update={
            "incomparable_pairs": forged_pairs,
            "poset_digest": finite_poset_digest(
                elements=poset.elements,
                strict_order_pairs=poset.strict_order_pairs,
                cover_relations=poset.cover_relations,
                incomparable_pairs=forged_pairs,
                minimal_elements=poset.minimal_elements,
                maximal_elements=poset.maximal_elements,
                graded=poset.graded,
                ranks=poset.ranks,
            ),
        }
    )

    assert verify_finite_poset(forged) is False
    with pytest.raises(OperationDomainValidationError, match="canonical finite poset"):
        width(forged)


def test_consumers_reject_list_incomparable_pair_containers_as_domain_errors() -> None:
    poset = _materialize(["a", "b"], [])
    forged = poset.model_copy(
        update={"incomparable_pairs": list(poset.incomparable_pairs)}
    )

    assert type(forged.incomparable_pairs) is list
    assert verify_finite_poset(forged) is False
    with pytest.raises(OperationDomainValidationError, match="canonical finite poset"):
        width(forged)


def test_consumers_reject_duck_typed_ordered_pair_carriers_as_domain_errors() -> None:
    class ForeignOrder:
        def __init__(self, lower: str, upper: str) -> None:
            self.lower = lower
            self.upper = upper

        def model_dump(self, mode: str = "json") -> dict[str, str]:
            return {"lower": self.lower, "upper": self.upper, "extra": "foreign"}

        def __eq__(self, other: object) -> bool:
            return (
                getattr(other, "lower", None) == self.lower
                and getattr(other, "upper", None) == self.upper
            )

    poset = _materialize(["a", "b"], [("a", "b")])
    forged_pairs = tuple(
        ForeignOrder(pair.lower, pair.upper) for pair in poset.strict_order_pairs
    )
    forged = poset.model_copy(
        update={
            "strict_order_pairs": forged_pairs,
            "cover_relations": forged_pairs,
            "poset_digest": finite_poset_digest(
                elements=poset.elements,
                strict_order_pairs=forged_pairs,  # type: ignore[arg-type]
                cover_relations=forged_pairs,  # type: ignore[arg-type]
                incomparable_pairs=poset.incomparable_pairs,
                minimal_elements=poset.minimal_elements,
                maximal_elements=poset.maximal_elements,
                graded=poset.graded,
                ranks=poset.ranks,
            ),
        }
    )
    assert verify_finite_poset(forged) is False
    with pytest.raises(OperationDomainValidationError, match="canonical finite poset"):
        width(forged)


def test_consumers_reject_equality_forging_digest_subclasses_as_domain_errors() -> None:
    class AlwaysEqualDigest(str):
        def __eq__(self, other: object) -> bool:
            return True

        def __hash__(self) -> int:
            return hash(str(self))

    poset = _materialize(["a", "b"], [("a", "b")])
    forged_digest = AlwaysEqualDigest("sha256:" + "0" * 64)
    forged = poset.model_copy(update={"poset_digest": forged_digest})

    assert type(forged.poset_digest) is not str
    assert forged.poset_digest == poset.poset_digest
    assert poset.poset_digest != "sha256:" + "0" * 64
    assert verify_finite_poset(forged) is False
    with pytest.raises(OperationDomainValidationError, match="canonical finite poset"):
        width(forged)


def test_consumers_reject_boolean_rank_claims_as_domain_errors() -> None:
    poset = _materialize(["a", "b"], [("a", "b")])
    forged_ranks = tuple(
        ElementRank.model_construct(element=entry.element, rank=False)
        if entry.rank == 0
        else entry
        for entry in poset.ranks or ()
    )
    forged = poset.model_copy(update={"ranks": forged_ranks})

    assert verify_finite_poset(forged) is False
    with pytest.raises(OperationDomainValidationError, match="canonical finite poset"):
        width(forged)


def test_comparable_pairs_require_complete_transitive_relation() -> None:
    with pytest.raises(OperationDomainValidationError) as exc:
        _materialize_request(
            FinitePosetRequest.model_validate(
                {
                    "elements": ["a", "b", "c"],
                    "relation": [
                        {"lower": "a", "upper": "b"},
                        {"lower": "b", "upper": "c"},
                    ],
                    "interpretation": "COMPARABLE_PAIRS",
                }
            )
        )
    _assert_operation_code(exc, "poset.comparable_pairs_complete")


def test_required_reflexive_policy_binds_the_entire_diagonal() -> None:
    with pytest.raises(OperationDomainValidationError) as exc:
        _materialize_request(
            FinitePosetRequest.model_validate(
                {
                    "elements": ["a", "b"],
                    "relation": [{"lower": "a", "upper": "a"}],
                    "interpretation": "COMPARABLE_PAIRS",
                    "reflexive_pairs": "REQUIRED",
                }
            )
        )
    _assert_operation_code(exc, "poset.required_reflexive_full_carrier")


def test_linear_extension_contract_has_a_separate_exponential_bound() -> None:
    antichain = _materialize([f"x{index}" for index in range(21)], [])
    request = LinearExtensionRequest(poset=antichain)
    with pytest.raises(OperationDomainValidationError) as exc:
        linear_extension_count(request.poset)
    _assert_operation_code(exc, "poset.linear_extension_size_bound")


def test_selected_mobius_scope_rejects_nonintervals_and_empty_selection() -> None:
    poset = _materialize(["a", "b"], [])
    with pytest.raises(ValidationError) as exc:
        MobiusFunctionRequest.model_validate(
            {"poset": poset, "scope": "SELECTED_INTERVALS"}
        )
    _assert_code(exc, "poset.selected_scope_nonempty")
    with pytest.raises(ValidationError) as exc:
        MobiusFunctionRequest.model_validate(
            {
                "poset": poset,
                "scope": "SELECTED_INTERVALS",
                "intervals": [{"lower": "a", "upper": "b"}],
            }
        )
    _assert_code(exc, "poset.interval_is_comparable")
