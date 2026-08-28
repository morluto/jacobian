"""Correctness and contract tests for bounded combinatorics on words."""

from __future__ import annotations

import itertools
import json
import random
from collections.abc import Iterator
from contextlib import contextmanager

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.logic.languages.words import (
    FiniteWord,
    ProlongableSubstitution,
    Substitution,
    SubstitutionDependencyGraph,
    WordMorphism,
    apply_morphism,
    compose_morphisms,
    conjugates,
    factor_occurrences,
    factors_of_length,
    fixed_point_prefix,
    incidence_matrix,
    parikh_vector,
    periods,
    prefix_function,
    primitive_root,
    substitution_dependency_graph,
    substitution_primitivity_profile,
)
from jacobian.math.logic.languages.words import _tools as word_operations
from jacobian.math.logic.languages.words._models import (
    FactorsLengthRequest,
    FactorsLengthResult,
    IncidenceMatrixRequest,
    IncidenceMatrixResult,
    PeriodsRequest,
    PeriodsResult,
    SubstitutionDependencyGraphRequest,
    SubstitutionDependencyGraphResult,
    SubstitutionFixedPointPrefixRequest,
    SubstitutionFixedPointPrefixResult,
    SubstitutionPrimitivityProfileRequest,
)
from jacobian.math.logic.languages.words._tools import (
    TOOLS,
    compute_factors_length,
    compute_incidence_matrix,
    compute_periods,
    compute_substitution_dependency_graph,
    compute_substitution_fixed_point_prefix,
    compute_substitution_primitivity_profile,
)


@contextmanager
def _raises_code(code: str) -> Iterator[None]:
    with pytest.raises(ValidationError) as error:
        yield
    assert error.value.errors()[0]["type"] == code


def _word(letters: str, alphabet: tuple[str, ...] = ("a", "b")) -> FiniteWord:
    return FiniteWord(alphabet=alphabet, letters=tuple(letters))


def _substitution(
    images: tuple[tuple[str, ...], ...],
    alphabet: tuple[str, ...] = ("0", "1"),
) -> Substitution:
    return Substitution(
        morphism=WordMorphism(
            source_alphabet=alphabet,
            target_alphabet=alphabet,
            images=images,
        )
    )


def test_public_catalog_surface_is_the_audited_operations() -> None:
    assert tuple(tool.operation_id for tool in TOOLS) == (
        "word.factors.length.compute",
        "word.periods.compute",
        "word_morphism.incidence_matrix.compute",
        "substitution.dependency_graph.compute",
        "substitution.primitivity_profile.compute",
        "substitution.fixed_point_prefix.compute",
    )


def test_narrowed_scalar_symbol_contract_rejects_lone_surrogates() -> None:
    periods = next(
        tool for tool in TOOLS if tool.operation_id == "word.periods.compute"
    )
    surrogate = json.loads('{"alphabet": ["\\ud800"], "letters": ["\\ud800"]}')
    with pytest.raises(ValidationError):
        periods.request_type.model_validate(surrogate)


def test_factor_result_is_complete_and_bound_to_the_request() -> None:
    request = FactorsLengthRequest(word=_word("abaab"), factor_length=2)
    result = compute_factors_length(request)
    assert result.factors == (("a", "b"), ("b", "a"), ("a", "a"))
    assert result.occurrences == ((0, 3), (1,), (2,))
    assert result.multiplicities == (2, 1, 1)
    assert result.first_occurrence == (0, 1, 2)
    assert result.distinct_count == 3

    payload = result.model_dump()
    payload["factors"] = (("b", "b"), *payload["factors"][1:])
    supplied = FactorsLengthResult.model_validate(payload)
    assert supplied.factors[0] == ("b", "b")


def test_empty_factor_occurs_at_every_boundary() -> None:
    result = compute_factors_length(
        FactorsLengthRequest(word=_word("aa", ("a",)), factor_length=0)
    )
    assert result.factors == ((),)
    assert result.occurrences == ((0, 1, 2),)


def test_factor_length_is_validated_at_operation_time() -> None:
    with pytest.raises(ValueError, match="factor length"):
        compute_factors_length(
            FactorsLengthRequest(word=_word("aa", ("a",)), factor_length=3)
        )


def test_periods_distinguish_overlap_period_from_proper_power() -> None:
    repeated = compute_periods(PeriodsRequest(word=_word("ababab")))
    assert repeated.periods == (2, 4, 6)
    assert repeated.least_period == 2
    assert repeated.is_primitive is False

    bordered_but_primitive = compute_periods(PeriodsRequest(word=_word("ababa")))
    assert bordered_but_primitive.periods == (2, 4, 5)
    assert bordered_but_primitive.least_period == 2
    assert bordered_but_primitive.is_primitive is True


def test_empty_period_convention_and_result_binding() -> None:
    result = compute_periods(PeriodsRequest(word=_word("", ("a",))))
    assert result.periods == ()
    assert result.least_period == 0
    assert result.is_primitive is False

    payload = result.model_dump()
    payload["is_primitive"] = True
    supplied = PeriodsResult.model_validate(payload)
    assert supplied.is_primitive is True


@pytest.mark.parametrize(
    ("operation", "operation_request", "kernel_name"),
    (
        (
            compute_factors_length,
            FactorsLengthRequest(word=_word("abaab"), factor_length=2),
            "factors_of_length",
        ),
        (
            compute_periods,
            PeriodsRequest(word=_word("ababab")),
            "periods",
        ),
    ),
)
def test_trusted_word_profile_producers_run_the_kernel_once(
    monkeypatch: pytest.MonkeyPatch,
    operation: object,
    operation_request: object,
    kernel_name: str,
) -> None:
    original = getattr(word_operations, kernel_name)
    calls = 0

    def counted(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(word_operations, kernel_name, counted)

    assert callable(operation)
    operation(operation_request)

    assert calls == 1


def test_fibonacci_incidence_matrix_and_binding() -> None:
    morphism = WordMorphism(
        source_alphabet=("a", "b"),
        target_alphabet=("a", "b"),
        images=(("a", "b"), ("a",)),
    )
    result = compute_incidence_matrix(IncidenceMatrixRequest(morphism=morphism))
    assert result.matrix == ((1, 1), (1, 0))
    assert result.orientation == "ROWS_TARGET_COLUMNS_SOURCE"

    payload = result.model_dump()
    payload["matrix"] = ((1, 0), (1, 1))
    supplied = IncidenceMatrixResult.model_validate(payload)
    assert supplied.matrix == ((1, 0), (1, 1))


def test_substitution_requires_an_endomorphism() -> None:
    with _raises_code("word.substitution_not_endomorphism"):
        Substitution(
            morphism=WordMorphism(
                source_alphabet=("a",),
                target_alphabet=("b",),
                images=(("b",),),
            )
        )


def test_fibonacci_dependency_graph_retains_positions_and_source() -> None:
    fibonacci = _substitution((("0", "1"), ("0",)))
    result = compute_substitution_dependency_graph(
        SubstitutionDependencyGraphRequest(substitution=fibonacci)
    )
    assert tuple(
        (edge.source, edge.target, edge.multiplicity, edge.positions)
        for edge in result.graph.edges
    ) == (
        ("0", "0", 1, (0,)),
        ("0", "1", 1, (1,)),
        ("1", "0", 1, (0,)),
    )

    payload = result.model_dump()
    payload["graph"]["edges"] = payload["graph"]["edges"][:-1]
    supplied = SubstitutionDependencyGraphResult.model_validate(payload)
    assert len(supplied.graph.edges) == 2


def test_dependency_graph_output_budget_is_admitted_before_enumeration() -> None:
    at_limit = _substitution((("0",) * 5_000, ("1",) * 5_000))
    request = SubstitutionDependencyGraphRequest(substitution=at_limit)
    assert sum(len(image) for image in request.substitution.morphism.images) == 10_000
    result = compute_substitution_dependency_graph(request)
    assert tuple(edge.multiplicity for edge in result.graph.edges) == (5_000, 5_000)

    above_limit = _substitution((("0",) * 5_001, ("1",) * 5_000))
    request = SubstitutionDependencyGraphRequest(substitution=above_limit)
    with pytest.raises(OperationDomainValidationError, match="aggregate bound"):
        compute_substitution_dependency_graph(request)
    graph = SubstitutionDependencyGraph(substitution=above_limit, edges=())
    with pytest.raises(ValueError, match="aggregate bound"):
        substitution_dependency_graph(above_limit)
    profile_request = SubstitutionPrimitivityProfileRequest(dependency_graph=graph)
    with pytest.raises(OperationDomainValidationError, match="aggregate bound"):
        compute_substitution_primitivity_profile(profile_request)
    with pytest.raises(ValueError, match="aggregate bound"):
        substitution_primitivity_profile(graph)


def test_primitivity_profile_distinguishes_positive_reducible_and_periodic() -> None:
    fibonacci_graph = substitution_dependency_graph(_substitution((("0", "1"), ("0",))))
    fibonacci = compute_substitution_primitivity_profile(
        SubstitutionPrimitivityProfileRequest(dependency_graph=fibonacci_graph)
    )
    assert "incidence_matrix" not in fibonacci.model_dump()
    assert fibonacci.strongly_connected_components == (("0", "1"),)
    assert fibonacci.primitive is True
    assert fibonacci.least_positive_power == 2
    assert fibonacci.exponent_upper_bound == 2
    assert fibonacci.obstruction == "NONE"

    reducible_graph = substitution_dependency_graph(_substitution((("0", "1"), ("1",))))
    reducible = substitution_primitivity_profile(reducible_graph)
    assert reducible.strongly_connected_components == (("0",), ("1",))
    assert reducible.irreducible is False
    assert reducible.aperiodic is None
    assert reducible.primitive is False
    assert reducible.obstruction == "REDUCIBLE_DEPENDENCY_GRAPH"

    periodic_graph = substitution_dependency_graph(_substitution((("1",), ("0",))))
    periodic = substitution_primitivity_profile(periodic_graph)
    assert periodic.strongly_connected_components == (("0", "1"),)
    assert periodic.irreducible is True
    assert periodic.aperiodic is False
    assert periodic.primitive is False
    assert periodic.obstruction == "PERIODIC_DEPENDENCY_GRAPH"


def test_fixed_point_prefix_uses_the_least_sufficient_iterate() -> None:
    fibonacci = ProlongableSubstitution(
        substitution=_substitution((("0", "1"), ("0",))), seed="0"
    )
    result = compute_substitution_fixed_point_prefix(
        SubstitutionFixedPointPrefixRequest(source=fibonacci, prefix_length=8)
    )
    assert result.prefix.letters == tuple("01001010")
    assert result.least_iterate_depth == 4
    assert result.retained_prefix_lengths == (1, 2, 3, 5, 8)

    reapplied = apply_morphism(fibonacci.substitution.morphism, result.prefix)
    assert reapplied.letters[:8] == result.prefix.letters

    payload = result.model_dump()
    payload["prefix"]["letters"] = (*payload["prefix"]["letters"][:-1], "1")
    supplied = SubstitutionFixedPointPrefixResult.model_validate(payload)
    assert supplied.prefix.letters[-1] == "1"


def test_fixed_point_prefix_empty_and_length_boundaries() -> None:
    doubling = ProlongableSubstitution(
        substitution=_substitution((("0", "0"),), ("0",)), seed="0"
    )
    empty = fixed_point_prefix(doubling, 0)
    assert empty.prefix.letters == ()
    assert empty.least_iterate_depth == 0
    assert empty.retained_prefix_lengths == (0,)

    boundary = compute_substitution_fixed_point_prefix(
        SubstitutionFixedPointPrefixRequest(source=doubling, prefix_length=500)
    )
    assert len(boundary.prefix.letters) == 500
    with _raises_code("less_than_equal"):
        SubstitutionFixedPointPrefixRequest(source=doubling, prefix_length=501)
    with pytest.raises(ValueError, match="prefix length"):
        fixed_point_prefix(doubling, 501)


def test_fixed_point_source_and_result_envelopes_cover_exact_boundaries() -> None:
    source_boundary = ProlongableSubstitution(
        substitution=_substitution(
            (("0",) * 10_000, ("1",) * 10_000),
        ),
        seed="0",
    )
    assert (
        sum(len(image) for image in source_boundary.substitution.morphism.images)
        == 20_000
    )
    assert fixed_point_prefix(source_boundary, 1).prefix.letters == ("0",)
    accepted = compute_substitution_fixed_point_prefix(
        SubstitutionFixedPointPrefixRequest(
            source=source_boundary,
            prefix_length=1,
        )
    )
    assert accepted.prefix.letters == ("0",)

    over_limit_source = {
        "substitution": {
            "morphism": {
                "source_alphabet": ["0", "1", "2", "3"],
                "target_alphabet": ["0", "1", "2", "3"],
                # The seed suffix eventually erases.  The public source limit
                # must reject this payload before prolongability analyzes that.
                "images": [
                    ["0", *("1",) * 9_999],
                    [],
                    ["2"] * 10_000,
                    ["3"],
                ],
            }
        },
        "seed": "0",
    }
    with _raises_code("word.fixed_point_source_occurrence_bound"):
        ProlongableSubstitution.model_validate(over_limit_source)
    with _raises_code("word.fixed_point_source_occurrence_bound"):
        SubstitutionFixedPointPrefixRequest.model_validate(
            {"source": over_limit_source, "prefix_length": 1}
        )

    accepted_symbol = "x" * 45
    byte_boundary = ProlongableSubstitution(
        substitution=_substitution(((accepted_symbol,) * 10_000,), (accepted_symbol,)),
        seed=accepted_symbol,
    )
    byte_result = compute_substitution_fixed_point_prefix(
        SubstitutionFixedPointPrefixRequest(
            source=byte_boundary,
            prefix_length=500,
        )
    )
    assert len(byte_result.prefix.letters) == 500

    rejected_symbol = "x" * 46
    byte_above = ProlongableSubstitution(
        substitution=_substitution(((rejected_symbol,) * 10_000,), (rejected_symbol,)),
        seed=rejected_symbol,
    )
    with pytest.raises(OperationDomainValidationError, match="byte bound"):
        compute_substitution_fixed_point_prefix(
            SubstitutionFixedPointPrefixRequest(source=byte_above, prefix_length=500)
        )


def test_fixed_point_generation_caps_the_intermediate_prefix() -> None:
    source = ProlongableSubstitution(
        substitution=_substitution(
            (("a",) + ("b",) * 498, ("b",) * 10_000),
            ("a", "b"),
        ),
        seed="a",
    )
    uncapped_second_length = 499 + 498 * 10_000
    assert uncapped_second_length > 4_000_000

    result = compute_substitution_fixed_point_prefix(
        SubstitutionFixedPointPrefixRequest(source=source, prefix_length=500)
    )
    assert result.retained_prefix_lengths == (1, 499, 500)
    assert len(result.prefix.letters) == 500


def test_prolongability_rejects_mortal_nongrowing_and_wrong_seed_images() -> None:
    with _raises_code("word.seed_suffix_eventually_erases"):
        ProlongableSubstitution(substitution=_substitution((("0", "1"), ())), seed="0")
    with _raises_code("word.seed_image_not_growing"):
        ProlongableSubstitution(substitution=_substitution((("0",), ("1",))), seed="0")
    with _raises_code("word.seed_image_not_prolongable"):
        ProlongableSubstitution(
            substitution=_substitution((("1", "0"), ("1",))), seed="0"
        )


def test_prolongability_allows_erasing_outside_the_growing_seed_orbit() -> None:
    source = ProlongableSubstitution(
        substitution=_substitution((("0", "0"), ())), seed="0"
    )
    assert fixed_point_prefix(source, 8).prefix.letters == ("0",) * 8


def test_alphabet_relabelling_preserves_primitivity_and_prefix_transport() -> None:
    binary = _substitution((("0", "1"), ("0",)))
    relabelled = _substitution((("b", "a"), ("b",)), ("b", "a"))
    binary_profile = substitution_primitivity_profile(
        substitution_dependency_graph(binary)
    )
    relabelled_profile = substitution_primitivity_profile(
        substitution_dependency_graph(relabelled)
    )
    assert relabelled_profile.primitive is binary_profile.primitive
    assert (
        relabelled_profile.least_positive_power == binary_profile.least_positive_power
    )

    binary_prefix = fixed_point_prefix(
        ProlongableSubstitution(substitution=binary, seed="0"), 20
    ).prefix.letters
    relabelled_prefix = fixed_point_prefix(
        ProlongableSubstitution(substitution=relabelled, seed="b"), 20
    ).prefix.letters
    assert relabelled_prefix == tuple(
        {"0": "b", "1": "a"}[letter] for letter in binary_prefix
    )


def test_all_small_dependency_graphs_match_integer_power_oracle() -> None:
    for order in range(1, 4):
        alphabet = tuple(str(index) for index in range(order))
        bound = 1 if order == 1 else (order - 1) ** 2 + 1
        for cells in itertools.product((0, 1), repeat=order * order):
            rows = tuple(
                tuple(cells[source * order + target] for target in range(order))
                for source in range(order)
            )
            substitution = _substitution(
                tuple(
                    tuple(
                        alphabet[target]
                        for target in range(order)
                        if rows[source][target]
                    )
                    for source in range(order)
                ),
                alphabet,
            )
            profile = substitution_primitivity_profile(
                substitution_dependency_graph(substitution)
            )

            power = rows
            expected_exponent = None
            for exponent in range(1, bound + 1):
                if all(entry > 0 for row in power for entry in row):
                    expected_exponent = exponent
                    break
                power = tuple(
                    tuple(
                        sum(power[i][k] * rows[k][j] for k in range(order))
                        for j in range(order)
                    )
                    for i in range(order)
                )
            assert profile.least_positive_power == expected_exponent
            assert profile.primitive is (expected_exponent is not None)


def test_primitivity_alphabet_boundary_is_complete() -> None:
    alphabet = tuple(f"x{index}" for index in range(50))
    identity = _substitution(tuple((symbol,) for symbol in alphabet), alphabet)
    result = compute_substitution_primitivity_profile(
        SubstitutionPrimitivityProfileRequest(
            dependency_graph=substitution_dependency_graph(identity)
        )
    )
    assert result.exponent_upper_bound == 2_402
    assert result.primitive is False
    assert result.obstruction == "REDUCIBLE_DEPENDENCY_GRAPH"

    oversized = tuple(f"x{index}" for index in range(51))
    with _raises_code("too_long"):
        WordMorphism(
            source_alphabet=oversized,
            target_alphabet=oversized,
            images=tuple((symbol,) for symbol in oversized),
        )


def test_one_image_mutation_changes_graph_primitivity_and_prefix() -> None:
    fibonacci = _substitution((("0", "1"), ("0",)))
    mutated = _substitution((("0", "0"), ("0",)))
    fibonacci_graph = substitution_dependency_graph(fibonacci)
    mutated_graph = substitution_dependency_graph(mutated)
    assert mutated_graph != fibonacci_graph
    assert substitution_primitivity_profile(fibonacci_graph).primitive is True
    assert substitution_primitivity_profile(mutated_graph).primitive is False

    fibonacci_prefix = fixed_point_prefix(
        ProlongableSubstitution(substitution=fibonacci, seed="0"), 8
    )
    mutated_prefix = fixed_point_prefix(
        ProlongableSubstitution(substitution=mutated, seed="0"), 8
    )
    assert fibonacci_prefix.prefix != mutated_prefix.prefix


def test_native_word_operations_are_exact_and_use_declared_order() -> None:
    word = _word("aaa", ("a",))
    assert factor_occurrences(word, ("a", "a")) == (0, 1)
    assert factor_occurrences(word, ()) == (0, 1, 2, 3)
    assert primitive_root(_word("abcabc", ("a", "b", "c"))) == (
        ("a", "b", "c"),
        2,
    )
    assert conjugates(_word("abb", ("b", "a"))) == (
        ("b", "b", "a"),
        ("b", "a", "b"),
        ("a", "b", "b"),
    )
    assert parikh_vector(_word("abaab")) == (3, 2)
    assert prefix_function(_word("aabaab")) == (0, 1, 0, 1, 2, 3)


def test_native_morphism_application_and_composition() -> None:
    fibonacci = WordMorphism(
        source_alphabet=("a", "b"),
        target_alphabet=("a", "b"),
        images=(("a", "b"), ("a",)),
    )
    swap = WordMorphism(
        source_alphabet=("a", "b"),
        target_alphabet=("x", "y"),
        images=(("y",), ("x",)),
    )
    assert apply_morphism(fibonacci, _word("ab")).letters == ("a", "b", "a")
    composed = compose_morphisms(fibonacci, swap)
    assert composed.images == (("y", "x"), ("y",))
    assert incidence_matrix(composed) == ((1, 0), (1, 1))


def test_value_models_reject_ambiguous_or_unbounded_inputs() -> None:
    with _raises_code("word.alphabet_symbols_not_distinct"):
        FiniteWord(alphabet=("a", "a"), letters=("a",))
    with _raises_code("word.word_letter_outside_alphabet"):
        FiniteWord(alphabet=("a",), letters=("b",))

    expanding = WordMorphism(
        source_alphabet=("a",),
        target_alphabet=("a",),
        images=(("a",) * 2,),
    )
    with pytest.raises(ValueError, match="output exceeds"):
        apply_morphism(
            expanding,
            FiniteWord(alphabet=("a",), letters=("a",) * 500),
        )


def test_empty_alphabet_carries_exactly_the_empty_word_through_json() -> None:
    """The empty word has a canonical empty ambient alphabet."""
    word = FiniteWord.model_validate_json(
        '{"alphabet": [], "letters": []}', strict=True
    )

    assert word.alphabet == ()
    assert word.letters == ()
    assert FiniteWord.model_validate(word.model_dump(mode="json")) == word


@pytest.mark.parametrize("symbol", ["\ud800", "\udfff", "a\ud800b"])
def test_symbols_admit_only_unicode_scalar_strings(symbol: str) -> None:
    with pytest.raises(ValidationError):
        FiniteWord(alphabet=(symbol,), letters=())
    with pytest.raises(ValidationError):
        WordMorphism(
            source_alphabet=("a",), target_alphabet=(symbol,), images=(("a",),)
        )


def test_random_words_match_independent_factor_and_period_oracles() -> None:
    random_source = random.Random(1966)
    for length in range(9):
        for _ in range(40):
            letters = tuple(random_source.choice(("a", "b")) for _ in range(length))
            word = FiniteWord(alphabet=("a", "b"), letters=letters)
            for factor_length in range(length + 1):
                analysis = factors_of_length(word, factor_length)
                windows = tuple(
                    letters[index : index + factor_length]
                    for index in range(length - factor_length + 1)
                )
                expected_factors = tuple(dict.fromkeys(windows))
                expected_positions = tuple(
                    tuple(
                        index
                        for index, window in enumerate(windows)
                        if window == factor
                    )
                    for factor in expected_factors
                )
                assert analysis.factors == expected_factors
                assert analysis.occurrences == expected_positions

            period_analysis = periods(word)
            expected_periods = tuple(
                period
                for period in range(1, length + 1)
                if letters[:-period] == letters[period:]
            )
            proper_power = any(
                length % root_length == 0
                and letters[:root_length] * (length // root_length) == letters
                for root_length in range(1, length)
            )
            assert period_analysis.periods == expected_periods
            assert period_analysis.primitive is (length > 0 and not proper_power)


def test_incidence_matrix_matches_independent_count_oracle() -> None:
    alphabet = ("a", "b")
    images = tuple(itertools.product(alphabet, repeat=2))
    for left in images:
        for right in images:
            morphism = WordMorphism(
                source_alphabet=alphabet,
                target_alphabet=alphabet,
                images=(left, right),
            )
            assert incidence_matrix(morphism) == tuple(
                tuple(image.count(target) for image in (left, right))
                for target in alphabet
            )
