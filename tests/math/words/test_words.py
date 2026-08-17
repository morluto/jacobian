"""Tests for combinatorics on words operations."""

import pytest
from pydantic import ValidationError

from jacobian.math.words._models import (
    ConjugatesRequest,
    FactorOccurrencesRequest,
    FactorsLengthRequest,
    IncidenceMatrixRequest,
    MorphismApplyRequest,
    MorphismComposeRequest,
    ParikhRequest,
    PeriodsRequest,
    PrefixFunctionRequest,
    PrimitiveRootRequest,
)
from jacobian.math.words._operations import (
    apply_morphism,
    compose_morphisms,
    compute_conjugates,
    compute_factor_occurrences,
    compute_factors_length,
    compute_incidence_matrix,
    compute_parikh_vector,
    compute_periods,
    compute_prefix_function,
    compute_primitive_root,
)


class TestFactorsLength:
    def test_abaab(self):
        req = FactorsLengthRequest(alphabet=("a", "b"), word=("a", "b", "a", "a", "b"), factor_length=2)
        result = compute_factors_length(req)
        assert result.distinct_count == 3
        factors = {f for f in result.factors}
        assert ("a", "b") in factors
        assert ("b", "a") in factors
        assert ("a", "a") in factors

    def test_zero_length(self):
        req = FactorsLengthRequest(alphabet=("a",), word=("a", "a"), factor_length=0)
        result = compute_factors_length(req)
        assert result.distinct_count == 1

    def test_full_length(self):
        req = FactorsLengthRequest(alphabet=("a", "b"), word=("a", "b"), factor_length=2)
        result = compute_factors_length(req)
        assert result.distinct_count == 1
        assert result.factors[0] == ("a", "b")

    def test_too_long(self):
        req = FactorsLengthRequest(alphabet=("a",), word=("a", "a"), factor_length=3)
        result = compute_factors_length(req)
        assert result.distinct_count == 0


class TestFactorOccurrences:
    def test_simple(self):
        req = FactorOccurrencesRequest(
            alphabet=("a", "b"),
            word=("a", "b", "a", "a", "b", "a", "b"),
            pattern=("a", "b"),
        )
        result = compute_factor_occurrences(req)
        assert result.count == 3
        assert result.occurrences == (0, 3, 5)

    def test_overlapping(self):
        req = FactorOccurrencesRequest(
            alphabet=("a", "a"),
            word=("a", "a", "a"),
            pattern=("a", "a"),
        )
        result = compute_factor_occurrences(req)
        assert result.count == 2

    def test_no_occurrence(self):
        req = FactorOccurrencesRequest(
            alphabet=("a", "b"),
            word=("a", "a", "a"),
            pattern=("b",),
        )
        result = compute_factor_occurrences(req)
        assert result.count == 0

    def test_empty_pattern(self):
        req = FactorOccurrencesRequest(
            alphabet=("a", "b"),
            word=("a", "a", "a"),
            pattern=(),
        )
        result = compute_factor_occurrences(req)
        assert result.count == 4


class TestPeriods:
    def test_ababab(self):
        req = PeriodsRequest(alphabet=("a", "b"), word=("a", "b", "a", "b", "a", "b"))
        result = compute_periods(req)
        assert 2 in result.periods
        assert result.least_period == 2
        assert result.is_primitive is False

    def test_primitive(self):
        req = PeriodsRequest(alphabet=("a", "b", "c"), word=("a", "b", "c"))
        result = compute_periods(req)
        assert result.least_period == 3
        assert result.is_primitive is True


class TestPrimitiveRoot:
    def test_repeated(self):
        req = PrimitiveRootRequest(
            alphabet=("a", "b", "c"),
            word=("a", "b", "c", "a", "b", "c"),
        )
        result = compute_primitive_root(req)
        assert result.root == ("a", "b", "c")
        assert result.exponent == 2

    def test_primitive(self):
        req = PrimitiveRootRequest(
            alphabet=("a", "b"),
            word=("a", "b", "a"),
        )
        result = compute_primitive_root(req)
        assert result.exponent == 1
        assert result.root == ("a", "b", "a")

    def test_empty(self):
        req = PrimitiveRootRequest(alphabet=("a",), word=())
        result = compute_primitive_root(req)
        assert result.root == ()
        assert result.exponent == 1


class TestConjugates:
    def test_baab(self):
        req = ConjugatesRequest(alphabet=("a", "b"), word=("b", "a", "a", "b"))
        result = compute_conjugates(req)
        assert len(result.conjugates) == 4
        assert result.least_lexicographic == ("a", "a", "b", "b")

    def test_empty(self):
        req = ConjugatesRequest(alphabet=("a",), word=())
        result = compute_conjugates(req)
        assert result.conjugates == ((),)


class TestParikh:
    def test_simple(self):
        req = ParikhRequest(alphabet=("a", "b", "c"), word=("a", "b", "a", "a", "b"))
        result = compute_parikh_vector(req)
        assert result.parikh_vector == (3, 2, 0)
        assert result.length == 5
        assert result.support == ("a", "b")


class TestPrefixFunction:
    def test_aabaab(self):
        req = PrefixFunctionRequest(alphabet=("a", "b"), word=("a", "a", "b", "a", "a", "b"))
        result = compute_prefix_function(req)
        assert result.prefix_function == (0, 1, 0, 1, 2, 3)

    def test_empty(self):
        req = PrefixFunctionRequest(alphabet=("a",), word=())
        result = compute_prefix_function(req)
        assert result.prefix_function == ()


class TestMorphismApply:
    def test_fibonacci(self):
        req = MorphismApplyRequest(
            source_alphabet=("a", "b"),
            target_alphabet=("a", "b"),
            images=(("a", "b"), ("a",)),
            word=("a", "b"),
        )
        result = apply_morphism(req)
        assert result.image == ("a", "b", "a")
        assert result.length == 3

    def test_empty_word(self):
        req = MorphismApplyRequest(
            source_alphabet=("a", "b"),
            target_alphabet=("a", "b"),
            images=(("a",), ("b",)),
            word=(),
        )
        result = apply_morphism(req)
        assert result.image == ()
        assert result.length == 0


class TestMorphismCompose:
    def test_compose(self):
        req = MorphismComposeRequest(
            source_alphabet=("a", "b"),
            middle_alphabet=("a", "b"),
            target_alphabet=("a", "b"),
            sigma_images=(("a", "b"), ("b",)),
            tau_images=(("b",), ("a", "a")),
        )
        result = compose_morphisms(req)
        assert result.images[0] == ("b", "a", "a")
        assert result.images[1] == ("a", "a")


class TestIncidenceMatrix:
    def test_fibonacci(self):
        req = IncidenceMatrixRequest(
            source_alphabet=("a", "b"),
            target_alphabet=("a", "b"),
            images=(("a", "b"), ("a",)),
        )
        result = compute_incidence_matrix(req)
        assert result.matrix == ((1, 1), (1, 0))


class TestValidation:
    def test_invalid_letter(self):
        with pytest.raises(ValidationError, match="not in the alphabet"):
            FactorsLengthRequest(
                alphabet=("a", "b"), word=("a", "c"), factor_length=1
            )

    def test_invalid_image_letter(self):
        with pytest.raises(ValidationError, match="target alphabet"):
            MorphismApplyRequest(
                source_alphabet=("a",),
                target_alphabet=("a",),
                images=(("a", "b"),),
                word=("a",),
            )

    def test_mismatched_images_count(self):
        with pytest.raises(ValidationError, match="one entry per source"):
            MorphismApplyRequest(
                source_alphabet=("a", "b"),
                target_alphabet=("a",),
                images=(("a",),),
                word=("a",),
            )
