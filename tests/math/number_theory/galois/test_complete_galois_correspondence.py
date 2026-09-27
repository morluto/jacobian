from __future__ import annotations

from fractions import Fraction
from itertools import combinations

from jacobian.catalog.models import MathTool
from jacobian.math.number_theory.galois._models import (
    GaloisCorrespondenceRequest,
    GaloisCorrespondenceResult,
    QQSplittingField,
)
from jacobian.math.number_theory.galois._tools import TOOLS
from jacobian.math.number_theory.galois.operations import (
    galois_correspondence,
    splitting_field,
)
from jacobian.math.number_theory.number_fields.values import (
    SimpleNumberFieldPresentation,
)
from jacobian.math.polynomials.values import RationalPolynomial


def _polynomial(coefficients: tuple[int, ...]) -> RationalPolynomial:
    return RationalPolynomial.model_validate(
        {
            "variables": ["x"],
            "polynomial": {
                "terms": [
                    {
                        "coefficient": {"num": value, "den": 1},
                        "exponents": [power],
                    }
                    for power, value in reversed(tuple(enumerate(coefficients)))
                    if value
                ]
            },
        }
    )


def _field(coefficients: tuple[int, ...]) -> QQSplittingField:
    return splitting_field(_polynomial(coefficients)).field


def _quadratic_subgroup_oracle() -> tuple[frozenset[int], ...]:
    """Enumerate C2 subgroups independently by sign-map multiplication."""
    elements = (0, 1)  # alpha -> alpha or -alpha
    identity = 0
    subgroups = []
    for size in range(1, len(elements) + 1):
        for candidate in combinations(elements, size):
            subset = frozenset(candidate)
            if identity in subset and all(
                (left ^ right) in subset for left in subset for right in subset
            ):
                subgroups.append(subset)
    return tuple(
        sorted(subgroups, key=lambda group: (-len(group), tuple(sorted(group))))
    )


def _fixed_vector_dimension(subgroup: frozenset[int]) -> int:
    """Solve sigma(a+b*sqrt(2)) = a+b*sqrt(2) over Q directly."""
    return 1 if 1 in subgroup else 2


def _map_sqrt2_element(
    sign: int, coordinates: tuple[Fraction, Fraction]
) -> tuple[Fraction, Fraction]:
    constant, radical = coordinates
    return constant, radical if sign == 0 else -radical


def test_quadratic_correspondence_matches_exhaustive_exact_field_oracle() -> None:
    field = _field((-2, 0, 1))
    result = galois_correspondence(field)
    expected_subgroups = _quadratic_subgroup_oracle()

    assert len(result.pairs) == 2 == len(expected_subgroups)
    assert tuple(pair.subgroup_order for pair in result.pairs) == (2, 1)
    assert tuple(pair.fixed_field_degree for pair in result.pairs) == (1, 2)
    assert tuple(pair.subgroup_index for pair in result.pairs) == (1, 2)
    assert tuple(pair.relative_field_degree for pair in result.pairs) == (2, 1)
    assert result.normal_subgroup_labels == ("H0", "H1")

    for pair, signs in zip(result.pairs, expected_subgroups, strict=True):
        assert pair.subgroup_label in result.subgroup_inclusion_poset.elements
        assert pair.field_label in result.intermediate_field_inclusion_poset.elements
        assert pair.normal
        assert pair.subgroup == pair.stabilizer
        observed_signs = set()
        for automorphism in pair.subgroup.elements:
            generator_image = tuple(
                coefficient.as_fraction()
                for coefficient in automorphism.basis_images[1].coefficients_ascending
            )
            if generator_image == (Fraction(0), Fraction(1)):
                observed_signs.add(0)
            elif generator_image == (Fraction(0), Fraction(-1)):
                observed_signs.add(1)
            else:
                raise AssertionError("automorphism does not map sqrt(2) to a root")
        assert frozenset(observed_signs) == signs
        assert _fixed_vector_dimension(signs) == pair.fixed_field_degree

        inclusion = pair.inclusion
        assert inclusion.target == field.extension
        assert inclusion.source.degree == pair.fixed_field_degree
        image = tuple(
            coefficient.as_fraction()
            for coefficient in inclusion.generator_image.coefficients_ascending
        )
        # The fixed subfield is Q for the full group, and L for the trivial
        # subgroup. The canonical degree-one generator maps to zero.
        if pair.fixed_field_degree == 1:
            assert inclusion.source == SimpleNumberFieldPresentation(
                coefficients_descending=(1, 0)
            )
            assert image == (Fraction(0), Fraction(0))
            assert all(_map_sqrt2_element(sign, image) == image for sign in (0, 1))
        else:
            assert inclusion.source == field.extension
            assert image == (Fraction(0), Fraction(1))
            stabilizing_signs = {
                sign for sign in (0, 1) if _map_sqrt2_element(sign, image) == image
            }
            assert stabilizing_signs == signs

    # H1 < H0 while F0 < F1: the two typed posets explicitly encode
    # inclusion reversal under the pairwise map.
    assert tuple(
        (pair.lower, pair.upper)
        for pair in result.subgroup_inclusion_poset.strict_order_pairs
    ) == (("H1", "H0"),)
    assert tuple(
        (pair.lower, pair.upper)
        for pair in result.intermediate_field_inclusion_poset.strict_order_pairs
    ) == (("F0", "F1"),)
    assert (
        GaloisCorrespondenceResult.model_validate_json(result.model_dump_json())
        == result
    )


def test_split_quadratic_has_one_subgroup_and_one_embedded_intermediate_field() -> None:
    field = _field((-1, 0, 1))
    result = galois_correspondence(field)

    assert field.degree == 1
    assert len(result.pairs) == 1
    (pair,) = result.pairs
    assert pair.subgroup_order == pair.subgroup_index == 1
    assert pair.fixed_field_degree == pair.relative_field_degree == 1
    assert pair.subgroup.elements[0].root_permutation == (0, 1)
    assert pair.inclusion.source == pair.inclusion.target == field.extension
    assert pair.stabilizer == pair.subgroup
    assert result.subgroup_inclusion_poset.elements == ("H0",)
    assert result.intermediate_field_inclusion_poset.elements == ("F0",)
    assert not result.subgroup_inclusion_poset.strict_order_pairs
    assert not result.intermediate_field_inclusion_poset.strict_order_pairs


def test_correspondence_operation_is_published_with_exact_typed_example() -> None:
    tool = next(
        tool
        for tool in TOOLS
        if tool.operation_id == "number_field.galois_correspondence.compute"
    )
    assert isinstance(tool, MathTool)
    assert tool.request_type is GaloisCorrespondenceRequest
    assert tool.result_type is GaloisCorrespondenceResult
    assert len(tool.examples) == 1
