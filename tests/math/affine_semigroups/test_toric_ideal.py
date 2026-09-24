from __future__ import annotations

import json
from collections import deque
from itertools import product

import pytest
from pydantic import ValidationError

from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.affine_semigroups import graver as graver_module
from jacobian.math.affine_semigroups import toric_ideal
from jacobian.math.affine_semigroups.graver_models import (
    IntegerConfigurationToricIdealRequest,
)
from jacobian.math.affine_semigroups.semigroup import AffineConfiguration
from jacobian.math.polynomials.ideals.operations import (
    groebner_basis,
    verify_groebner_basis,
)
from jacobian.math.polynomials.values import RationalPolynomialIdeal


def _configuration(weights: tuple[int, ...], labels: tuple[str, ...] | None = None):
    names = labels or tuple(f"x{i + 1}" for i in range(len(weights)))
    return AffineConfiguration(
        row_labels=("degree",),
        generator_labels=names,
        entries=(weights,),
    )


def _exponent_coefficients(ideal: RationalPolynomialIdeal):
    return tuple(
        tuple(
            (term.exponents, term.coefficient.as_fraction())
            for term in generator.polynomial.terms
        )
        for generator in ideal.generators
    )


def test_toric_ideal_uses_the_labelled_axis_and_exact_graver_binomials() -> None:
    ideal = toric_ideal(_configuration((1, 1, 2), ("u", "v", "w")))

    assert ideal.variables == ("u", "v", "w")
    assert _exponent_coefficients(ideal) == (
        (((0, 2, 0), 1), ((0, 0, 1), -1)),
        (((1, 0, 0), 1), ((0, 1, 0), -1)),
        (((1, 1, 0), 1), ((0, 0, 1), -1)),
        (((2, 0, 0), 1), ((0, 0, 1), -1)),
    )
    for generator in ideal.generators:
        weighted_degrees = {
            sum(
                weight * exponent
                for weight, exponent in zip((1, 1, 2), term.exponents, strict=True)
            )
            for term in generator.polynomial.terms
        }
        assert len(weighted_degrees) == 1


def test_toric_ideal_composes_with_polynomial_groebner_operation() -> None:
    ideal = toric_ideal(_configuration((1, 1, 2), ("u", "v", "w")))

    basis = groebner_basis(ideal, "lex")

    assert basis.basis.variables == ideal.variables
    assert verify_groebner_basis(basis)


def test_graver_moves_connect_every_tiny_admitted_fiber() -> None:
    weights = (1, 1, 2)
    ideal = toric_ideal(_configuration(weights))
    # Read the monomial exponent difference independently from each returned
    # binomial; the move direction is immaterial because edges are undirected.
    moves = tuple(
        tuple(
            left - right
            for left, right in zip(
                generator.polynomial.terms[0].exponents,
                generator.polynomial.terms[1].exponents,
                strict=True,
            )
        )
        for generator in ideal.generators
        if len(generator.polynomial.terms) == 2
    )
    for target in range(9):
        fiber = tuple(
            vector
            for vector in product(range(target + 1), repeat=len(weights))
            if sum(
                weight * value for weight, value in zip(weights, vector, strict=True)
            )
            == target
        )
        if not fiber:
            continue
        reached = {fiber[0]}
        pending = deque([fiber[0]])
        while pending:
            current = pending.popleft()
            for move in moves:
                for sign in (-1, 1):
                    neighbor = tuple(
                        value + sign * delta
                        for value, delta in zip(current, move, strict=True)
                    )
                    if neighbor in fiber and neighbor not in reached:
                        reached.add(neighbor)
                        pending.append(neighbor)
        assert reached == set(fiber)


def test_zero_row_and_one_positive_weight_have_exact_kernel_ideals() -> None:
    zero_row = toric_ideal(_configuration((0, 0), ("a", "b")))
    assert _exponent_coefficients(zero_row) == (
        (((0, 1), 1), ((0, 0), -1)),
        (((1, 0), 1), ((0, 0), -1)),
    )

    zero_kernel = toric_ideal(_configuration((99_999_999,), ("q",)))
    assert zero_kernel.variables == ("q",)
    assert len(zero_kernel.generators) == 1
    assert zero_kernel.generators[0].polynomial.terms == ()


def test_two_column_exact_kernel_avoids_coordinate_box_expansion() -> None:
    ideal = toric_ideal(_configuration((32_768, 32_767), ("p", "q")))

    assert _exponent_coefficients(ideal) == ((((32_767, 0), 1), ((0, 32_768), -1)),)
    scaled = toric_ideal(_configuration((99_999_999, 99_999_999), ("r", "s")))
    assert _exponent_coefficients(scaled) == ((((1, 0), 1), ((0, 1), -1)),)


def test_catalog_example_runs_and_json_roundtrips() -> None:
    tool = next(
        item
        for item in BUILTIN_TOOLS
        if item.operation_id == "integer_configuration.toric_ideal.compute"
    )
    request = IntegerConfigurationToricIdealRequest.model_validate_json(
        json.dumps(dict(tool.examples[0].input))
    )
    result = tool.run(request)

    assert isinstance(result, RationalPolynomialIdeal)
    decoded = RationalPolynomialIdeal.model_validate_json(result.model_dump_json())
    assert decoded == result


def test_toric_request_schema_publishes_its_representable_input_bounds() -> None:
    configuration = IntegerConfigurationToricIdealRequest.model_json_schema()[
        "properties"
    ]["configuration"]["properties"]
    assert configuration["row_labels"]["maxItems"] == 1
    assert configuration["generator_labels"]["maxItems"] == 5
    assert configuration["generator_labels"]["items"]["maxLength"] == 32
    assert configuration["entries"]["maxItems"] == 1
    assert configuration["entries"]["items"]["maxItems"] == 5
    assert configuration["entries"]["items"]["items"]["maxLength"] == 8


def test_domain_limits_and_polynomial_axis_are_explicit() -> None:
    with pytest.raises(OperationDomainValidationError, match="nonnegative"):
        toric_ideal(_configuration((1, -1)))
    with pytest.raises(OperationDomainValidationError, match="one-row"):
        toric_ideal(
            AffineConfiguration(
                row_labels=("degree", "weight"),
                generator_labels=("x", "y"),
                entries=((1, 2), (3, 4)),
            )
        )
    with pytest.raises(OperationDomainValidationError, match="polynomial variable"):
        toric_ideal(_configuration((1, 2), ("x", "not-a-variable")))
    with pytest.raises(ValidationError):
        IntegerConfigurationToricIdealRequest.model_validate(
            {
                "configuration": {
                    "row_labels": ["d"],
                    "generator_labels": [],
                    "entries": [[]],
                }
            }
        )


def test_generator_candidate_envelope_rejects_before_graver_enumeration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_called(_configuration):
        raise AssertionError("Graver enumeration ran before the 64-generator preflight")

    monkeypatch.setattr(graver_module, "_enumerate_graver_vectors", fail_if_called)
    with pytest.raises(OperationResourceAdmissionError, match="more than 64"):
        toric_ideal(_configuration((1, 2, 3)))
