"""Exact contracts for invariant integral bilinear-form lattices."""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction
from threading import Event
from typing import Any

import pytest
from pydantic import ValidationError
from sympy import Matrix
from tests.fixtures.accounting import assert_charged_work_parity

from jacobian._exact import CanonicalRational
from jacobian._execution import (
    OperationExecutionCancelledError,
    request_cancellation,
)
from jacobian.canonical import CanonicalLimits, encode_strict_json
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.lattices._lattice_ops import saturate_lattice
from jacobian.math.lattices.invariant_forms import (
    EmbeddedRealNumberFieldActionGenerator,
    EmbeddedRealNumberFieldMatrixAction,
    RationalMatrixAction,
    compute_invariant_bilinear_form_lattice,
)
from jacobian.math.lattices.invariant_forms._models import (
    FormKind,
    IntegralBilinearForm,
    InvariantBilinearFormLattice,
    InvariantBilinearFormLatticeRequest,
    constraint_coefficient_count,
)
from jacobian.math.lattices.invariant_forms._tools import (
    INVARIANT_BILINEAR_FORM_LATTICE_OPERATION,
)
from jacobian.math.matrices.values import EmbeddedRealSimpleNumberFieldMatrix
from jacobian.math.number_theory.number_fields import (
    RealNumberFieldEmbedding,
    SimpleNumberFieldElement,
    SimpleNumberFieldPresentation,
    embeddings,
)


def _rational(value: int | Fraction) -> dict[str, str]:
    fraction = Fraction(value)
    return {
        "num": str(fraction.numerator),
        "den": str(fraction.denominator),
    }


def _field_element(
    presentation: SimpleNumberFieldPresentation,
    *coordinates: int | Fraction,
) -> SimpleNumberFieldElement:
    return SimpleNumberFieldElement.model_validate(
        {
            "presentation": presentation,
            "coefficients_ascending": [_rational(value) for value in coordinates],
        }
    )


def _quartic_action(
    entries: Sequence[Sequence[Sequence[int | Fraction]]],
) -> EmbeddedRealNumberFieldMatrixAction:
    presentation = SimpleNumberFieldPresentation(
        coefficients_descending=("1", "0", "0", "0", "-2")
    )
    embedding = embeddings(presentation).records[1].embedding
    assert isinstance(embedding, RealNumberFieldEmbedding)
    matrix = EmbeddedRealSimpleNumberFieldMatrix(
        embedding=embedding,
        entries=tuple(
            tuple(_field_element(presentation, *value) for value in row)
            for row in entries
        ),
    )
    return EmbeddedRealNumberFieldMatrixAction(
        coordinate_axis=("e1", "e2", "e3", "e4"),
        generators=(EmbeddedRealNumberFieldActionGenerator(label="J", matrix=matrix),),
    )


def _action(
    matrices: Sequence[tuple[str, Sequence[Sequence[int | Fraction]]]],
    *,
    axis: tuple[str, ...] | None = None,
) -> RationalMatrixAction:
    dimension = len(matrices[0][1]) if matrices else len(axis or ())
    coordinate_axis = axis or tuple(f"e{index + 1}" for index in range(dimension))
    return RationalMatrixAction.model_validate(
        {
            "coordinate_axis": list(coordinate_axis),
            "generators": [
                {
                    "label": label,
                    "matrix": {
                        "entries": [
                            [_rational(value) for value in row] for row in matrix
                        ]
                    },
                }
                for label, matrix in matrices
            ],
        }
    )


def _paper_action() -> RationalMatrixAction:
    return _action(
        [
            (
                "T1",
                [
                    [1, 0, -6, 2],
                    [0, -1, 1, 1],
                    [0, -1, 0, 1],
                    [0, 0, 0, 1],
                ],
            ),
            (
                "T2",
                [
                    [1, 6, 0, -3],
                    [0, 0, -1, 1],
                    [0, 1, 0, 0],
                    [0, 0, 0, 1],
                ],
            ),
        ],
        axis=("gamma", "u", "w", "delta"),
    )


def _integer_matrix(form: IntegralBilinearForm) -> Matrix:
    entries = form.matrix.entries
    return Matrix([[int(value) for value in row] for row in entries])


def _assert_every_basis_form_is_invariant(
    result: InvariantBilinearFormLattice,
) -> None:
    assert isinstance(result.action, RationalMatrixAction)
    generators = tuple(
        Matrix(
            [[entry.as_fraction() for entry in row] for row in generator.matrix.entries]
        )
        for generator in result.action.generators
    )
    for form in result.basis_forms:
        matrix = _integer_matrix(form)
        for generator in generators:
            assert generator.T * matrix * generator == matrix


def _flatten_forms(result: InvariantBilinearFormLattice) -> list[list[int]]:
    return [
        [int(value) for row in form.matrix.entries for value in row]
        for form in result.basis_forms
    ]


def test_source_paper_alternating_lattice_recovers_q_zero() -> None:
    # Bogomolov-Halle-Pazuki-Tanimoto, Lemma 2.8. Their matrices act on
    # column vectors in the ordered basis (gamma, u, w, delta), and Q0 is the
    # unique primitive invariant alternating form up to sign.
    action = _paper_action()
    request = InvariantBilinearFormLatticeRequest(action=action, kind="ALTERNATING")

    result = INVARIANT_BILINEAR_FORM_LATTICE_OPERATION.run(request)

    assert result.coefficient_dimension == 6
    assert result.constraint_rank == 5
    assert result.rank == 1
    assert result.basis_forms[0].matrix.entries == (
        ("0", "0", "0", "1"),
        ("0", "0", "6", "0"),
        ("0", "-6", "0", "0"),
        ("-1", "0", "0", "0"),
    )
    _assert_every_basis_form_is_invariant(result)


def test_source_paper_all_bilinear_forms_include_delta_square() -> None:
    action = _paper_action()

    result = compute_invariant_bilinear_form_lattice(action, "BILINEAR")

    assert result.rank == 2
    assert result.constraint_rank == 14
    assert result.basis_forms[0].matrix.entries == (
        ("0", "0", "0", "1"),
        ("0", "0", "6", "0"),
        ("0", "-6", "0", "0"),
        ("-1", "0", "0", "0"),
    )
    assert result.basis_forms[1].matrix.entries[-1][-1] == "1"
    _assert_every_basis_form_is_invariant(result)


@pytest.mark.parametrize(
    ("kind", "expected_dimension"),
    (("BILINEAR", 9), ("SYMMETRIC", 6), ("ALTERNATING", 3)),
)
def test_empty_generator_family_returns_full_coefficient_lattice(
    kind: FormKind, expected_dimension: int
) -> None:
    action = _action([], axis=("x", "y", "z"))

    result = compute_invariant_bilinear_form_lattice(action, kind)

    assert result.coefficient_dimension == expected_dimension
    assert result.constraint_rank == 0
    assert result.rank == expected_dimension
    _assert_every_basis_form_is_invariant(result)


def test_strict_json_action_can_omit_the_defaulted_generator_family() -> None:
    request = InvariantBilinearFormLatticeRequest.model_validate_json(
        encode_strict_json(
            {
                "action": {"coordinate_axis": ["e1"]},
                "kind": "ALTERNATING",
            }
        ),
        strict=True,
    )

    assert request.action.coordinate_axis == ("e1",)
    assert request.action.generators == ()


def test_rank_zero_lattice_retains_its_ambient_coefficient_dimension() -> None:
    action = _action([("twice", [[2, 0], [0, 2]])])

    result = compute_invariant_bilinear_form_lattice(action, "ALTERNATING")

    assert result.coefficient_dimension == 1
    assert result.constraint_rank == 1
    assert result.rank == 0
    assert result.basis_forms == ()


def test_quartic_complex_structure_has_rank_zero_alternating_lattice() -> None:
    zero = (0, 0, 0, 0)
    action = _quartic_action(
        (
            (zero, zero, (-1, 0, 0, 0), (0, -1, 0, 0)),
            (zero, zero, (0, 0, 0, -1), (0, 0, -1, 0)),
            ((-1, 0, -1, 0), (0, 1, 0, Fraction(1, 2)), zero, zero),
            ((0, 1, 0, 1), (-1, 0, Fraction(-1, 2), 0), zero, zero),
        )
    )

    result = compute_invariant_bilinear_form_lattice(action, "ALTERNATING")

    assert result.coefficient_dimension == 6
    assert result.constraint_rank == 6
    assert result.rank == 0
    assert result.basis_forms == ()


def test_algebraic_invariant_forms_obey_caller_cancellation() -> None:
    zero = (0, 0, 0, 0)
    one = (1, 0, 0, 0)
    action = _quartic_action(
        tuple(
            tuple(one if row == column else zero for column in range(4))
            for row in range(4)
        )
    )
    cancellation = Event()
    cancellation.set()
    with (
        request_cancellation(cancellation),
        pytest.raises(OperationExecutionCancelledError),
    ):
        compute_invariant_bilinear_form_lattice(action, "ALTERNATING")


def test_nonmonic_number_field_reduction_preserves_an_invariant_area_form() -> None:
    presentation = SimpleNumberFieldPresentation(
        coefficients_descending=("2", "0", "-1")
    )
    embedding = embeddings(presentation).records[1].embedding
    assert isinstance(embedding, RealNumberFieldEmbedding)
    zero = _field_element(presentation, 0, 0)
    alpha = _field_element(presentation, 0, 1)
    twice_alpha = _field_element(presentation, 0, 2)
    action = EmbeddedRealNumberFieldMatrixAction(
        coordinate_axis=("e1", "e2"),
        generators=(
            EmbeddedRealNumberFieldActionGenerator(
                label="determinant_one",
                matrix=EmbeddedRealSimpleNumberFieldMatrix(
                    embedding=embedding,
                    entries=((alpha, zero), (zero, twice_alpha)),
                ),
            ),
        ),
    )

    result = compute_invariant_bilinear_form_lattice(action, "ALTERNATING")

    assert result.rank == 1
    assert result.basis_forms[0].matrix.entries == (("0", "1"), ("-1", "0"))


def test_symmetric_constraint_path_keeps_exactly_the_diagonal_forms() -> None:
    action = _action([("reflection", [[-1, 0], [0, 1]])])

    result = compute_invariant_bilinear_form_lattice(action, "SYMMETRIC")

    assert result.coefficient_dimension == 3
    assert result.constraint_rank == 1
    assert result.rank == 2
    assert _flatten_forms(result) == [[1, 0, 0, 0], [0, 0, 0, 1]]
    _assert_every_basis_form_is_invariant(result)


def test_one_dimensional_alternating_space_has_zero_coefficient_dimension() -> None:
    action = _action([], axis=("x",))

    result = compute_invariant_bilinear_form_lattice(action, "ALTERNATING")

    assert result.coefficient_dimension == 0
    assert result.constraint_rank == 0
    assert result.rank == 0
    assert result.basis_forms == ()


def test_one_dimensional_alternating_action_does_not_parse_unused_scalars() -> None:
    action = _action([("nontrivial", [[Fraction(3, 2)]])])

    result = compute_invariant_bilinear_form_lattice(action, "ALTERNATING")

    assert result.coefficient_dimension == 0
    assert result.basis_forms == ()


def test_native_api_rejects_unknown_form_kind() -> None:
    """Native callers cannot route an unknown kind through alternating semantics."""
    action = _action([], axis=("x",))
    with pytest.raises(OperationDomainValidationError, match="kind must be"):
        compute_invariant_bilinear_form_lattice(action, "UNKNOWN")  # type: ignore[arg-type]


def test_raw_request_rejects_unknown_kind_before_nested_action_parsing() -> None:
    """Malformed kinds fail before expensive nested rational validation."""
    with pytest.raises(ValueError, match="kind must be"):
        InvariantBilinearFormLatticeRequest.model_validate(
            {
                "action": {
                    "coordinate_axis": ["x"],
                    "generators": [
                        {
                            "label": "g",
                            "matrix": {"entries": [[{"num": "1", "den": "2"}]]},
                        }
                    ],
                },
                "kind": "UNKNOWN",
            }
        )


def test_rational_action_regression_saturates_the_full_integer_kernel() -> None:
    action = _action([("A", [[Fraction(1, 2), 0], [2, 2]])])

    result = compute_invariant_bilinear_form_lattice(action, "BILINEAR")

    # FLINT's raw nullspace columns generate an index-three sublattice here.
    # Saturation followed by the repository's row-HNF convention gives the
    # complete integer kernel, including the primitive difference vector.
    assert _flatten_forms(result) == [[4, 0, 3, 0], [0, 1, -1, 0]]
    saturated, _, index = saturate_lattice(_flatten_forms(result))
    assert saturated == [
        [4, 3, 0, 0],
        [4, 2, 1, 0],
    ]
    assert index == 1
    _assert_every_basis_form_is_invariant(result)


def test_eight_axis_reflection_retains_its_small_primitive_kernel() -> None:
    dimension = 8
    reflection = [
        [(-1 if row == 0 else 1) if row == column else 0 for column in range(dimension)]
        for row in range(dimension)
    ]
    action = _action([("reflection", reflection)])

    result = compute_invariant_bilinear_form_lattice(action, "BILINEAR")

    assert result.coefficient_dimension == 64
    assert result.constraint_rank == 14
    assert result.rank == 50
    _assert_every_basis_form_is_invariant(result)


def test_generator_order_is_canonical_and_does_not_change_the_result() -> None:
    first = _action([("z", [[1, 1], [0, 1]]), ("a", [[-1, 0], [0, 1]])])
    second = _action([("a", [[-1, 0], [0, 1]]), ("z", [[1, 1], [0, 1]])])

    first_result = compute_invariant_bilinear_form_lattice(first, "BILINEAR")
    second_result = compute_invariant_bilinear_form_lattice(second, "BILINEAR")

    assert tuple(generator.label for generator in first.generators) == ("a", "z")
    assert first_result == second_result


def test_duplicate_generator_labels_are_rejected_after_order_normalization() -> None:
    with pytest.raises(ValidationError) as exc_info:
        _action([("A", [[1]]), ("A", [[-1]])])

    assert exc_info.value.errors()[0]["type"] == (
        "lattice.invariant_form.duplicate_generator_label"
    )


@pytest.mark.parametrize("extra_location", ("action", "generator"))
def test_deep_unknown_data_is_rejected_without_recursive_preprocessing(
    extra_location: str,
) -> None:
    nested: object = None
    for _ in range(1_500):
        nested = {"next": nested}
    generator: dict[str, object] = {
        "label": "identity",
        "matrix": {"entries": [[_rational(1)]]},
    }
    action: dict[str, object] = {
        "coordinate_axis": ["e1"],
        "generators": [generator],
    }
    if extra_location == "action":
        action["unknown"] = nested
    else:
        generator["unknown"] = nested

    with pytest.raises(ValidationError) as exc_info:
        RationalMatrixAction.model_validate(action)

    assert exc_info.value.errors(include_input=False)[0]["type"] == "extra_forbidden"


def test_unimodular_coordinate_change_transports_the_invariant_lattice() -> None:
    action = _action([("A", [[-1, 0], [0, 1]])])
    conjugated_action = _action([("A", [[-1, -2], [0, 1]])])
    result = compute_invariant_bilinear_form_lattice(action, "BILINEAR")
    conjugated_result = compute_invariant_bilinear_form_lattice(
        conjugated_action, "BILINEAR"
    )
    coordinate_change = Matrix([[1, 1], [0, 1]])
    transported = [
        coordinate_change.T * _integer_matrix(form) * coordinate_change
        for form in result.basis_forms
    ]
    transported_rows = [[int(value) for value in matrix] for matrix in transported]
    transported_saturation, _, _ = saturate_lattice(transported_rows)
    conjugated_saturation, _, _ = saturate_lattice(_flatten_forms(conjugated_result))

    assert transported_saturation == conjugated_saturation
    _assert_every_basis_form_is_invariant(conjugated_result)


def test_result_round_trip_retains_source_and_exact_empty_lattice() -> None:
    action = _action([("twice", [[2, 0], [0, 2]])])
    result = compute_invariant_bilinear_form_lattice(action, "SYMMETRIC")

    round_tripped = InvariantBilinearFormLattice.model_validate(
        result.model_dump(mode="json")
    )

    assert round_tripped == result
    assert round_tripped.action == action
    assert round_tripped.basis_forms == ()


def test_one_dimensional_hundred_digit_action_uses_derived_height_admission() -> None:
    action = _action([("large", [[10**100]])])

    result = compute_invariant_bilinear_form_lattice(action, "BILINEAR")

    assert result.coefficient_dimension == 1
    assert result.constraint_rank == 1
    assert result.rank == 0
    assert result.basis_forms == ()


def test_nine_axis_trivial_action_uses_exact_output_admission() -> None:
    action = _action([], axis=tuple(f"e{index}" for index in range(9)))

    result = compute_invariant_bilinear_form_lattice(action, "BILINEAR")

    assert result.coefficient_dimension == 81
    assert result.constraint_rank == 0
    assert result.rank == 81
    assert len(result.basis_forms) == 81
    encoded = encode_strict_json(result.model_dump(mode="json"))
    assert len(encoded) < CanonicalLimits().max_output_bytes
    assert InvariantBilinearFormLattice.model_validate_json(encoded) == result


def test_oversized_trivial_action_is_rejected_by_exact_output_admission() -> None:
    action = _action([], axis=tuple(f"e{index}" for index in range(128)))

    with pytest.raises(OperationDomainValidationError) as exc_info:
        compute_invariant_bilinear_form_lattice(action, "BILINEAR")

    assert exc_info.value.errors()[0]["type"] == (
        "lattice.invariant_form.budget_exceeded"
    )


def test_near_envelope_constraint_count_matches_realized_expansion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jacobian.math.lattices.invariant_forms._kernel as kernel

    dimension = 8
    generator_count = 16
    identity = [
        [int(row == column) for column in range(dimension)] for row in range(dimension)
    ]
    action = _action([(f"A{index:02d}", identity) for index in range(generator_count)])
    original = kernel._constraint_coefficient
    executed = 0

    def counted(*args: Any, **kwargs: Any) -> Any:
        nonlocal executed
        executed += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(kernel, "_constraint_coefficient", counted)

    admission = kernel._admit_invariant_bilinear_form_lattice(
        action,
        "BILINEAR",
    )
    plan = kernel._build_constraint_plan(
        action,
        "BILINEAR",
        admission=admission,
    )
    charged = constraint_coefficient_count(dimension, generator_count, "BILINEAR")

    assert_charged_work_parity(
        charged={"constraint_coefficient": charged},
        executed={"constraint_coefficient": executed},
    )
    assert charged == 65_536
    assert plan.constraints == ()


def test_source_height_is_coupled_to_constraint_expansion_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jacobian.math.lattices.invariant_forms._kernel as kernel

    dimension = 8
    large = {"num": "1" + "0" * 10_000, "den": "1"}
    zero = {"num": "0", "den": "1"}
    one = {"num": "1", "den": "1"}
    entries = [
        [
            large if row == column == 0 else one if row == column else zero
            for column in range(dimension)
        ]
        for row in range(dimension)
    ]
    action = RationalMatrixAction.model_validate(
        {
            "coordinate_axis": [f"e{index}" for index in range(dimension)],
            "generators": [
                {"label": f"A{index}", "matrix": {"entries": entries}}
                for index in range(3)
            ],
        }
    )

    def unexpected_coefficient(*_args: Any, **_kwargs: Any) -> Fraction:
        pytest.fail("over-height expansion reached coefficient construction")

    monkeypatch.setattr(kernel, "_constraint_coefficient", unexpected_coefficient)

    with pytest.raises(OperationDomainValidationError) as exc_info:
        compute_invariant_bilinear_form_lattice(action, "BILINEAR")

    assert exc_info.value.errors()[0]["type"] == (
        "lattice.invariant_form.budget_exceeded"
    )


def test_algebraic_degree_is_charged_before_embedding_recognition() -> None:
    presentation = SimpleNumberFieldPresentation(
        coefficients_descending=("1", "0", "0", "0", "0", "0", "0", "0", "1")
    )
    # x^8 + 1 has no real root, so reaching recognition would report an invalid
    # embedding. The degree-expanded structural ledger must reject first.
    embedding = RealNumberFieldEmbedding.model_validate(
        {
            "kind": "REAL",
            "presentation": presentation.model_dump(mode="json"),
            "root": {
                "polynomial": presentation.coefficients_descending,
                "real_root_index": 0,
            },
        }
    )
    zero = SimpleNumberFieldElement(
        presentation=presentation,
        coefficients_ascending=tuple(
            CanonicalRational.from_fraction(Fraction(0)) for _ in range(8)
        ),
    )
    dimension = 12
    matrix = EmbeddedRealSimpleNumberFieldMatrix(
        embedding=embedding,
        entries=tuple(tuple(zero for _ in range(dimension)) for _ in range(dimension)),
    )
    action = EmbeddedRealNumberFieldMatrixAction(
        coordinate_axis=tuple(f"e{index}" for index in range(dimension)),
        generators=(EmbeddedRealNumberFieldActionGenerator(label="A", matrix=matrix),),
    )

    with pytest.raises(OperationDomainValidationError) as exc_info:
        compute_invariant_bilinear_form_lattice(action, "ALTERNATING")

    assert exc_info.value.errors()[0]["type"] == (
        "lattice.invariant_form.budget_exceeded"
    )


def test_catalog_publishes_typed_operation_and_valid_example() -> None:
    operation = INVARIANT_BILINEAR_FORM_LATTICE_OPERATION

    assert operation is not None
    schema = operation.request_type.model_json_schema()
    assert schema["additionalProperties"] is False
    assert (
        "generator_count * axis_dimension^2"
        in schema["properties"]["action"]["description"]
    )
    request = operation.request_type.model_validate_json(
        encode_strict_json(operation.examples[0].input), strict=True
    )
    result = operation.run(request)
    assert (
        operation.result_type.model_validate(result.model_dump(mode="json")) == result
    )
