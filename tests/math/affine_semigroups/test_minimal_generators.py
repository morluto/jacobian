from __future__ import annotations

import itertools
import json
from collections.abc import Callable
from fractions import Fraction
from typing import cast

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.affine_semigroups import (
    AffineConfiguration,
    PositiveAffineSemigroup,
    minimal_generators,
)
from jacobian.math.affine_semigroups.atoms import (
    AffineMinimalGenerators,
    AffineMinimalGeneratorsRequest,
)


def _semigroup(vectors: tuple[tuple[int, ...], ...]) -> PositiveAffineSemigroup:
    rows = len(vectors[0])
    configuration = AffineConfiguration(
        row_labels=tuple(f"r{i}" for i in range(rows)),
        generator_labels=tuple(f"g{i}" for i in range(len(vectors))),
        entries=tuple(tuple(vector[row] for vector in vectors) for row in range(rows)),
    )
    return PositiveAffineSemigroup(
        configuration=configuration,
        grading=tuple(
            CanonicalRational.from_fraction(Fraction(1)) for _ in range(rows)
        ),
    )


def _brute_factorizations(
    vectors: tuple[tuple[int, ...], ...], target: tuple[int, ...]
) -> tuple[tuple[int, ...], ...]:
    maxima = tuple(
        max(
            (target[r] // vector[r] for r in range(len(target)) if vector[r] > 0),
            default=0,
        )
        for vector in vectors
    )
    return tuple(
        counts
        for counts in itertools.product(*(range(bound + 1) for bound in maxima))
        if tuple(
            sum(counts[j] * vectors[j][r] for j in range(len(vectors)))
            for r in range(len(target))
        )
        == target
    )


@pytest.mark.parametrize(
    "vectors",
    [
        ((1,), (1,), (2,), (3,), (4,), (6,)),
        ((2, 0), (0, 2), (1, 1), (2, 2), (3, 1)),
        ((2, 1, 0), (0, 2, 1), (1, 1, 1), (4, 4, 2)),
    ],
)
def test_minimal_generators_match_independent_complete_fiber_oracle(
    vectors: tuple[tuple[int, ...], ...],
) -> None:
    result = minimal_generators(_semigroup(vectors))
    vector_set = tuple(sorted(set(vectors)))
    expected_atoms = tuple(
        target
        for target in vector_set
        if not any(sum(row) >= 2 for row in _brute_factorizations(vectors, target))
    )
    assert result.atoms.configuration.columns_vectors == expected_atoms
    for source, factors in zip(vectors, result.source_factorizations, strict=True):
        assert (
            tuple(
                sum(
                    factors[i] * expected_atoms[i][row]
                    for i in range(len(expected_atoms))
                )
                for row in range(len(source))
            )
            == source
        )


def test_result_round_trip_retains_minimal_atom_parent_and_factorizations() -> None:
    result = minimal_generators(_semigroup(((1,), (1,), (2,))))
    restored = AffineMinimalGenerators.model_validate_json(result.model_dump_json())

    assert restored == result
    assert restored.atoms.configuration.generator_labels == ("g0",)
    assert restored.source_factorizations == ((1,), (1,), (2,))
    for source, factors in zip(
        restored.source.configuration.columns_vectors,
        restored.source_factorizations,
        strict=True,
    ):
        assert (
            tuple(
                sum(
                    factors[i] * restored.atoms.configuration.columns_vectors[i][row]
                    for i in range(len(factors))
                )
                for row in range(len(source))
            )
            == source
        )


def test_result_rejects_a_source_factorization_on_the_wrong_atom_axis() -> None:
    result = minimal_generators(_semigroup(((1,), (2,))))
    payload = result.model_dump(mode="python")
    payload["source_factorizations"] = ((0,), (1,))

    # Structural deserialization does not replay the computed reconstruction.
    restored = AffineMinimalGenerators.model_validate(payload)
    assert restored.source_factorizations == ((0,), (1,))


def test_minimal_generators_support_positive_graded_configurations_with_negative_entries() -> (
    None
):
    configuration = AffineConfiguration(
        row_labels=("x", "y"),
        generator_labels=("right", "left", "unit", "twice_unit"),
        entries=((1, -1, 0, 0), (0, 2, 1, 2)),
    )
    semigroup = PositiveAffineSemigroup(
        configuration=configuration,
        grading=(
            CanonicalRational.from_fraction(Fraction(1)),
            CanonicalRational.from_fraction(Fraction(1)),
        ),
    )

    result = minimal_generators(semigroup)

    assert result.atoms.configuration.columns_vectors == ((-1, 2), (0, 1), (1, 0))
    assert result.source_factorizations[3] == (0, 2, 0)


def test_aggregate_work_is_admitted_before_factorization_enumeration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jacobian.math.affine_semigroups.atoms as atoms_module

    monkeypatch.setattr(atoms_module, "MAX_AFFINE_ATOM_WORK", 0)
    monkeypatch.setattr(
        atoms_module,
        "_enumerate_fiber",
        lambda *_args: pytest.fail("fiber enumerated before aggregate admission"),
    )

    with pytest.raises(OperationResourceAdmissionError, match="aggregate work"):
        minimal_generators(_semigroup(((1,), (2,))))


def test_catalog_example_executes_with_the_published_operation_contract() -> None:
    from jacobian.catalog.builtins import BUILTIN_TOOLS

    tool = next(
        tool
        for tool in BUILTIN_TOOLS
        if tool.operation_id == "affine_semigroup.minimal_generators.compute"
    )
    example = tool.examples[0]
    request_type = cast(type[AffineMinimalGeneratorsRequest], tool.request_type)
    request = request_type.model_validate_json(json.dumps(example.input))

    run = cast(
        Callable[[AffineMinimalGeneratorsRequest], AffineMinimalGenerators],
        tool.run,
    )
    result = run(request)

    assert result.atoms.configuration.columns_vectors == ((1, 0),)
    assert result.source_factorizations == ((1,), (1,), (2,))
