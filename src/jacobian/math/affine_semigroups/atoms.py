"""Exact minimal generators of bounded positive affine semigroups."""

from __future__ import annotations

from fractions import Fraction

from pydantic import model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.affine_semigroups.semigroup import (
    AffineConfiguration,
    PositiveAffineSemigroup,
    _admit_fiber,
    _admit_semigroup,
    _enumerate_fiber,
)

MAX_AFFINE_ATOM_WORK = 1_000_000
MAX_AFFINE_ATOM_OUTPUT_BYTES = 2_000_000


class AffineMinimalGenerators(StrictModel):
    """The unique atom set and factorizations of source generators in it."""

    source: PositiveAffineSemigroup
    atoms: PositiveAffineSemigroup
    source_factorizations: tuple[tuple[int, ...], ...]

    @model_validator(mode="after")
    def _structure(self) -> AffineMinimalGenerators:
        source_config = self.source.configuration
        atom_config = self.atoms.configuration
        if (
            atom_config.row_labels != source_config.row_labels
            or self.atoms.grading != self.source.grading
        ):
            raise PydanticCustomError(
                "affine_semigroup.atom_parent",
                "atom semigroup must retain the source ambient rows and grading",
            )
        atoms = atom_config.columns_vectors
        if atoms != tuple(sorted(set(atoms))):
            raise PydanticCustomError(
                "affine_semigroup.atom_order",
                "atom vectors must be sorted and unique",
            )
        if len(self.source_factorizations) != source_config.columns:
            raise PydanticCustomError(
                "affine_semigroup.atom_factorization_count",
                "one atom factorization is required per source generator",
            )
        if any(
            len(factorization) != len(atoms) or any(x < 0 for x in factorization)
            for factorization in self.source_factorizations
        ):
            raise PydanticCustomError(
                "affine_semigroup.atom_factorization_axis",
                "source factorizations must be nonnegative on the atom axis",
            )
        return self


class AffineMinimalGeneratorsRequest(StrictModel):
    semigroup: PositiveAffineSemigroup


def minimal_generators(
    semigroup: PositiveAffineSemigroup,
) -> AffineMinimalGenerators:
    """Return the irreducible generators, with every input column factored in them.

    An input generator is reducible exactly when its complete fiber contains a
    factorization with at least two factors. Positive grading makes each fiber
    finite; all candidate boxes and aggregate work are admitted before any
    factorization enumeration.
    """
    semigroup = _admit_semigroup(semigroup)
    configuration = semigroup.configuration
    distinct_vectors = tuple(sorted(set(configuration.columns_vectors)))
    plans: dict[
        tuple[int, ...], tuple[tuple[int, ...], tuple[Fraction, ...], Fraction]
    ] = {}
    total_work = 0
    for target in distinct_vectors:
        grades, target_grade, maxima = _admit_fiber(semigroup, target)
        work = 1
        for maximum in maxima:
            work *= maximum + 1
        total_work += work
        plans[target] = (maxima, grades, target_grade)
    # A second pass factors the original columns in the atom configuration.
    # Its coefficient box is no larger than the already-admitted source box.
    if 2 * total_work > MAX_AFFINE_ATOM_WORK:
        raise OperationResourceAdmissionError(
            location=("semigroup",),
            code="affine_semigroup.atom_work",
            message=(
                "minimal-generator classification and transport exceed the "
                f"{MAX_AFFINE_ATOM_WORK}-state aggregate work envelope"
            ),
        )
    _admit_atom_output(semigroup)

    fibers = {
        target: _enumerate_fiber(semigroup, target, grades, target_grade, maxima)
        for target, (maxima, grades, target_grade) in plans.items()
    }
    atom_vectors = tuple(
        target
        for target in distinct_vectors
        if not any(sum(row) >= 2 for row in fibers[target])
    )
    first_labels: dict[tuple[int, ...], str] = {}
    for label, vector in zip(
        configuration.generator_labels, configuration.columns_vectors, strict=True
    ):
        first_labels.setdefault(vector, label)
    atom_configuration = AffineConfiguration(
        row_labels=configuration.row_labels,
        generator_labels=tuple(first_labels[vector] for vector in atom_vectors),
        entries=tuple(
            tuple(vector[row] for vector in atom_vectors)
            for row in range(configuration.rows)
        ),
    )
    atom_semigroup = PositiveAffineSemigroup(
        configuration=atom_configuration,
        grading=semigroup.grading,
    )
    source_grades = {
        vector: sum(
            (
                semigroup.grading[row].as_fraction() * vector[row]
                for row in range(configuration.rows)
            ),
            Fraction(0),
        )
        for vector in distinct_vectors
    }
    atom_grades = tuple(source_grades[vector] for vector in atom_vectors)
    atom_factorizations: dict[tuple[int, ...], tuple[int, ...]] = {}
    for target in distinct_vectors:
        target_grade = plans[target][2]
        maxima = tuple(int(target_grade // grade) for grade in atom_grades)
        row = next(
            iter(
                _enumerate_fiber(
                    atom_semigroup, target, atom_grades, target_grade, maxima
                )
            ),
            None,
        )
        if row is None:
            # Since atoms are a subset of source generators, this indicates an
            # internal contract failure rather than semigroup nonmembership.
            raise ValueError("source generator did not factor through its atoms")
        atom_factorizations[target] = row
    source_factorizations = tuple(
        atom_factorizations[vector] for vector in configuration.columns_vectors
    )
    return AffineMinimalGenerators(
        source=semigroup,
        atoms=atom_semigroup,
        source_factorizations=source_factorizations,
    )


def _admit_atom_output(semigroup: PositiveAffineSemigroup) -> None:
    """Bound output from source dimensions before factorization enumeration."""
    config = semigroup.configuration
    labels = (*config.row_labels, *config.generator_labels)
    # Python character count is a cheap lower bound on UTF-8/JSON output size.
    # Reject before encoding so a forged native value cannot force an
    # unbounded temporary allocation on the refusal path.
    if any(len(label) > MAX_AFFINE_ATOM_OUTPUT_BYTES for label in labels):
        raise OperationResourceAdmissionError(
            location=("semigroup",),
            code="affine_semigroup.atom_output",
            message="minimal-generator result exceeds its output-byte envelope",
        )
    label_characters = sum(
        len(label.encode("utf-8"))
        + sum(ord(character) < 32 for character in label) * 5
        + sum(character in ('"', "\\") for character in label)
        for label in labels
    )
    grading_bits = sum(
        value.num.bit_length() + value.den.bit_length() for value in semigroup.grading
    )
    # The atom parent has no more labels, rows, columns, or matrix cells than
    # the source parent. JSON escapes control characters as six ASCII bytes;
    # input matrix entries have at most eight decimal digits. Factor entries
    # fit the already-admitted 50,000-state fiber box.
    one_parent = (
        label_characters
        + 64 * (config.rows + config.columns)
        + 16 * config.rows * config.columns
        + 2 * ((grading_bits + 2) // 3 + config.rows)
        + 512
    )
    result_bound = 2 * one_parent + 12 * config.columns * config.columns
    if result_bound > MAX_AFFINE_ATOM_OUTPUT_BYTES:
        raise OperationResourceAdmissionError(
            location=("semigroup",),
            code="affine_semigroup.atom_output",
            message="minimal-generator result exceeds its output-byte envelope",
        )


__all__ = [
    "AffineMinimalGenerators",
    "AffineMinimalGeneratorsRequest",
    "minimal_generators",
]
