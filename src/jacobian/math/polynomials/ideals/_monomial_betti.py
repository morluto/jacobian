"""Exact bounded Betti profiles of monomial ideals over ``QQ``.

For a multidegree ``m`` in the lcm lattice, the Gasharov--Peeva--Welker
formula identifies ``beta_(i,m)(I)`` with reduced homology in degree ``i-1``
of the crosscut complex whose faces have lcm strictly below ``m``.  FLINT
computes the ranks of the integral simplicial boundary matrices exactly.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from flint import fmpz_mat

from jacobian.math.polynomials.values import RationalPolynomialIdeal

ExponentVector = tuple[int, ...]
MultigradedBettiData = tuple[int, ExponentVector, int]
GradedBettiData = tuple[int, int, int]


@dataclass(frozen=True)
class LcmLatticeHomologyData:
    """One lcm-lattice element and its complete crosscut rank profile."""

    multidegree: ExponentVector
    face_counts: tuple[int, ...]
    boundary_ranks: tuple[int, ...]
    reduced_homology_dimensions: tuple[int, ...]


@dataclass(frozen=True)
class MonomialBettiKernelResult:
    """The complete output of one admitted lcm-lattice computation."""

    lattice_homology: tuple[LcmLatticeHomologyData, ...]
    multigraded_betti: tuple[MultigradedBettiData, ...]
    graded_betti: tuple[GradedBettiData, ...]
    regularity: int
    has_linear_resolution: bool


def _lcm(left: ExponentVector, right: ExponentVector) -> ExponentVector:
    return tuple(max(a, b) for a, b in zip(left, right, strict=True))


def _strictly_divides(left: ExponentVector, right: ExponentVector) -> bool:
    return left != right and all(a <= b for a, b in zip(left, right, strict=True))


def _subset_lcms(generators: tuple[ExponentVector, ...]) -> tuple[ExponentVector, ...]:
    """Return every subset lcm, indexed by the subset bit mask."""

    empty = (0,) * len(generators[0])
    lcms = [empty] * (1 << len(generators))
    for mask in range(1, len(lcms)):
        least_bit = mask & -mask
        index = least_bit.bit_length() - 1
        lcms[mask] = _lcm(lcms[mask ^ least_bit], generators[index])
    return tuple(lcms)


def _boundary_rank(
    source_faces: tuple[int, ...],
    target_faces: tuple[int, ...],
    *,
    augmented: bool = False,
) -> int:
    """Return the exact rank of one reduced simplicial boundary over ``QQ``."""

    if not source_faces or not target_faces:
        return 0
    if augmented:
        return 1
    target_index = {face: index for index, face in enumerate(target_faces)}
    matrix = [[0] * len(source_faces) for _ in target_faces]
    for column, face in enumerate(source_faces):
        remaining = face
        position = 0
        while remaining:
            bit = remaining & -remaining
            target = face ^ bit
            matrix[target_index[target]][column] = -1 if position % 2 else 1
            remaining ^= bit
            position += 1
    return int(fmpz_mat(matrix).rank())


def _crosscut_homology(
    faces: tuple[int, ...], generator_count: int
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Return face counts, boundary ranks, and reduced homology dimensions."""

    by_size = tuple(
        tuple(face for face in faces if face.bit_count() == size)
        for size in range(generator_count + 1)
    )
    ranks = tuple(
        _boundary_rank(by_size[size], by_size[size - 1], augmented=size == 1)
        for size in range(1, generator_count + 1)
    )
    dimensions = [len(by_size[0]) - ranks[0]]
    for size in range(1, generator_count):
        dimension = len(by_size[size]) - ranks[size - 1] - ranks[size]
        if dimension < 0:
            raise RuntimeError("crosscut boundary ranks are internally inconsistent")
        dimensions.append(dimension)
    return tuple(map(len, by_size)), ranks, tuple(dimensions)


def compute_monomial_betti_kernel(
    ideal: RationalPolynomialIdeal,
) -> MonomialBettiKernelResult:
    """Compute one complete source-bound Betti profile without replay."""

    generators = tuple(
        generator.polynomial.terms[0].exponents for generator in ideal.generators
    )
    subset_lcms = _subset_lcms(generators)
    lattice = tuple(sorted(set(subset_lcms[1:])))
    lattice_homology: list[LcmLatticeHomologyData] = []
    multigraded: list[MultigradedBettiData] = []

    for multidegree in lattice:
        faces = tuple(
            mask
            for mask, face_lcm in enumerate(subset_lcms)
            if _strictly_divides(face_lcm, multidegree)
        )
        face_counts, boundary_ranks, homology = _crosscut_homology(
            faces, len(generators)
        )
        lattice_homology.append(
            LcmLatticeHomologyData(
                multidegree=multidegree,
                face_counts=face_counts,
                boundary_ranks=boundary_ranks,
                reduced_homology_dimensions=homology,
            )
        )
        multigraded.extend(
            (homological_degree, multidegree, dimension)
            for homological_degree, dimension in enumerate(homology)
            if dimension
        )

    multigraded_tuple = tuple(sorted(multigraded))
    totals: defaultdict[tuple[int, int], int] = defaultdict(int)
    for homological_degree, multidegree, value in multigraded_tuple:
        totals[homological_degree, sum(multidegree)] += value
    graded = tuple(
        (homological_degree, internal_degree, value)
        for (homological_degree, internal_degree), value in sorted(totals.items())
    )
    regularity = max(
        internal_degree - homological_degree
        for homological_degree, internal_degree, _ in graded
    )
    generator_degrees = {sum(generator) for generator in generators}
    has_linear_resolution = len(generator_degrees) == 1 and all(
        internal_degree == homological_degree + regularity
        for homological_degree, internal_degree, _ in graded
    )
    return MonomialBettiKernelResult(
        lattice_homology=tuple(lattice_homology),
        multigraded_betti=multigraded_tuple,
        graded_betti=graded,
        regularity=regularity,
        has_linear_resolution=has_linear_resolution,
    )


__all__ = [
    "LcmLatticeHomologyData",
    "MonomialBettiKernelResult",
    "compute_monomial_betti_kernel",
]
