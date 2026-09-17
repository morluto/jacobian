"""Cup products, cohomology rings, and induced maps (#1716).

The torus/wedge pair shares Betti numbers (1, 2, 1) over GF(2) but has
different rings: the torus carries nonzero H^1 x H^1 products while every
positive-degree wedge product vanishes.  Functoriality and cup commutation
of pullbacks are checked as exact matrix identities.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import MathTool, OperationDomainValidationError
from jacobian.math.topology._models import FiniteSimplicialComplex
from jacobian.math.topology.cohomology.operations._models import (
    CohomologyRingRequest,
    CohomologyRingResult,
    CupProductRequest,
    CupProductResult,
    InducedCohomologyMapRequest,
    InducedCohomologyMapResult,
    SimplicialCochain,
    SimplicialMap,
)
from jacobian.math.topology.cohomology.operations._tools import TOOLS
from jacobian.math.topology.cohomology.operations.operations import (
    cohomology_ring,
    cup_product,
    induced_cohomology_map,
    verify_cohomology_ring,
    verify_cup_product,
    verify_induced_cohomology_map,
)
from jacobian.math.topology.operations import canonicalize


def _tool(operation_id: str) -> MathTool[Any, Any]:
    return next(tool for tool in TOOLS if tool.operation_id == operation_id)


def _complex(
    vertices: tuple[str, ...] | list[str],
    facets: tuple[list[str], ...] | list[list[str]],
) -> FiniteSimplicialComplex:
    return canonicalize(
        tuple(vertices), tuple(tuple(facet) for facet in facets)
    ).complex


_CIRCLE = _complex(["a", "b", "c"], [["a", "b"], ["b", "c"], ["a", "c"]])


def _solid_tetrahedron() -> FiniteSimplicialComplex:
    return _complex(["a", "b", "c", "d"], [["a", "b", "c", "d"]])


def _torus() -> FiniteSimplicialComplex:
    verts = [f"v{i}{j}" for i in range(3) for j in range(3)]
    facets = []
    for i in range(3):
        for j in range(3):
            a = f"v{i}{j}"
            b = f"v{(i + 1) % 3}{j}"
            c = f"v{i}{(j + 1) % 3}"
            d = f"v{(i + 1) % 3}{(j + 1) % 3}"
            facets.append([a, b, d])
            facets.append([a, d, c])
    return _complex(verts, facets)


def _wedge() -> FiniteSimplicialComplex:
    return _complex(
        ["w", "a", "b", "c", "d", "p", "q", "r"],
        [
            ["w", "a"],
            ["a", "b"],
            ["b", "w"],
            ["w", "c"],
            ["c", "d"],
            ["d", "w"],
            ["w", "p", "q"],
            ["w", "p", "r"],
            ["w", "q", "r"],
            ["p", "q", "r"],
        ],
    )


def _cochain(
    complex_: FiniteSimplicialComplex,
    prime: int,
    degree: int,
    coefficients: tuple[int, ...],
) -> SimplicialCochain:
    return SimplicialCochain(
        complex=complex_, prime=prime, degree=degree, coefficients=coefficients
    )


def _faces(
    complex_: FiniteSimplicialComplex, degree: int
) -> tuple[tuple[str, ...], ...]:
    for entry in complex_.faces_by_dimension:
        if entry.dimension == degree:
            return entry.faces
    return ()


def _coboundary_matrix(
    complex_: FiniteSimplicialComplex, degree: int, prime: int
) -> list[list[int]]:
    """Independent coboundary matrix for the tests (transpose of boundary)."""

    if degree >= complex_.dimension:
        return []
    source = _faces(complex_, degree + 1)
    target = _faces(complex_, degree)
    row_for = {face: index for index, face in enumerate(target)}
    matrix = [[0] * len(source) for _ in range(len(target))]
    for column, simplex in enumerate(source):
        for removed in range(len(simplex)):
            face = simplex[:removed] + simplex[removed + 1 :]
            matrix[row_for[face]][column] = 1 if removed % 2 == 0 else prime - 1
    # Coboundary delta^degree maps C^degree -> C^(degree+1): transpose.
    return [
        [matrix[row][column] for row in range(len(target))]
        for column in range(len(source))
    ]


def _apply_coboundary(
    complex_: FiniteSimplicialComplex,
    degree: int,
    coefficients: tuple[int, ...],
    prime: int,
) -> tuple[int, ...]:
    matrix = _coboundary_matrix(complex_, degree, prime)
    if not matrix:
        return ()
    return tuple(
        sum(
            entry * coefficient
            for entry, coefficient in zip(row, coefficients, strict=True)
        )
        % prime
        for row in matrix
    )


class TestCupProduct:
    def test_above_dimension_product_is_empty(self) -> None:
        left = _cochain(_CIRCLE, 2, 1, (1, 0, 0))
        right = _cochain(_CIRCLE, 2, 1, (1, 0, 0))

        result = cup_product(_CIRCLE, 2, left, right)

        assert result.product.degree == 2
        assert result.product.coefficients == ()
        assert verify_cup_product(result)

    def test_leibniz_rule_on_solid_tetrahedron(self) -> None:
        # d(a U b) = da U b + (-1)^p a U db, checked against the naive
        # coboundary with all arithmetic recomputed in the test.
        ball = _solid_tetrahedron()
        prime = 5
        edges = _faces(ball, 1)
        triangles = _faces(ball, 2)
        tetrahedra = _faces(ball, 3)
        assert len(edges) == 6 and len(triangles) == 4 and len(tetrahedra) == 1
        left = _cochain(ball, prime, 1, (1, 2, 0, 1, 0, 3))
        right = _cochain(ball, prime, 1, (0, 1, 4, 0, 2, 1))

        def cup_on(
            left_degree: int,
            left_faces: tuple[tuple[str, ...], ...],
            left_coeffs: tuple[int, ...],
            right_faces: tuple[tuple[str, ...], ...],
            right_coeffs: tuple[int, ...],
            target_faces: tuple[tuple[str, ...], ...],
        ) -> tuple[int, ...]:
            left_values = dict(zip(left_faces, left_coeffs, strict=True))
            right_values = dict(zip(right_faces, right_coeffs, strict=True))
            return tuple(
                (
                    left_values[face[: left_degree + 1]]
                    * right_values[face[left_degree:]]
                )
                % prime
                for face in target_faces
            )

        product = cup_on(
            1, edges, left.coefficients, edges, right.coefficients, triangles
        )
        left_side = _apply_coboundary(ball, 2, product, prime)
        left_coboundary = _apply_coboundary(ball, 1, left.coefficients, prime)
        right_coboundary = _apply_coboundary(ball, 1, right.coefficients, prime)
        right_side = tuple(
            (a + prime - b) % prime
            for a, b in zip(
                cup_on(
                    2, triangles, left_coboundary, edges, right.coefficients, tetrahedra
                ),
                cup_on(
                    1, edges, left.coefficients, triangles, right_coboundary, tetrahedra
                ),
                strict=True,
            )
        )
        assert left_side == right_side

    def test_cocycle_product_is_cocycle(self) -> None:
        ball = _solid_tetrahedron()
        prime = 5
        unit = _cochain(ball, prime, 0, (1, 1, 1, 1))
        # A 1-coboundary is a 1-cocycle; (1,0,0,0) is not constant.
        coboundary = _cochain(
            ball,
            prime,
            1,
            _apply_coboundary(ball, 0, (1, 0, 0, 0), prime),
        )
        assert any(coboundary.coefficients)

        product = cup_product(ball, prime, unit, coboundary).product

        assert _apply_coboundary(ball, 1, product.coefficients, prime) == tuple(
            0 for _ in _faces(ball, 2)
        )

    def test_native_and_catalog_paths_agree(self) -> None:
        left = _cochain(_CIRCLE, 2, 1, (1, 0, 0))
        right = _cochain(_CIRCLE, 2, 1, (0, 1, 0))
        tool = _tool("topology.simplicial.cup_product.compute")

        assert tool.run(
            CupProductRequest(complex=_CIRCLE, prime=2, left=left, right=right)
        ) == cup_product(_CIRCLE, 2, left, right)

    def test_round_trip_and_forgery(self) -> None:
        left = _cochain(_CIRCLE, 2, 1, (1, 0, 0))
        right = _cochain(_CIRCLE, 2, 1, (0, 1, 0))
        result = cup_product(_CIRCLE, 2, left, right)
        restored = CupProductResult.model_validate_json(result.model_dump_json())

        assert restored == result
        assert verify_cup_product(restored)
        forged = json.loads(restored.model_dump_json())
        forged["product"]["coefficients"] = [1]
        with pytest.raises(ValidationError):
            CupProductResult.model_validate_json(json.dumps(forged))

    def test_product_above_the_complex_dimension_is_empty(self) -> None:
        # A 6-simplex with two degree-6 cochains: the product degree 12 is
        # above the complex dimension and above MAX_TOPOLOGY_DIMENSION, so it
        # must be the empty cochain rather than a schema error.
        simplex = _complex([str(index) for index in range(7)], [list("0123456")])
        left = _cochain(simplex, 2, 6, (1,))
        right = _cochain(simplex, 2, 6, (1,))

        product = cup_product(simplex, 2, left, right).product

        assert product.degree == 12
        assert product.coefficients == ()
        assert verify_cup_product(cup_product(simplex, 2, left, right))

    def test_product_degree_beyond_the_envelope_is_rejected(self) -> None:
        # Two degree-14 cochains have total degree 28, outside the retained
        # MAX_COCHAIN_DEGREE envelope; the operation raises a typed error.
        left = _cochain(_CIRCLE, 2, 14, ())
        right = _cochain(_CIRCLE, 2, 14, ())

        with pytest.raises(OperationDomainValidationError) as exc_info:
            cup_product(_CIRCLE, 2, left, right)
        assert (
            exc_info.value.errors()[0]["type"] == "topology.cup_product_degree_bound"
        )


class TestCohomologyRing:
    def test_torus_has_nonzero_degree_one_products(self) -> None:
        ring = cohomology_ring(_torus(), 2)
        table = {
            (entry.left_index, entry.right_index): entry.class_components
            for entry in ring.products
            if entry.left_degree == 1 and entry.right_degree == 1
        }

        assert table == {(0, 0): (0,), (0, 1): (1,), (1, 0): (1,), (1, 1): (0,)}
        assert verify_cohomology_ring(ring)

    def test_wedge_has_vanishing_positive_products(self) -> None:
        ring = cohomology_ring(_wedge(), 2)

        assert {
            group.dimension: group.betti_number for group in ring.cohomology.groups
        } == {
            0: 1,
            1: 2,
            2: 1,
        }
        for entry in ring.products:
            if entry.left_degree > 0 and entry.right_degree > 0:
                assert entry.class_components == (0,) * len(entry.class_components)

    def test_graded_commutativity_over_f3(self) -> None:
        ring = cohomology_ring(_torus(), 3)
        table = {
            (entry.left_index, entry.right_index): entry.class_components
            for entry in ring.products
            if entry.left_degree == 1 and entry.right_degree == 1
        }

        for (first, second), components in table.items():
            mirror = table[(second, first)]
            assert tuple((-value) % 3 for value in mirror) == components

    def test_unit_element(self) -> None:
        ring = cohomology_ring(_torus(), 2)
        group_one = next(
            group for group in ring.cohomology.groups if group.dimension == 1
        )
        assert len(group_one.cohomology_basis) == 2
        for entry in ring.products:
            if entry.left_degree == 0 and entry.right_degree == 1:
                expected = [0, 0]
                expected[entry.right_index] = 1
                assert entry.class_components == tuple(expected)

    def test_exact_cochain_equation_replays(self) -> None:
        # product == sum(class) + delta(exact), recomputed naively in the test.
        torus = _torus()
        prime = 2
        ring = cohomology_ring(torus, prime)
        groups = {group.dimension: group for group in ring.cohomology.groups}
        for entry in ring.products:
            total = entry.left_degree + entry.right_degree
            target_faces = _faces(torus, total)
            left_vector = (
                groups[entry.left_degree]
                .cohomology_basis[entry.left_index]
                .coefficients
            )
            right_vector = (
                groups[entry.right_degree]
                .cohomology_basis[entry.right_index]
                .coefficients
            )
            left = _cochain(torus, prime, entry.left_degree, left_vector)
            right = _cochain(torus, prime, entry.right_degree, right_vector)
            product = cup_product(torus, prime, left, right).product.coefficients
            target = groups[total]
            rebuilt = [0] * len(target_faces)
            for coefficient, basis in zip(
                entry.class_components, target.cohomology_basis, strict=True
            ):
                for position, value in enumerate(basis.coefficients):
                    rebuilt[position] = (
                        rebuilt[position] + coefficient * value
                    ) % prime
            exact = [0] * len(target_faces)
            for coefficient, basis in zip(
                entry.coboundary_components, target.coboundary_basis, strict=True
            ):
                # Coboundary basis vectors already live in C^total coordinates.
                for position, value in enumerate(basis.coefficients):
                    exact[position] = (exact[position] + coefficient * value) % prime
            assert (
                tuple(
                    (rebuilt[position] + exact[position]) % prime
                    for position in range(len(target_faces))
                )
                == product
            )

    def test_native_and_catalog_paths_agree(self) -> None:
        tool = _tool("topology.simplicial.cohomology_ring.compute")

        assert tool.run(
            CohomologyRingRequest(complex=_CIRCLE, prime=2)
        ) == cohomology_ring(_CIRCLE, 2)

    def test_round_trip_and_forgery(self) -> None:
        result = cohomology_ring(_torus(), 2)
        restored = CohomologyRingResult.model_validate_json(result.model_dump_json())

        assert restored == result
        assert verify_cohomology_ring(restored)
        forged = json.loads(restored.model_dump_json())
        position = next(
            index
            for index, entry in enumerate(forged["products"])
            if all(value == 0 for value in entry["class_components"])
        )
        forged["products"][position]["class_components"] = [1] * len(
            forged["products"][position]["class_components"]
        )
        forged_claim = CohomologyRingResult.model_validate_json(json.dumps(forged))
        assert not verify_cohomology_ring(forged_claim)


class TestInducedMaps:
    def _identity(self, complex_: FiniteSimplicialComplex) -> SimplicialMap:
        return SimplicialMap(
            source=complex_, target=complex_, vertex_map=complex_.vertices
        )

    def test_identity_pulls_back_identically(self) -> None:
        result = induced_cohomology_map(self._identity(_torus()), 2)

        for matrix, group in zip(
            result.matrices, result.source_cohomology.groups, strict=True
        ):
            size = group.betti_number
            assert matrix.rows == tuple(
                tuple(1 if row == column else 0 for column in range(size))
                for row in range(size)
            )
        assert verify_induced_cohomology_map(result)

    def test_constant_map_kills_positive_degrees(self) -> None:
        constant = SimplicialMap(
            source=_CIRCLE, target=_CIRCLE, vertex_map=("a", "a", "a")
        )
        result = induced_cohomology_map(constant, 2)
        by_degree = {matrix.degree: matrix for matrix in result.matrices}

        assert by_degree[0].rows == ((1,),)
        assert by_degree[1].rows == ((0,),)

    def test_zero_source_cohomology_pullback_is_the_zero_matrix(self) -> None:
        # A disc collapses onto a circle edge.  H^1(disc) = 0, so every
        # pullback is a coboundary; the pulled generator need not be the zero
        # cochain, and the induced degree-1 matrix must be the 0x1 zero map.
        disc = _complex(["0", "1", "2"], [["0", "1", "2"]])
        collapse = SimplicialMap(source=disc, target=_CIRCLE, vertex_map=("a", "b", "b"))

        result = induced_cohomology_map(collapse, 2)
        by_degree = {matrix.degree: matrix for matrix in result.matrices}

        assert by_degree[1].rows == ()
        assert verify_induced_cohomology_map(result)

    def test_functoriality_on_the_circle(self) -> None:
        # (g o f)* = f* o g* as exact matrices, checked entrywise.
        fold = SimplicialMap(source=_CIRCLE, target=_CIRCLE, vertex_map=("a", "a", "c"))
        rotation = SimplicialMap(
            source=_CIRCLE, target=_CIRCLE, vertex_map=("b", "c", "a")
        )
        composed_vertices = tuple(
            rotation.vertex_map[_CIRCLE.vertices.index(label)]
            for label in fold.vertex_map
        )
        composed = SimplicialMap(
            source=_CIRCLE, target=_CIRCLE, vertex_map=composed_vertices
        )
        direct = induced_cohomology_map(composed, 2)
        first = induced_cohomology_map(fold, 2)
        second = induced_cohomology_map(rotation, 2)

        for direct_matrix, fold_matrix, rotation_matrix in zip(
            direct.matrices, first.matrices, second.matrices, strict=True
        ):
            # (g o f)*[i][j] = sum_k fold*[i][k] rotation*[k][j].
            rotation_columns = (
                list(zip(*rotation_matrix.rows, strict=True))
                if (rotation_matrix.rows and rotation_matrix.rows[0])
                else [() for _ in fold_matrix.rows[0]]
                if fold_matrix.rows
                else []
            )
            expected = tuple(
                tuple(
                    sum(
                        fold_entry * rotation_entry
                        for fold_entry, rotation_entry in zip(
                            fold_row, rotation_column, strict=True
                        )
                    )
                    % 2
                    for rotation_column in rotation_columns
                )
                for fold_row in fold_matrix.rows
            )
            assert direct_matrix.rows == expected

    def test_cup_commutes_with_pullback_cohomologously(self) -> None:
        # Cochain-level Alexander-Whitney naturality is FALSE in general
        # for order-mixing simplicial maps; it holds in cohomology.  Both
        # facts are checked here with independent naive code: the two sides
        # differ as cochains but agree modulo coboundaries.
        torus = _torus()
        prime = 2
        rotation = SimplicialMap(
            source=torus,
            target=torus,
            vertex_map=tuple(f"v{(i + 1) % 3}{j}" for i in range(3) for j in range(3)),
        )
        ring = cohomology_ring(torus, prime)
        group_one = next(
            group for group in ring.cohomology.groups if group.dimension == 1
        )
        left = group_one.cohomology_basis[0].coefficients
        right = group_one.cohomology_basis[1].coefficients
        edges = _faces(torus, 1)
        triangles = _faces(torus, 2)
        vertex_index = {
            label: position for position, label in enumerate(torus.vertices)
        }

        def sign_of(image: tuple[str, ...]) -> int:
            ordered = tuple(sorted(image))
            inversions = sum(
                1
                for first in range(len(image))
                for second in range(first + 1, len(image))
                if ordered.index(image[first]) > ordered.index(image[second])
            )
            return prime - 1 if inversions % 2 else 1

        def naive_pullback(
            faces: tuple[tuple[str, ...], ...], coefficients: tuple[int, ...]
        ) -> tuple[int, ...]:
            values = dict(zip(faces, coefficients, strict=True))
            pulled = []
            for face in faces:
                image = tuple(
                    rotation.vertex_map[vertex_index[vertex]] for vertex in face
                )
                if len(set(image)) != len(image):
                    pulled.append(0)
                    continue
                pulled.append(sign_of(image) * values[tuple(sorted(image))] % prime)
            return tuple(pulled)

        def naive_cup(
            left_coeffs: tuple[int, ...], right_coeffs: tuple[int, ...]
        ) -> tuple[int, ...]:
            left_values = dict(zip(edges, left_coeffs, strict=True))
            right_values = dict(zip(edges, right_coeffs, strict=True))
            return tuple(
                (left_values[face[:2]] * right_values[face[1:]]) % prime
                for face in triangles
            )

        together = naive_pullback(triangles, naive_cup(left, right))
        separate = naive_cup(naive_pullback(edges, left), naive_pullback(edges, right))
        assert together != separate
        difference = tuple(
            (a + b) % prime for a, b in zip(together, separate, strict=True)
        )
        # The difference is a coboundary: solve delta^1 e = difference.
        boundary = _coboundary_matrix(torus, 1, prime)
        augmented = [
            [*row, value] for row, value in zip(boundary, difference, strict=True)
        ]
        rows = [row[:] for row in augmented]
        pivot_row = 0
        for column in range(len(edges)):
            pivot = next(
                (
                    candidate
                    for candidate in range(pivot_row, len(rows))
                    if rows[candidate][column]
                ),
                None,
            )
            if pivot is None:
                continue
            rows[pivot_row], rows[pivot] = rows[pivot], rows[pivot_row]
            for other in range(len(rows)):
                if other != pivot_row and rows[other][column]:
                    rows[other] = [
                        (a + b) % prime
                        for a, b in zip(rows[other], rows[pivot_row], strict=True)
                    ]
            pivot_row += 1
        # The difference is a coboundary: the augmented system has no
        # inconsistent row (zero coefficients with nonzero remainder).
        assert not any(
            all(value == 0 for value in row[: len(edges)]) and row[len(edges)]
            for row in rows
        )

    def test_non_simplicial_map_is_rejected(self) -> None:
        target = _complex(["x", "y"], [["x"], ["y"]])
        with pytest.raises(ValidationError):
            SimplicialMap(source=_CIRCLE, target=target, vertex_map=("x", "y", "x"))

    def test_native_and_catalog_paths_agree(self) -> None:
        tool = _tool("topology.simplicial_map.induced_cohomology.compute")

        assert tool.run(
            InducedCohomologyMapRequest(map=self._identity(_CIRCLE), prime=2)
        ) == induced_cohomology_map(self._identity(_CIRCLE), 2)

    def test_round_trip_and_forgery(self) -> None:
        result = induced_cohomology_map(self._identity(_torus()), 2)
        restored = InducedCohomologyMapResult.model_validate_json(
            result.model_dump_json()
        )

        assert restored == result
        assert verify_induced_cohomology_map(restored)
        forged = json.loads(restored.model_dump_json())
        forged["matrices"][1]["rows"] = [[0, 0], [0, 0]]
        forged_claim = InducedCohomologyMapResult.model_validate_json(
            json.dumps(forged)
        )
        assert not verify_induced_cohomology_map(forged_claim)


class TestAdmission:
    def test_mismatched_parent_cochain_is_rejected(self) -> None:
        left = SimplicialCochain(
            complex=_CIRCLE, prime=2, degree=1, coefficients=(1, 0, 0)
        )
        other = _complex(["p", "q"], [["p", "q"]])
        right = SimplicialCochain(complex=other, prime=2, degree=1, coefficients=(1,))

        with pytest.raises(OperationDomainValidationError):
            cup_product(_CIRCLE, 2, left, right)

    def test_degenerate_map_is_simplicial(self) -> None:
        # Collapsing an edge onto a vertex still sends faces to faces.
        collapsed = SimplicialMap(
            source=_CIRCLE, target=_CIRCLE, vertex_map=("a", "a", "c")
        )
        assert collapsed.vertex_map == ("a", "a", "c")

    def test_oversized_prime_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CupProductRequest(
                complex=_CIRCLE,
                prime=252,
                left=SimplicialCochain(
                    complex=_CIRCLE, prime=252, degree=1, coefficients=(1, 0, 0)
                ),
                right=SimplicialCochain(
                    complex=_CIRCLE, prime=252, degree=1, coefficients=(1, 0, 0)
                ),
            )
