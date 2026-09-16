"""Tests for topology.discrete_morse.gradient_paths/complex.compute (#3755)."""

from __future__ import annotations

import json
import random
from collections.abc import Iterator

import pytest

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.topology._models import (
    FiniteSimplicialComplex,
    HomologyConvention,
)
from jacobian.math.topology._simplicial_kernel import homology
from jacobian.math.topology.discrete_morse import (
    GradientPathsResult,
    MatchingPair,
    MorseComplexResult,
    MorseMatchingOutcome,
    compute_gradient_paths,
    compute_morse_complex,
    construct_matching,
)
from jacobian.math.topology.discrete_morse._models import (
    GradientPathsRequest,
    MorseComplexRequest,
)
from jacobian.math.topology.discrete_morse._tools import TOOLS
from jacobian.math.topology.operations import canonicalize

GRADIENT_OPERATION_ID = "topology.discrete_morse.gradient_paths.compute"
COMPLEX_OPERATION_ID = "topology.discrete_morse.complex.compute"


def _tool(operation_id: str):
    return next(tool for tool in TOOLS if tool.operation_id == operation_id)


def _complex(vertices: list[str], facets: list[list[str]]) -> FiniteSimplicialComplex:
    return canonicalize(
        tuple(vertices), tuple(tuple(facet) for facet in facets)
    ).complex


def _pairs(
    entries: list[tuple[list[str], list[str]]],
) -> tuple[MatchingPair, ...]:
    return tuple(
        MatchingPair(face=tuple(face), coface=tuple(coface)) for face, coface in entries
    )


def _oracle_betti(complex_: FiniteSimplicialComplex) -> tuple[int, ...]:
    result = homology(complex_, 2, HomologyConvention.UNREDUCED)
    return tuple(group.betti_number for group in result.groups)


CIRCLE = _complex(["a", "b", "c"], [["a", "b"], ["b", "c"], ["a", "c"]])
CIRCLE_PAIRS = _pairs([(["a"], ["a", "b"]), (["c"], ["a", "c"])])
CIRCLE_CYCLIC = _pairs([(["a"], ["a", "b"]), (["b"], ["b", "c"]), (["c"], ["a", "c"])])
INTERVAL = _complex(["a", "b"], [["a", "b"]])
TRIANGLE = _complex(["a", "b", "c"], [["a", "b", "c"]])
TRIANGLE_PAIRS = _pairs(
    [(["b"], ["a", "b"]), (["c"], ["b", "c"]), (["a", "c"], ["a", "b", "c"])]
)
DISK = _complex(["a", "b", "c", "d"], [["a", "b", "c"], ["a", "c", "d"]])
TETRA = _complex(
    ["a", "b", "c", "d"],
    [["a", "b", "c"], ["a", "b", "d"], ["a", "c", "d"], ["b", "c", "d"]],
)
TETRA_PAIRS = _pairs(
    [
        (["a"], ["a", "b"]),
        (["a", "c"], ["a", "c", "d"]),
        (["b", "d"], ["a", "b", "d"]),
        (["c"], ["b", "c"]),
        (["c", "d"], ["b", "c", "d"]),
        (["d"], ["a", "d"]),
    ]
)


def _torus() -> FiniteSimplicialComplex:
    vertices = [f"{i}{j}" for i in range(3) for j in range(3)]
    facets: list[list[str]] = []
    for i in range(3):
        for j in range(3):
            a = f"{i}{j}"
            b = f"{(i + 1) % 3}{j}"
            c = f"{i}{(j + 1) % 3}"
            d = f"{(i + 1) % 3}{(j + 1) % 3}"
            facets.append([a, b, c])
            facets.append([b, d, c])
    return _complex(vertices, facets)


TORUS = _torus()
TORUS_PAIRS = _pairs(
    [
        (["00", "20"], ["00", "20", "21"]),
        (["01", "10"], ["01", "10", "11"]),
        (["02"], ["02", "11"]),
        (["11", "21"], ["11", "12", "21"]),
        (["21"], ["12", "21"]),
        (["00", "02"], ["00", "02", "12"]),
        (["11", "20"], ["10", "11", "20"]),
        (["10"], ["10", "11"]),
        (["12"], ["11", "12"]),
        (["01", "22"], ["01", "21", "22"]),
        (["00", "01"], ["00", "01", "10"]),
        (["20", "21"], ["11", "20", "21"]),
        (["22"], ["21", "22"]),
        (["01"], ["01", "11"]),
        (["00", "12"], ["00", "10", "12"]),
        (["20"], ["20", "22"]),
        (["10", "20"], ["10", "20", "22"]),
        (["00", "21"], ["00", "01", "21"]),
        (["01", "02"], ["01", "02", "11"]),
        (["02", "22"], ["01", "02", "22"]),
        (["02", "20"], ["02", "20", "22"]),
        (["10", "12"], ["10", "12", "22"]),
        (["12", "22"], ["12", "21", "22"]),
        (["00"], ["00", "10"]),
        (["02", "12"], ["02", "11", "12"]),
    ]
)


def _all_covers(complex_: FiniteSimplicialComplex) -> list[tuple[tuple, tuple]]:
    cells = tuple(tuple(group.faces) for group in complex_.faces_by_dimension)
    covers: list[tuple[tuple, tuple]] = []
    for dimension in range(len(cells) - 1):
        lower = set(cells[dimension])
        for coface in cells[dimension + 1]:
            for index in range(len(coface)):
                face = coface[:index] + coface[index + 1 :]
                if face in lower:
                    covers.append((face, coface))
    return sorted(covers)


def _acyclic_matchings(
    complex_: FiniteSimplicialComplex, count: int = 3
) -> Iterator[tuple[MatchingPair, ...]]:
    covers = _all_covers(complex_)
    found = 0
    for seed in range(2000):
        rng = random.Random(seed)
        shuffled = list(covers)
        rng.shuffle(shuffled)
        used: set[tuple] = set()
        pairs: list[MatchingPair] = []
        for face, coface in shuffled:
            if face in used or coface in used:
                continue
            used.add(face)
            used.add(coface)
            pairs.append(MatchingPair(face=face, coface=coface))
        candidate = tuple(pairs)
        if (
            construct_matching(complex_, candidate).outcome
            is MorseMatchingOutcome.ACYCLIC_MATCHING
        ):
            yield candidate
            found += 1
            if found == count:
                return


class TestGradientPathsKnownAnswer:
    def test_circle_edge_has_two_gradient_paths_to_critical_vertex(self) -> None:
        result = compute_gradient_paths(CIRCLE, CIRCLE_PAIRS, ("b", "c"), ("b",))
        assert len(result.paths) == 2
        assert result.counts_by_target[0].target == ("b",)
        assert result.counts_by_target[0].count == 2
        short, long = sorted(result.paths, key=lambda path: len(path.steps))
        assert [
            (step.kind.value, step.source, step.target) for step in short.steps
        ] == [("DOWN", ("b", "c"), ("b",))]
        assert [step.kind.value for step in long.steps] == [
            "DOWN",
            "UP",
            "DOWN",
            "UP",
            "DOWN",
        ]

    def test_interval_single_down_step(self) -> None:
        result = compute_gradient_paths(INTERVAL, (), ("a", "b"), ("a",))
        assert len(result.paths) == 1
        assert result.paths[0].start == ("a", "b")
        assert result.paths[0].target == ("a",)
        assert len(result.paths[0].steps) == 1

    def test_all_lower_targets_when_target_omitted(self) -> None:
        result = compute_gradient_paths(TRIANGLE, (), ("a", "b", "c"))
        assert {item.target for item in result.counts_by_target} == {
            ("a", "b"),
            ("a", "c"),
            ("b", "c"),
        }
        assert all(item.count == 1 for item in result.counts_by_target)
        assert len(result.paths) == 3

    def test_vertex_start_has_no_lower_critical_target(self) -> None:
        result = compute_gradient_paths(CIRCLE, CIRCLE_PAIRS, ("b",))
        assert result.paths == ()


class TestGradientPathReplay:
    def _matched_and_upper(
        self, pairs: tuple[MatchingPair, ...]
    ) -> tuple[set[tuple[tuple, tuple]], dict[tuple, tuple]]:
        matched = {(pair.face, pair.coface) for pair in pairs}
        upper = {pair.coface: pair.face for pair in pairs}
        return matched, upper

    @pytest.mark.parametrize(
        "complex_, pairs", [(CIRCLE, CIRCLE_PAIRS), (TRIANGLE, ())]
    )
    def test_steps_are_cover_relations_alternating_matched(
        self, complex_: FiniteSimplicialComplex, pairs: tuple[MatchingPair, ...]
    ) -> None:
        matched, upper = self._matched_and_upper(pairs)
        starts = [cell for cell in _all_critical(complex_, pairs) if len(cell) >= 2]
        for start in starts:
            result = compute_gradient_paths(complex_, pairs, start)
            for path in result.paths:
                for step in path.steps:
                    lower, higher = sorted((step.source, step.target), key=len)
                    assert len(higher) == len(lower) + 1
                    assert set(lower).issubset(higher)
                    if step.kind.value == "UP":
                        assert (step.source, step.target) in matched
                    else:
                        assert (step.target, step.source) not in matched
                        assert upper.get(step.source) != step.target


def _all_critical(
    complex_: FiniteSimplicialComplex, pairs: tuple[MatchingPair, ...]
) -> tuple[tuple, ...]:
    result = construct_matching(complex_, pairs)
    assert result.outcome is MorseMatchingOutcome.ACYCLIC_MATCHING
    assert result.critical_profile is not None
    return result.critical_profile.critical_cells


class TestMorseComplexKnownAnswer:
    def test_circle_morse_complex(self) -> None:
        result = compute_morse_complex(CIRCLE, CIRCLE_PAIRS)
        assert result.critical_profile.counts_by_dimension == (1, 1)
        assert tuple(basis.cells for basis in result.critical_cells_by_dimension) == (
            (("b",),),
            (("b", "c"),),
        )
        entries = [
            (entry.source, entry.target, entry.gradient_path_count, entry.coefficient)
            for entry in result.boundary_entries
        ]
        assert entries == [(("b", "c"), ("b",), 2, 0)]
        assert result.betti_numbers == (1, 1)
        assert result.boundary_square_zero is True

    def test_filled_triangle_collapses_to_single_critical_vertex(self) -> None:
        result = compute_morse_complex(TRIANGLE, TRIANGLE_PAIRS)
        assert result.critical_profile.counts_by_dimension == (1, 0, 0)
        assert result.critical_profile.critical_cells == (("a",),)
        assert result.boundary_entries == ()
        assert result.betti_numbers == (1, 0, 0)

    def test_two_sphere_tetrahedron(self) -> None:
        result = compute_morse_complex(TETRA, TETRA_PAIRS)
        assert result.critical_profile.counts_by_dimension == (1, 0, 1)
        assert result.betti_numbers == (1, 0, 1)
        assert result.boundary_entries == ()

    def test_interval_empty_matching_has_full_boundary(self) -> None:
        result = compute_morse_complex(INTERVAL, ())
        entries = [
            (entry.source, entry.target, entry.gradient_path_count, entry.coefficient)
            for entry in result.boundary_entries
        ]
        assert entries == [
            (("a", "b"), ("a",), 1, 1),
            (("a", "b"), ("b",), 1, 1),
        ]
        assert result.betti_numbers == (1, 0)

    def test_torus_minimal_matching(self) -> None:
        result = compute_morse_complex(TORUS, TORUS_PAIRS)
        assert result.critical_profile.counts_by_dimension == (1, 2, 1)
        assert result.betti_numbers == (1, 2, 1)
        assert result.boundary_square_zero is True
        # Every reduced boundary coefficient is the parity of a complete
        # gradient-path count, so each entry's parity is inspectable.
        assert all(
            entry.coefficient == entry.gradient_path_count % 2
            for entry in result.boundary_entries
        )


class TestMorseBettiOracle:
    @pytest.mark.parametrize(
        ("complex_", "pairs"),
        [
            (INTERVAL, ()),
            (CIRCLE, ()),
            (CIRCLE, CIRCLE_PAIRS),
            (TRIANGLE, ()),
            (TRIANGLE, TRIANGLE_PAIRS),
            (DISK, ()),
            (TETRA, ()),
            (TETRA, TETRA_PAIRS),
            (TORUS, ()),
            (TORUS, TORUS_PAIRS),
        ],
    )
    def test_reduced_betti_matches_simplicial_homology(
        self, complex_: FiniteSimplicialComplex, pairs: tuple[MatchingPair, ...]
    ) -> None:
        result = compute_morse_complex(complex_, pairs)
        assert result.betti_numbers == _oracle_betti(complex_)

    @pytest.mark.parametrize("complex_", [CIRCLE, DISK, TETRA, TORUS])
    def test_random_acyclic_matchings_match_homology(
        self, complex_: FiniteSimplicialComplex
    ) -> None:
        checked = 0
        for pairs in _acyclic_matchings(complex_, count=2):
            result = compute_morse_complex(complex_, pairs)
            assert result.betti_numbers == _oracle_betti(complex_)
            assert result.boundary_square_zero is True
            checked += 1
        assert checked >= 1


class TestEulerIdentity:
    @pytest.mark.parametrize(
        ("complex_", "pairs"),
        [
            (INTERVAL, ()),
            (CIRCLE, CIRCLE_PAIRS),
            (TRIANGLE, TRIANGLE_PAIRS),
            (TETRA, TETRA_PAIRS),
            (TORUS, TORUS_PAIRS),
        ],
    )
    def test_morse_euler_equals_closure_euler_and_critical_sum(
        self, complex_: FiniteSimplicialComplex, pairs: tuple[MatchingPair, ...]
    ) -> None:
        result = compute_morse_complex(complex_, pairs)
        counts = result.critical_profile.counts_by_dimension
        alternating = sum(
            (-1) ** dimension * count for dimension, count in enumerate(counts)
        )
        closure = sum(
            (-1) ** dimension * count
            for dimension, count in enumerate(complex_.f_vector)
        )
        assert result.morse_euler_characteristic == alternating
        assert result.closure_euler_characteristic == closure
        assert alternating == closure


class TestBoundarySquareZero:
    @pytest.mark.parametrize(
        ("complex_", "pairs"),
        [(CIRCLE, CIRCLE_PAIRS), (INTERVAL, ()), (TETRA, TETRA_PAIRS), (TORUS, ())],
    )
    def test_reported_square_zero_replays_from_entries(
        self, complex_: FiniteSimplicialComplex, pairs: tuple[MatchingPair, ...]
    ) -> None:
        result = compute_morse_complex(complex_, pairs)
        rows: dict[tuple, set[tuple]] = {}
        for entry in result.boundary_entries:
            if entry.coefficient:
                rows.setdefault(entry.source, set()).add(entry.target)
        for targets in rows.values():
            accumulated: set[tuple] = set()
            for target in targets:
                accumulated ^= rows.get(target, set())
            assert accumulated == set()
        assert result.boundary_square_zero is True


class TestAdversarial:
    def test_cyclic_matching_is_rejected(self) -> None:
        with pytest.raises(OperationDomainValidationError):
            compute_gradient_paths(CIRCLE, CIRCLE_CYCLIC, ("b", "c"), ("b",))
        with pytest.raises(OperationDomainValidationError):
            compute_morse_complex(CIRCLE, CIRCLE_CYCLIC)

    def test_non_cover_pair_is_rejected(self) -> None:
        invalid = _pairs([(["a"], ["a", "b", "c"])])
        with pytest.raises(OperationDomainValidationError):
            compute_morse_complex(TRIANGLE, invalid)

    def test_non_critical_start_is_rejected(self) -> None:
        with pytest.raises(OperationDomainValidationError):
            compute_gradient_paths(CIRCLE, CIRCLE_PAIRS, ("a",), ("a", "b"))

    def test_unknown_start_is_rejected(self) -> None:
        with pytest.raises(OperationDomainValidationError):
            compute_gradient_paths(CIRCLE, CIRCLE_PAIRS, ("z",))

    def test_non_critical_target_is_rejected(self) -> None:
        with pytest.raises(OperationDomainValidationError):
            compute_gradient_paths(CIRCLE, CIRCLE_PAIRS, ("b", "c"), ("c",))

    def test_target_dimension_is_rejected_natively(self) -> None:
        with pytest.raises(OperationDomainValidationError):
            compute_gradient_paths(CIRCLE, CIRCLE_PAIRS, ("b", "c"), ("b", "c"))

    def test_request_rejects_adjacent_dimension_violation(self) -> None:
        with pytest.raises(ValueError):
            GradientPathsRequest.model_validate(
                {
                    "complex": {
                        "vertices": ["a", "b", "c"],
                        "facets": [["a", "b"], ["b", "c"], ["a", "c"]],
                    },
                    "pairs": [],
                    "start": ["b", "c"],
                    "target": ["a", "b", "c"],
                }
            )

    def test_request_rejects_non_canonical_start(self) -> None:
        with pytest.raises(ValueError):
            GradientPathsRequest.model_validate(
                {
                    "complex": {
                        "vertices": ["a", "b", "c"],
                        "facets": [["a", "b"], ["b", "c"], ["a", "c"]],
                    },
                    "pairs": [],
                    "start": ["c", "b"],
                }
            )


class TestResourceEnvelope:
    def test_gradient_path_cap_is_a_resource_rejection(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import jacobian.math.topology.discrete_morse._kernel as kernel

        monkeypatch.setattr(kernel, "MAX_MORSE_GRADIENT_PATHS", 1)
        with pytest.raises(OperationResourceAdmissionError) as excinfo:
            compute_gradient_paths(CIRCLE, CIRCLE_PAIRS, ("b", "c"), ("b",))
        assert excinfo.value.errors()[0]["type"] == (
            "topology.discrete_morse.admission.gradient_paths"
        )

    def test_gradient_path_cap_at_boundary_is_admitted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import jacobian.math.topology.discrete_morse._kernel as kernel

        monkeypatch.setattr(kernel, "MAX_MORSE_GRADIENT_PATHS", 2)
        result = compute_gradient_paths(CIRCLE, CIRCLE_PAIRS, ("b", "c"), ("b",))
        assert len(result.paths) == 2

    def test_search_state_cap_is_a_resource_rejection(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import jacobian.math.topology.discrete_morse._kernel as kernel

        monkeypatch.setattr(kernel, "MAX_MORSE_GRADIENT_STATES", 0)
        with pytest.raises(OperationResourceAdmissionError) as excinfo:
            compute_gradient_paths(CIRCLE, CIRCLE_PAIRS, ("b", "c"), ("b",))
        assert excinfo.value.errors()[0]["type"] == (
            "topology.discrete_morse.admission.gradient_path_states"
        )

    def test_path_step_cap_is_a_resource_rejection(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import jacobian.math.topology.discrete_morse._kernel as kernel

        monkeypatch.setattr(kernel, "MAX_MORSE_PATH_STEPS", 1)
        with pytest.raises(OperationResourceAdmissionError) as excinfo:
            compute_gradient_paths(CIRCLE, CIRCLE_PAIRS, ("b", "c"), ("b",))
        assert excinfo.value.errors()[0]["type"] == (
            "topology.discrete_morse.admission.gradient_path_length"
        )

    def test_critical_cell_cap_is_a_resource_rejection(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import jacobian.math.topology.discrete_morse._kernel as kernel

        monkeypatch.setattr(kernel, "MAX_MORSE_CRITICAL_CELLS", 1)
        with pytest.raises(OperationResourceAdmissionError) as excinfo:
            compute_morse_complex(CIRCLE, CIRCLE_PAIRS)
        assert excinfo.value.errors()[0]["type"] == (
            "topology.discrete_morse.admission.critical_cells"
        )

    def test_boundary_entry_cap_is_a_resource_rejection(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import jacobian.math.topology.discrete_morse._kernel as kernel

        monkeypatch.setattr(kernel, "MAX_MORSE_BOUNDARY_ENTRIES", 0)
        with pytest.raises(OperationResourceAdmissionError) as excinfo:
            compute_morse_complex(INTERVAL, ())
        assert excinfo.value.errors()[0]["type"] == (
            "topology.discrete_morse.admission.boundary_entries"
        )


class TestNativeVsCatalogParity:
    def test_gradient_paths_catalog_matches_native(self) -> None:
        tool = _tool(GRADIENT_OPERATION_ID)
        request = GradientPathsRequest.model_validate(
            {
                "complex": {
                    "vertices": ["a", "b", "c"],
                    "facets": [["a", "b"], ["b", "c"], ["a", "c"]],
                },
                "pairs": [
                    {"face": ["a"], "coface": ["a", "b"]},
                    {"face": ["c"], "coface": ["a", "c"]},
                ],
                "start": ["b", "c"],
                "target": ["b"],
            }
        )
        assert tool.run(request) == compute_gradient_paths(
            CIRCLE, CIRCLE_PAIRS, ("b", "c"), ("b",)
        )

    def test_morse_complex_catalog_matches_native(self) -> None:
        tool = _tool(COMPLEX_OPERATION_ID)
        request = MorseComplexRequest.model_validate(
            {
                "complex": {"vertices": ["a", "b"], "facets": [["a", "b"]]},
                "pairs": [],
            }
        )
        assert tool.run(request) == compute_morse_complex(INTERVAL, ())

    @pytest.mark.parametrize(
        "operation_id", [GRADIENT_OPERATION_ID, COMPLEX_OPERATION_ID]
    )
    def test_published_examples_execute_via_strict_json_round_trip(
        self, operation_id: str
    ) -> None:
        tool = _tool(operation_id)
        assert tool.examples
        for example in tool.examples:
            request = tool.request_type.model_validate_json(json.dumps(example.input))
            result = tool.run(request)
            assert isinstance(result, (GradientPathsResult, MorseComplexResult))


class TestSerialization:
    def test_gradient_paths_result_round_trips(self) -> None:
        result = compute_gradient_paths(CIRCLE, CIRCLE_PAIRS, ("b", "c"), ("b",))
        restored = GradientPathsResult.model_validate_json(result.model_dump_json())
        assert restored == result

    def test_morse_complex_result_round_trips(self) -> None:
        result = compute_morse_complex(TORUS, TORUS_PAIRS)
        restored = MorseComplexResult.model_validate_json(result.model_dump_json())
        assert restored == result

    def test_forged_boundary_coefficient_fails_validation(self) -> None:
        result = compute_morse_complex(INTERVAL, ())
        payload = result.model_dump(mode="json")
        payload["boundary_entries"][0]["coefficient"] = 0
        with pytest.raises(ValueError):
            MorseComplexResult.model_validate(payload)

    def test_forged_basis_cell_fails_validation(self) -> None:
        result = compute_morse_complex(INTERVAL, ())
        payload = result.model_dump(mode="json")
        payload["critical_cells_by_dimension"][0]["cells"] = [["a", "b"]]
        with pytest.raises(ValueError):
            MorseComplexResult.model_validate(payload)


class TestConsumerComposition:
    def test_matching_construct_output_feeds_morse_complex_unchanged(self) -> None:
        matching = construct_matching(TORUS, TORUS_PAIRS)
        assert matching.outcome is MorseMatchingOutcome.ACYCLIC_MATCHING
        result = compute_morse_complex(matching.complex, matching.pairs)
        assert result.critical_profile == matching.critical_profile
        assert (
            tuple(
                cell
                for basis in result.critical_cells_by_dimension
                for cell in basis.cells
            )
            == matching.critical_profile.critical_cells
        )
        assert result.betti_numbers == _oracle_betti(matching.complex)

    def test_matching_construct_output_feeds_gradient_paths_unchanged(self) -> None:
        matching = construct_matching(CIRCLE, CIRCLE_PAIRS)
        assert matching.critical_profile is not None
        start = matching.critical_profile.critical_cells[1]
        result = compute_gradient_paths(matching.complex, matching.pairs, start)
        assert all(path.start == start for path in result.paths)
