from __future__ import annotations

from fractions import Fraction
from typing import Any, Literal, cast, overload

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import MathTool, OperationDomainValidationError
from jacobian.math.topology._homology import (
    IntegralSimplicialHomologyRequest,
    IntegralSimplicialHomologyResult,
    SimplicialHomologyRequest,
    SimplicialHomologyResult,
)
from jacobian.math.topology._models import (
    ChainCoefficientRing,
    ChainComplexRequest,
    ChainComplexResult,
    FiniteSimplicialComplex,
    SimplicialComplexCanonicalizationResult,
    SimplicialComplexRequest,
    simplicial_complex_digest,
)
from jacobian.math.topology._tools import TOOLS
from jacobian.math.topology.chain_complexes.values import (
    CoefficientRing,
    HomologyResult,
    IntegralHomologyGroupValue,
)


def test_simplicial_homology_result_roundtrip_does_not_retest_prime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    complex_ = _canonical_complex(("a", "b"), (("a", "b"),))
    result = _operation("topology.simplicial_homology.compute").run(
        SimplicialHomologyRequest(complex=complex_, prime=2)
    )
    calls: list[int] = []
    import jacobian.math.topology._models as topology_models

    original = topology_models.is_bounded_prime

    def tracked(value: int) -> bool:
        calls.append(value)
        return original(value)

    monkeypatch.setattr(topology_models, "is_bounded_prime", tracked)
    assert (
        SimplicialHomologyResult.model_validate(result.model_dump(mode="json"))
        == result
    )
    assert calls == []


@overload
def _operation(
    operation_id: Literal["topology.simplicial_complex.canonicalize"],
) -> MathTool[SimplicialComplexRequest, SimplicialComplexCanonicalizationResult]: ...


@overload
def _operation(
    operation_id: Literal["topology.simplicial_complex.chain_complex.compute"],
) -> MathTool[ChainComplexRequest, ChainComplexResult]: ...


@overload
def _operation(
    operation_id: Literal["topology.simplicial_homology.compute"],
) -> MathTool[SimplicialHomologyRequest, SimplicialHomologyResult]: ...


@overload
def _operation(
    operation_id: Literal["topology.simplicial_homology.integral.compute"],
) -> MathTool[IntegralSimplicialHomologyRequest, IntegralSimplicialHomologyResult]: ...


def _operation(operation_id: str) -> MathTool[Any, Any]:
    return next(tool for tool in TOOLS if tool.operation_id == operation_id)


def _integral_groups(
    result: HomologyResult,
) -> tuple[IntegralHomologyGroupValue, ...]:
    assert all(
        isinstance(group, IntegralHomologyGroupValue)
        for group in result.homology_groups
    )
    return cast(tuple[IntegralHomologyGroupValue, ...], result.homology_groups)


def _canonical_complex(
    vertices: tuple[str, ...], facets: tuple[tuple[str, ...], ...]
) -> FiniteSimplicialComplex:
    """Build a canonical FiniteSimplicialComplex via its owner declaration."""
    request = SimplicialComplexRequest(vertices=vertices, facets=facets)
    operation = _operation("topology.simplicial_complex.canonicalize")
    return operation.run(request).complex


def _rational_rank(rows: tuple[tuple[str, ...], ...]) -> int:
    """Small independent Gaussian rank oracle for topology fixtures."""

    matrix = [[Fraction(int(value)) for value in row] for row in rows]
    if not matrix:
        return 0
    pivot_row = 0
    for column in range(len(matrix[0])):
        pivot = next(
            (row for row in range(pivot_row, len(matrix)) if matrix[row][column]),
            None,
        )
        if pivot is None:
            continue
        matrix[pivot_row], matrix[pivot] = matrix[pivot], matrix[pivot_row]
        pivot_value = matrix[pivot_row][column]
        matrix[pivot_row] = [value / pivot_value for value in matrix[pivot_row]]
        for row in range(len(matrix)):
            if row == pivot_row or not matrix[row][column]:
                continue
            factor = matrix[row][column]
            matrix[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(
                    matrix[row], matrix[pivot_row], strict=True
                )
            ]
        pivot_row += 1
    return pivot_row


def test_facet_request_rejects_duplicates_nonmaximal_faces_and_hidden_isolates() -> (
    None
):
    with pytest.raises(OperationDomainValidationError):
        _operation("topology.simplicial_complex.canonicalize").run(
            SimplicialComplexRequest(
                vertices=("a", "b"), facets=(("a", "b"), ("b", "a"))
            )
        )
    with pytest.raises(OperationDomainValidationError):
        _operation("topology.simplicial_complex.canonicalize").run(
            SimplicialComplexRequest(vertices=("a", "b"), facets=(("a",), ("a", "b")))
        )
    with pytest.raises(OperationDomainValidationError):
        _operation("topology.simplicial_complex.canonicalize").run(
            SimplicialComplexRequest(
                vertices=("a", "b", "isolated"), facets=(("a", "b"),)
            )
        )


def test_chain_and_homology_requests_validate_prime_semantics() -> None:
    complex_ = _canonical_complex(("a", "b"), (("a", "b"),))

    with pytest.raises(OperationDomainValidationError):
        _operation("topology.simplicial_complex.chain_complex.compute").run(
            ChainComplexRequest(
                complex=complex_,
                coefficient_ring=ChainCoefficientRing.INTEGER,
                prime=2,
            )
        )
    with pytest.raises(OperationDomainValidationError):
        _operation("topology.simplicial_complex.chain_complex.compute").run(
            ChainComplexRequest(
                complex=complex_,
                coefficient_ring=ChainCoefficientRing.PRIME_FIELD,
                prime=9,
            )
        )
    with pytest.raises(OperationDomainValidationError):
        _operation("topology.simplicial_homology.compute").run(
            SimplicialHomologyRequest(complex=complex_, prime=15)
        )


def test_false_face_closure_is_rejected_at_native_admission() -> None:
    source = _canonical_complex(("a", "b", "c"), (("a", "b", "c"),))
    payload = source.model_dump()
    payload["faces_by_dimension"] = (
        payload["faces_by_dimension"][0],
        {"dimension": 1, "faces": (("a", "b"), ("a", "c"))},
        payload["faces_by_dimension"][2],
    )
    payload["f_vector"] = (3, 2, 1)
    payload["closure_size"] = 6
    payload["complex_digest"] = simplicial_complex_digest(
        vertices=payload["vertices"],
        maximal_simplices=payload["maximal_simplices"],
        faces_by_dimension=tuple(
            type(source.faces_by_dimension[index]).model_validate(item)
            for index, item in enumerate(payload["faces_by_dimension"])
        ),
        dimension=payload["dimension"],
        f_vector=payload["f_vector"],
        closure_size=payload["closure_size"],
    )
    malformed = FiniteSimplicialComplex.model_validate(payload)
    with pytest.raises(OperationDomainValidationError):
        _operation("topology.simplicial_complex.chain_complex.compute").run(
            ChainComplexRequest(complex=malformed)
        )


def test_canonical_complex_composes_as_the_authoritative_object() -> None:
    request = SimplicialComplexRequest(
        vertices=("c", "a", "b"),
        facets=(("b", "a"), ("c", "b"), ("c", "a")),
    )
    canonical = _operation("topology.simplicial_complex.canonicalize").run(request)
    complex_ = canonical.complex

    chain_operation = _operation("topology.simplicial_complex.chain_complex.compute")
    chain = chain_operation.run(
        ChainComplexRequest(
            complex=complex_,
            coefficient_ring=ChainCoefficientRing.PRIME_FIELD,
            prime=2,
        )
    )

    homology_operation = _operation("topology.simplicial_homology.compute")
    homology = homology_operation.run(
        SimplicialHomologyRequest(complex=complex_, prime=2)
    )

    assert chain.complex_digest == complex_.complex_digest
    assert homology.complex_digest == complex_.complex_digest
    assert tuple(group.betti_number for group in homology.groups) == (1, 1)


def test_integral_homology_runs_through_the_public_operation() -> None:
    complex_ = _canonical_complex(("a", "b", "c"), (("a", "b"), ("a", "c"), ("b", "c")))
    operation = _operation("topology.simplicial_homology.integral.compute")

    result = operation.run(IntegralSimplicialHomologyRequest(complex=complex_))

    assert result.complex_digest == complex_.complex_digest
    assert result.homology.coefficient_ring is CoefficientRing.INTEGER
    assert tuple(group.free_rank for group in _integral_groups(result.homology)) == (
        1,
        1,
    )


def test_integral_homology_admits_one_tetrahedron() -> None:
    """The canonical simplex contraction has H_0 = ZZ and no higher homology."""

    complex_ = _canonical_complex(
        ("a", "b", "c", "d"),
        (("a", "b", "c", "d"),),
    )
    operation = _operation("topology.simplicial_homology.integral.compute")

    result = operation.run(IntegralSimplicialHomologyRequest(complex=complex_))

    groups = _integral_groups(result.homology)
    assert tuple(
        (group.free_rank, group.torsion_invariant_factors) for group in groups
    ) == ((1, ()), (0, ()), (0, ()), (0, ()))
    chain = result.homology.complex
    independent_betti = tuple(
        chain.basis_sizes[index]
        - (_rational_rank(chain.differential_matrices[index - 1]) if index > 0 else 0)
        - (
            _rational_rank(chain.differential_matrices[index])
            if index < len(chain.differential_matrices)
            else 0
        )
        for index in range(len(chain.basis_sizes))
    )
    assert tuple(group.free_rank for group in groups) == independent_betti


def test_chain_bounds_are_checked_after_materialization_but_before_computation() -> (
    None
):
    vertices = tuple(f"v{index}" for index in range(64))
    facets = tuple(
        tuple(f"v{start + offset}" for offset in range(8)) for start in range(0, 64, 8)
    )
    complex_ = _canonical_complex(vertices, facets)

    assert complex_.closure_size == 8 * (
        2**8 - 1
    )  # 8 simplices, each closing to 2^8-1 faces
    with pytest.raises(ValueError):
        _operation("topology.simplicial_homology.compute").run(
            SimplicialHomologyRequest(complex=complex_, prime=2)
        )


def test_inline_homology_rejects_basis_that_exceeds_its_inline_budget() -> None:
    vertices = tuple(f"v{index}" for index in range(64))
    edges = (
        *((f"v{index}", f"v{(index + 1) % 64}") for index in range(64)),
        ("v0", "v2"),
    )
    complex_ = _canonical_complex(vertices, edges)

    with pytest.raises(ValueError):
        _operation("topology.simplicial_homology.compute").run(
            SimplicialHomologyRequest(complex=complex_, prime=2)
        )


def test_integral_homology_chain_groups_derive_from_certificate_dimension() -> None:
    """Every integral-homology certificate matrix is a ``IntegerMatrix``
    bounded at ``MAX_CERTIFIED_SNF_DIMENSION`` = 32, so a chain group of 33
    simplices must be rejected at admission rather than fail construction."""
    too_many_vertices = tuple(f"v{index}" for index in range(33))
    vertex_complex = _canonical_complex(
        too_many_vertices,
        tuple((vertex,) for vertex in too_many_vertices),
    )
    assert max(vertex_complex.f_vector) == 33
    with pytest.raises(ValueError):
        _operation("topology.simplicial_homology.integral.compute").run(
            IntegralSimplicialHomologyRequest(complex=vertex_complex)
        )


def test_integral_homology_certificate_boundary_runs_the_public_operation() -> None:
    """32 isolated vertices sit exactly on the certificate-dimension boundary:
    the public operation returns a typed result whose H_0 carries one free
    generator per component instead of failing result construction."""
    vertices = tuple(f"v{index}" for index in range(32))
    complex_ = _canonical_complex(vertices, tuple((vertex,) for vertex in vertices))
    operation = _operation("topology.simplicial_homology.integral.compute")

    result = operation.run(IntegralSimplicialHomologyRequest(complex=complex_))

    group = _integral_groups(result.homology)[0]
    assert group.free_rank == 32
    assert len(group.free_generators) == 32


def test_stale_complex_digest_reports_field_level_loc() -> None:
    """A stale ``complex_digest`` must produce a Pydantic error whose ``loc``
    targets the ``complex_digest`` field (not a model-level ``()``), so the
    enrichment helper can surface ``complex/complex_digest`` to the agent.
    """

    good = _canonical_complex(("a", "b", "c"), (("a", "b"), ("b", "c")))
    bad_payload = good.model_dump(mode="python")
    bad_payload["complex_digest"] = "sha256:" + "0" * 64
    with pytest.raises(ValidationError) as exc_info:
        FiniteSimplicialComplex.model_validate(bad_payload)
    errors = exc_info.value.errors()
    assert len(errors) == 1
    assert errors[0]["loc"] == ("complex_digest",)
    assert errors[0]["type"] == "topology.require_digest_binds_canonical_complex_1"
    assert "complex_digest" in errors[0]["msg"]
