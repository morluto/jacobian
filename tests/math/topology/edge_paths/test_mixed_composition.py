"""Mixed based-map composition retains its geometric carrier through consumers."""

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.topology._models import FiniteSimplicialComplex, canonical_complex
from jacobian.math.topology.cohomology.operations._models import SimplicialMap
from jacobian.math.topology.edge_paths._models import (
    FundamentalGroupBasepointChangeRequest,
    FundamentalGroupMapRequest,
    FundamentalGroupMapResult,
    PresentationBasepointChangePath,
    PresentationMapCompositionRequest,
    PresentationTransportedSimplicialMap,
)
from jacobian.math.topology.edge_paths.presentation_maps import (
    change_fundamental_group_basepoint,
    compose_fundamental_group_maps,
    induced_fundamental_group_map,
)


def _circle(labels: tuple[str, str, str]) -> FiniteSimplicialComplex:
    a, b, c = labels
    return canonical_complex(labels, ((a, b), (a, c), (b, c)))


def _path(
    complex_: FiniteSimplicialComplex, vertices: tuple[str, ...]
) -> FundamentalGroupMapResult:
    return change_fundamental_group_basepoint(
        FundamentalGroupBasepointChangeRequest(
            path=PresentationBasepointChangePath(
                complex=complex_,
                source_base_vertex=vertices[0],
                target_base_vertex=vertices[-1],
                path_vertices=vertices,
            )
        )
    )


def _map(
    source: FiniteSimplicialComplex,
    target: FiniteSimplicialComplex,
    vertices: tuple[str, ...],
    source_base: str,
    target_base: str,
) -> FundamentalGroupMapResult:
    return induced_fundamental_group_map(
        FundamentalGroupMapRequest(
            map=SimplicialMap(source=source, target=target, vertex_map=vertices),
            source_base_vertex=source_base,
            target_base_vertex=target_base,
        )
    )


def _compose(
    first: FundamentalGroupMapResult, second: FundamentalGroupMapResult
) -> FundamentalGroupMapResult:
    result = compose_fundamental_group_maps(
        PresentationMapCompositionRequest(first=first, second=second)
    )
    assert FundamentalGroupMapResult.model_validate(result.model_dump()) == result
    decoded = FundamentalGroupMapResult.model_validate_json(result.model_dump_json())
    assert decoded == result
    return decoded


@pytest.mark.parametrize("distinct", (False, True))
@pytest.mark.parametrize("path_first", (False, True))
def test_mixed_composition_round_trips_and_composes_again(
    distinct: bool, path_first: bool
) -> None:
    source = _circle(("a", "b", "c"))
    target = _circle(("x", "y", "z")) if distinct else source
    x, y, _ = target.vertices
    if path_first:
        first = _path(source, ("a", "b"))
        second = _map(source, target, target.vertices, "b", y)
    else:
        first = _map(source, target, target.vertices, "a", x)
        second = _path(target, (x, y))
    result = _compose(first, second)
    assert result.source_presentation.base_vertex == "a"
    assert result.target_presentation.base_vertex == y
    # The canonical spanning trees at a and b orient the free generator oppositely.
    assert result.abelianization_map.entries == ((-1,),)
    identity = _map(target, target, target.vertices, y, y)
    assert _compose(result, identity).generator_images == result.generator_images
    source_identity = _map(source, source, source.vertices, "a", "a")
    assert _compose(source_identity, result).generator_images == result.generator_images


def test_mixed_composition_maps_collapsed_path_edges() -> None:
    source = _circle(("a", "b", "c"))
    target = canonical_complex(("x",), (("x",),))
    path = _path(source, ("a", "b", "c"))
    constant = _map(source, target, ("x", "x", "x"), "c", "x")
    result = _compose(path, constant)
    assert all(not word.letters for word in result.generator_images)


def test_mixed_composition_preserves_nontrivial_basepoint_conjugation() -> None:
    source = canonical_complex(
        ("a", "b", "c", "d", "e"),
        (("a", "b"), ("b", "c"), ("a", "c"), ("a", "d"), ("d", "e"), ("a", "e")),
    )
    loop = _path(source, ("a", "b", "c", "a"))
    identity = _map(source, source, source.vertices, "a", "a")
    result = _compose(loop, identity)
    # The second free generator is conjugated by the first loop, not fixed.
    letters = tuple(
        (letter.generator, letter.exponent)
        for letter in result.generator_images[1].letters
    )
    assert letters == ((0, -1), (1, 1), (0, 1))
    twice = _compose(result, result)
    letters = tuple(
        (letter.generator, letter.exponent)
        for letter in twice.generator_images[1].letters
    )
    assert letters == ((0, -1), (0, -1), (1, 1), (0, 1), (0, 1))


def test_mixed_composition_rejects_changed_enclosing_basepoint() -> None:
    source = _circle(("a", "b", "c"))
    result = _compose(
        _path(source, ("a", "b")), _map(source, source, source.vertices, "b", "b")
    )
    payload = result.model_dump()
    payload["source_presentation"] = result.target_presentation.model_dump()
    with pytest.raises(ValidationError):
        FundamentalGroupMapResult.model_validate(payload)


@pytest.mark.parametrize("first_edges", (64, 66))
def test_mixed_composition_retains_the_path_output_envelope(first_edges: int) -> None:
    source = _circle(("a", "b", "c"))
    first = _path(
        source, tuple("a" if i % 2 == 0 else "b" for i in range(first_edges + 1))
    )
    identity = _map(source, source, source.vertices, "a", "a")
    transported = _compose(first, identity)
    second = _path(source, tuple("a" if i % 2 == 0 else "b" for i in range(65)))
    if first_edges == 64:
        result = _compose(transported, second)
        assert isinstance(result.map, PresentationTransportedSimplicialMap)
        assert len(result.map.basepoint_path.path_vertices) == 129
    else:
        with pytest.raises(OperationResourceAdmissionError, match="word bound"):
            _compose(transported, second)


def test_native_composition_rechecks_forged_transported_path_binding() -> None:
    source = _circle(("a", "b", "c"))
    result = _compose(
        _path(source, ("a", "b")), _map(source, source, source.vertices, "b", "b")
    )
    assert isinstance(result.map, PresentationTransportedSimplicialMap)
    forged = result.model_copy(
        update={
            "map": result.map.model_copy(
                update={"basepoint_path": _path(source, ("c", "b")).map}
            )
        }
    )
    identity = _map(source, source, source.vertices, "b", "b")
    request = PresentationMapCompositionRequest.model_construct(
        first=forged, second=identity
    )
    with pytest.raises(OperationDomainValidationError, match="basepoint"):
        compose_fundamental_group_maps(request)


@pytest.mark.parametrize("field", ("basepoint_path", "simplicial_map"))
@pytest.mark.parametrize("replacement", (None, 1, {}, ()))
def test_transported_carrier_rejects_malformed_nested_types(
    field: str, replacement: object
) -> None:
    source = _circle(("a", "b", "c"))
    result = _compose(
        _path(source, ("a", "b")), _map(source, source, source.vertices, "b", "b")
    )
    assert isinstance(result.map, PresentationTransportedSimplicialMap)
    malformed = result.map.model_copy(update={field: replacement})
    forged = result.model_copy(update={"map": malformed})
    identity = _map(source, source, source.vertices, "b", "b")
    with pytest.raises(ValidationError):
        PresentationMapCompositionRequest(first=forged, second=identity)
    request = PresentationMapCompositionRequest.model_construct(
        first=forged, second=identity
    )
    with pytest.raises(OperationDomainValidationError, match="typed"):
        compose_fundamental_group_maps(request)


@pytest.mark.parametrize(
    "member, field",
    (
        ("simplicial_map", "source"),
        ("simplicial_map", "target"),
        ("simplicial_map", "vertex_map"),
        ("basepoint_path", "complex"),
        ("basepoint_path", "path_vertices"),
        ("basepoint_path", "source_base_vertex"),
    ),
)
def test_transported_carrier_rejects_malformed_member_fields(
    member: str, field: str
) -> None:
    source = _circle(("a", "b", "c"))
    result = _compose(
        _path(source, ("a", "b")), _map(source, source, source.vertices, "b", "b")
    )
    assert isinstance(result.map, PresentationTransportedSimplicialMap)
    malformed = getattr(result.map, member).model_copy(update={field: None})
    forged = result.model_copy(
        update={"map": result.map.model_copy(update={member: malformed})}
    )
    identity = _map(source, source, source.vertices, "b", "b")
    with pytest.raises(ValidationError):
        PresentationMapCompositionRequest(first=forged, second=identity)
    request = PresentationMapCompositionRequest.model_construct(
        first=forged, second=identity
    )
    with pytest.raises(OperationDomainValidationError, match="typed"):
        compose_fundamental_group_maps(request)


@pytest.mark.parametrize("member", ("basepoint_path", "simplicial_map"))
@pytest.mark.parametrize(
    "field", ("maximal_simplices", "faces_by_dimension", "faces", "f_vector")
)
def test_composition_bounds_retained_complexes_before_serialization(
    member: str, field: str
) -> None:
    source = _circle(("a", "b", "c"))
    result = _compose(
        _path(source, ("a", "b")), _map(source, source, source.vertices, "b", "b")
    )
    assert isinstance(result.map, PresentationTransportedSimplicialMap)
    if field == "faces":
        group = source.faces_by_dimension[0].model_copy(
            update={"faces": (("a",),) * 2049}
        )
        malformed_complex = source.model_copy(update={"faces_by_dimension": (group,)})
    else:
        bound = 128 if field == "maximal_simplices" else 8
        malformed_complex = source.model_copy(
            update={field: (getattr(source, field)[0],) * (bound + 1)}
        )
    inner_field = "complex" if member == "basepoint_path" else "source"
    malformed = getattr(result.map, member).model_copy(
        update={inner_field: malformed_complex}
    )
    forged = result.model_copy(
        update={"map": result.map.model_copy(update={member: malformed})}
    )
    identity = _map(source, source, source.vertices, "b", "b")
    request = PresentationMapCompositionRequest.model_construct(
        first=forged, second=identity
    )
    with pytest.raises(OperationResourceAdmissionError, match="envelope"):
        compose_fundamental_group_maps(request)


@pytest.mark.parametrize("field", ("source_presentation", "target_presentation"))
def test_composition_bounds_presentation_complexes_before_comparison(
    field: str,
) -> None:
    source = _circle(("a", "b", "c"))
    identity = _map(source, source, source.vertices, "a", "a")
    malformed = source.model_copy(update={"maximal_simplices": (("a",),) * 129})
    presentation = getattr(identity, field).model_copy(update={"complex": malformed})
    first = identity.model_copy(update={field: presentation})
    request = PresentationMapCompositionRequest.model_construct(
        first=first, second=identity
    )
    with pytest.raises(OperationResourceAdmissionError, match="envelope"):
        compose_fundamental_group_maps(request)


def test_transported_path_subclasses_are_refused_before_dumping() -> None:
    class ExtendedPath(PresentationBasepointChangePath):
        payload: tuple[int, ...] = ()

    source = _circle(("a", "b", "c"))
    result = _compose(
        _path(source, ("a", "b")), _map(source, source, source.vertices, "b", "b")
    )
    assert isinstance(result.map, PresentationTransportedSimplicialMap)
    path = ExtendedPath.model_construct(
        **result.map.basepoint_path.__dict__, payload=(1,) * 10000
    )
    forged = result.model_copy(
        update={"map": result.map.model_copy(update={"basepoint_path": path})}
    )
    identity = _map(source, source, source.vertices, "b", "b")
    request = PresentationMapCompositionRequest.model_construct(
        first=forged, second=identity
    )
    with pytest.raises(OperationDomainValidationError, match="exact typed carrier"):
        compose_fundamental_group_maps(request)


def test_native_producers_reject_missing_request_fields() -> None:
    with pytest.raises(OperationDomainValidationError, match="exact typed carrier"):
        induced_fundamental_group_map(FundamentalGroupMapRequest.model_construct())
    with pytest.raises(OperationDomainValidationError, match="exact typed carrier"):
        change_fundamental_group_basepoint(
            FundamentalGroupBasepointChangeRequest.model_construct()
        )
