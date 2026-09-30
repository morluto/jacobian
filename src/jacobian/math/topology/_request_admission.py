"""Request-time admission for finite simplicial topology operations."""

from __future__ import annotations

from collections.abc import Callable

from pydantic_core import PydanticCustomError

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.topology._models import (
    MAX_TOPOLOGY_DIMENSION,
    MAX_TOPOLOGY_FACES,
    MAX_TOPOLOGY_FACETS,
    MAX_TOPOLOGY_VERTICES,
    FacesInDimension,
    FiniteSimplicialComplex,
    SimplicialComplexRequest,
    _require_request_complex,
    face_closure,
)


def run_topology_admission[T](
    admission: Callable[[], T], *, location: tuple[str | int, ...]
) -> T:
    """Normalize owner semantic failures at the public operation boundary."""

    try:
        return admission()
    except (OperationDomainValidationError, OperationResourceAdmissionError):
        raise
    except PydanticCustomError as exc:
        raise OperationDomainValidationError(
            location=location,
            code=exc.type,
            message=exc.message(),
        ) from exc
    except ValueError as exc:
        raise OperationDomainValidationError(
            location=location,
            code="topology.request_not_admitted",
            message=str(exc),
        ) from exc


def require_complex_admission(request: SimplicialComplexRequest) -> None:
    """Check semantic complex bounds immediately before a kernel runs."""

    def admit() -> None:
        if any(
            not 1 <= len(facet) <= MAX_TOPOLOGY_DIMENSION + 1
            for facet in request.facets
        ):
            raise ValueError(
                "each facet must contain between 1 and "
                f"{MAX_TOPOLOGY_DIMENSION + 1} vertices"
            )
        _require_request_complex(request.vertices, request.facets)

    run_topology_admission(admit, location=("facets",))


def _shape_domain(field: str) -> OperationDomainValidationError:
    return OperationDomainValidationError(
        location=("complex", field),
        code="topology.complex_shape",
        message=f"canonical complex {field} has an invalid native shape",
    )


def _shape_sequence(
    value: object, bound: int, field: str
) -> tuple[object, ...] | list[object]:
    if not isinstance(value, (tuple, list)) or type(value) not in (tuple, list):
        raise _shape_domain(field)
    if len(value) > bound:
        raise OperationResourceAdmissionError(
            location=("complex", field),
            code="topology.complex_shape_bound",
            message=f"canonical complex {field} exceeds its {bound}-element envelope",
        )
    return value


def _shape_labels(value: object, bound: int, field: str) -> None:
    labels = _shape_sequence(value, bound, field)
    if any(type(label) is not str for label in labels):
        raise _shape_domain(field)


def require_canonical_complex_shape(value: object) -> None:
    """Bound every native complex container before recursive dump/decoding.

    This only establishes finite scalar/container shape. Canonical face order,
    closure and other semantic invariants remain owned by model validation and
    canonical-complex admission.
    """
    if type(value) is not FiniteSimplicialComplex:
        raise _shape_domain("type")
    fields = FiniteSimplicialComplex.model_fields
    if len(value.__dict__) != len(fields) or any(
        key not in fields for key in value.__dict__
    ):
        raise _shape_domain("fields")
    _shape_labels(value.vertices, MAX_TOPOLOGY_VERTICES, "vertices")
    facets = _shape_sequence(
        value.maximal_simplices, MAX_TOPOLOGY_FACETS, "maximal_simplices"
    )
    for facet in facets:
        _shape_labels(facet, MAX_TOPOLOGY_DIMENSION + 1, "facet")
    groups = _shape_sequence(
        value.faces_by_dimension, MAX_TOPOLOGY_DIMENSION + 1, "faces_by_dimension"
    )
    face_count = 0
    for group in groups:
        if (
            type(group) is not FacesInDimension
            or len(group.__dict__) != 2
            or set(group.__dict__) != {"dimension", "faces"}
            or type(group.dimension) is not int
        ):
            raise _shape_domain("face group")
        faces = _shape_sequence(group.faces, MAX_TOPOLOGY_FACES, "faces")
        face_count += len(faces)
        if face_count > MAX_TOPOLOGY_FACES:
            raise OperationResourceAdmissionError(
                location=("complex", "faces"),
                code="topology.complex_shape_bound",
                message="canonical complex total faces exceed the admitted envelope",
            )
        for face in faces:
            _shape_labels(face, MAX_TOPOLOGY_DIMENSION + 1, "simplex")
    counts = _shape_sequence(value.f_vector, MAX_TOPOLOGY_DIMENSION + 1, "f_vector")
    if (
        any(type(count) is not int for count in counts)
        or type(value.dimension) is not int
        or type(value.closure_size) is not int
    ):
        raise _shape_domain("face counts")
    if (
        not isinstance(value.orientation_convention, str)
        or type(value.empty_simplex_stored) is not bool
    ):
        raise _shape_domain("conventions")


def require_canonical_complex_admission(complex_: FiniteSimplicialComplex) -> None:
    """Establish every authored canonical invariant before a consumer runs."""

    # Nested model instances and model_copy(update=...) can bypass Pydantic
    # validation. Revalidate the complete model contract before face admission.
    FiniteSimplicialComplex.model_validate(
        complex_.model_dump(mode="python"), strict=True
    )
    closure = face_closure(complex_.maximal_simplices)
    expected_faces = tuple(tuple(sorted(faces)) for faces in closure)
    actual_faces = tuple(
        tuple(sorted(item.faces)) for item in complex_.faces_by_dimension
    )
    if actual_faces != expected_faces:
        raise ValueError("canonical complex face closure is incomplete or inconsistent")
    if complex_.f_vector != tuple(len(faces) for faces in closure):
        raise ValueError("canonical complex f-vector does not match its face closure")


__all__ = [
    "require_canonical_complex_admission",
    "require_canonical_complex_shape",
    "require_complex_admission",
    "run_topology_admission",
]
