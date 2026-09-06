"""Request-time admission for finite simplicial topology operations."""

from __future__ import annotations

from collections.abc import Callable

from pydantic_core import PydanticCustomError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.topology._models import (
    MAX_TOPOLOGY_DIMENSION,
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
    except OperationDomainValidationError:
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


def require_canonical_complex_admission(complex_: FiniteSimplicialComplex) -> None:
    """Establish the authored face closure before a canonical consumer runs."""

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
    "require_complex_admission",
    "run_topology_admission",
]
