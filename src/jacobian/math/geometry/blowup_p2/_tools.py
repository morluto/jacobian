"""Public exact blow-up P2 divisor operations."""

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.geometry.blowup_p2._models import (
    AdjunctionProfile,
    AdjunctionRequest,
    BlowupDivisorClass,
    BlowupP2Surface,
    BlowupSurfaceRequest,
    CanonicalClassRequest,
    DivisorClassRequest,
    IntersectionRequest,
    IntersectionResult,
)
from jacobian.math.geometry.blowup_p2.operations import (
    adjunction_profile,
    canonical_class,
    construct_divisor_class,
    construct_surface,
    intersect_classes,
)

_SURFACE: dict[str, object] = {"points": []}

TOOLS = (
    MathTool(
        operation_id="algebraic_geometry.blowup_p2.surface.construct",
        title="Construct a labelled rational blow-up of P2",
        description="Canonicalize a finite labelled rational projective point set as the parent of a finite blow-up; the points must be distinct and nonzero.",
        request_type=BlowupSurfaceRequest,
        result_type=BlowupP2Surface,
        run=lambda request: construct_surface(request.points),
        tags=("algebraic-geometry", "blow-up", "projective"),
        discovery_terms=("blow-up P2", "labelled rational blowup surface"),
        examples=(
            OperationExample(
                name="empty_blowup",
                description="Construct the unblown P2 parent; the point list may be empty.",
                input=_SURFACE,
            ),
        ),
    ),
    MathTool(
        operation_id="algebraic_geometry.blowup_p2.strict_transform_class.compute",
        title="Construct a labelled blow-up P2 divisor class",
        description="Return D=dH-sum m_i E_i with explicit multiplicities bound to one labelled blow-up point axis; multiplicities are supplied, never inferred from a first jet.",
        request_type=DivisorClassRequest,
        result_type=BlowupDivisorClass,
        run=lambda request: construct_divisor_class(
            request.surface, request.degree, request.multiplicities
        ),
        tags=("algebraic-geometry", "blow-up", "divisor"),
        discovery_terms=("strict transform divisor class", "blowup divisor class"),
        examples=(
            OperationExample(
                name="line_class",
                description="Construct the line class H on P2; the multiplicity axis is empty for the unblown parent.",
                input={"surface": _SURFACE, "degree": "1", "multiplicities": []},
            ),
        ),
    ),
    MathTool(
        operation_id="algebraic_geometry.blowup_p2.intersection.compute",
        title="Compute exact intersection of labelled blow-up divisor classes",
        description="Compute D.D'=de-sum m_i n_i with one exact contribution per labelled exceptional divisor; cross-parent classes are rejected.",
        request_type=IntersectionRequest,
        result_type=IntersectionResult,
        run=lambda request: intersect_classes(request.left, request.right),
        tags=("algebraic-geometry", "blow-up", "intersection"),
        discovery_terms=("blow-up intersection form", "divisor intersection P2 blowup"),
        examples=(
            OperationExample(
                name="line_self_intersection",
                description="Compute H.H on P2; both classes must use the identical labelled parent.",
                input={
                    "left": {"surface": _SURFACE, "degree": "1", "multiplicities": []},
                    "right": {"surface": _SURFACE, "degree": "1", "multiplicities": []},
                },
            ),
        ),
    ),
    MathTool(
        operation_id="algebraic_geometry.blowup_p2.canonical_class.compute",
        title="Compute the canonical class of a finite blow-up of P2",
        description="Return K=-3H+sum E_i as the exact class bound to the labelled blow-up parent.",
        request_type=CanonicalClassRequest,
        result_type=BlowupDivisorClass,
        run=lambda request: canonical_class(request.surface),
        tags=("algebraic-geometry", "blow-up", "canonical-class"),
        discovery_terms=("canonical divisor blowup P2",),
        examples=(
            OperationExample(
                name="canonical_p2",
                description="Compute the canonical class of P2; the labelled blow-up point set is empty.",
                input={"surface": _SURFACE},
            ),
        ),
    ),
    MathTool(
        operation_id="algebraic_geometry.blowup_p2.adjunction_profile.compute",
        title="Compute the exact adjunction profile of a blow-up divisor class",
        description="Return D^2, D.K, D.(D+K), and the integral arithmetic genus 1+(D^2+D.K)/2 for one labelled finite blow-up.",
        request_type=AdjunctionRequest,
        result_type=AdjunctionProfile,
        run=lambda request: adjunction_profile(request.divisor),
        tags=("algebraic-geometry", "blow-up", "adjunction"),
        discovery_terms=("adjunction formula divisor class", "arithmetic genus blowup"),
        examples=(
            OperationExample(
                name="line_adjunction",
                description="Compute the adjunction profile of H on P2; the divisor class uses the identical empty exceptional axis.",
                input={
                    "divisor": {
                        "surface": _SURFACE,
                        "degree": "1",
                        "multiplicities": [],
                    }
                },
            ),
        ),
    ),
)
__all__ = ["TOOLS"]
