"""Owner-local admission decisions for built-in math operations."""

from __future__ import annotations

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.geometry._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "geometry.points.compute.circle_inversion",
        AdmissionDecision.KEEP,
        "one exact rational planar point transform with material leverage over bespoke coordinate algebra",
    ),
    OperationAdmission(
        "geometry.line.compute.projection",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "geometry.lines.compute.intersection",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "geometry.lines.decide.parallel",
        AdmissionDecision.DROP,
        "elementary exact formula without material leverage over direct Python",
    ),
    OperationAdmission(
        "geometry.lines.decide.perpendicular",
        AdmissionDecision.DROP,
        "elementary exact formula without material leverage over direct Python",
    ),
    OperationAdmission(
        "geometry.points.compute.convex_hull",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "geometry.points.compute.squared_distance",
        AdmissionDecision.DROP,
        "elementary exact formula without material leverage over direct Python",
    ),
    OperationAdmission(
        "geometry.points.decide.collinear",
        AdmissionDecision.DROP,
        "elementary exact formula without material leverage over direct Python",
    ),
    OperationAdmission(
        "geometry.points.decide.concyclic",
        AdmissionDecision.KEEP,
        "distinct exact bounded predicate or candidate check with typed semantics",
    ),
    OperationAdmission(
        "geometry.polygon.compute.signed_area",
        AdmissionDecision.DROP,
        "elementary exact formula without material leverage over direct Python",
    ),
    OperationAdmission(
        "geometry.polygon.point.classify",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "geometry.polygon.simple.decide",
        AdmissionDecision.KEEP,
        "distinct exact bounded predicate or candidate check with typed semantics",
    ),
    OperationAdmission(
        "geometry.polygon.triangulation.minimum_weight.compute",
        AdmissionDecision.KEEP,
        "distinct exact or explicitly bounded search outcome with material computational leverage",
    ),
    OperationAdmission(
        "geometry.segment.compute.midpoint",
        AdmissionDecision.DROP,
        "elementary exact formula without material leverage over direct Python",
    ),
    OperationAdmission(
        "geometry.segments.intersection.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "geometry.triangle.compute.centroid",
        AdmissionDecision.DROP,
        "elementary exact formula without material leverage over direct Python",
    ),
    OperationAdmission(
        "geometry.triangle.compute.circumcircle",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "geometry.triangle.compute.orientation",
        AdmissionDecision.DROP,
        "elementary exact formula without material leverage over direct Python",
    ),
    OperationAdmission(
        "geometry.points.general_position.search",
        AdmissionDecision.KEEP,
        "distinct exact or explicitly bounded search outcome with material computational leverage",
    ),
    OperationAdmission(
        "geometry.points.circumradius_profile.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
