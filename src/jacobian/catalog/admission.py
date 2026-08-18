"""Explicit admission decisions for the frozen public math-operation basis.

This ledger is intentionally exhaustive. New candidate declarations must receive a
reviewed decision before they can enter the public catalog.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from jacobian.catalog.models import MathTools


class AdmissionDecision(StrEnum):
    KEEP = "KEEP"
    NATIVE_ONLY = "NATIVE_ONLY"
    SPLIT = "SPLIT"
    DROP = "DROP"
    CONTRACT_FIX = "CONTRACT_FIX"


@dataclass(frozen=True, slots=True)
class OperationAdmission:
    operation_id: str
    decision: AdmissionDecision
    rationale: str
    native_symbol: str | None = None


REVIEWED_BASE_REVISION = "61589543bbbff546edbc51d34a07887982fa4ad6"

OPERATION_ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "additive.direct_sum_predicate.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "additive.energy.compute",
        AdmissionDecision.DROP,
        "cheap deterministic projection of additive.representation_profile.compute",
    ),
    OperationAdmission(
        "additive.representation_profile.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "additive.sumset_cardinality.compute",
        AdmissionDecision.DROP,
        "cheap deterministic projection of additive.representation_profile.compute",
    ),
    OperationAdmission(
        "algebraic_number.compare",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "analysis.real_function.point_enclosure.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "arithmetic.dirichlet_convolution.compute",
        AdmissionDecision.KEEP,
        "reusable typed mathematical construction or transformation with a distinct discovery intent",
    ),
    OperationAdmission(
        "arithmetic.dirichlet_inverse.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "arithmetic.mobius_transform.compute",
        AdmissionDecision.KEEP,
        "reusable typed mathematical construction or transformation with a distinct discovery intent",
    ),
    OperationAdmission(
        "arithmetic.real_quadratic.order.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "arithmetic.summatory_function.compute",
        AdmissionDecision.DROP,
        "ordinary finite prefix sum without catalog-level leverage",
    ),
    OperationAdmission(
        "arithmetic_dynamics.cycle.multiplier.compute",
        AdmissionDecision.KEEP,
        "exact multiplier spectrum of a rational map at a cycle",
    ),
    OperationAdmission(
        "arithmetic_dynamics.dynatomic_polynomial.compute",
        AdmissionDecision.KEEP,
        "exact dynatomic polynomial for a bounded iteration depth",
    ),
    OperationAdmission(
        "arithmetic_dynamics.finite_field.functional_graph.compute",
        AdmissionDecision.KEEP,
        "exact functional graph of a map over a finite field",
    ),
    OperationAdmission(
        "arithmetic_dynamics.map.iterate.compute",
        AdmissionDecision.KEEP,
        "exact rational map iterate over an explicit domain",
    ),
    OperationAdmission(
        "arithmetic_dynamics.point.orbit.compute",
        AdmissionDecision.KEEP,
        "exact orbit computation with typed periodicity",
    ),
    OperationAdmission(
        "boolean.erasure_noise.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "boolean.fourier.walsh_transform.compute",
        AdmissionDecision.KEEP,
        "reusable typed mathematical construction or transformation with a distinct discovery intent",
    ),
    OperationAdmission(
        "boolean.fourier_spectrum.compute",
        AdmissionDecision.DROP,
        "duplicate of boolean.fourier.walsh_transform.compute",
    ),
    OperationAdmission(
        "boolean.multilinear_extension.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "boolean.truth_table.compute",
        AdmissionDecision.DROP,
        "echoes a caller-supplied truth table without a new mathematical outcome",
    ),
    OperationAdmission(
        "code.covering_radius.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "code.minimum_distance.compute",
        AdmissionDecision.KEEP,
        "distinct exact or explicitly bounded search outcome with material computational leverage",
    ),
    OperationAdmission(
        "code.weight_distribution.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "combinatorics.compute.bell",
        AdmissionDecision.NATIVE_ONLY,
        "useful classical number retained without a scalar catalog slot",
        native_symbol="jacobian.math.combinatorics.bell_number",
    ),
    OperationAdmission(
        "combinatorics.compute.bernoulli",
        AdmissionDecision.NATIVE_ONLY,
        "useful classical number retained without a scalar catalog slot",
        native_symbol="jacobian.math.combinatorics.bernoulli_number",
    ),
    OperationAdmission(
        "combinatorics.compute.binomial",
        AdmissionDecision.DROP,
        "ordinary scalar or finite enumeration better authored directly in Python",
    ),
    OperationAdmission(
        "combinatorics.compute.catalan",
        AdmissionDecision.NATIVE_ONLY,
        "useful classical number retained without a scalar catalog slot",
        native_symbol="jacobian.math.combinatorics.catalan_number",
    ),
    OperationAdmission(
        "combinatorics.compute.central_binomial",
        AdmissionDecision.DROP,
        "ordinary scalar or finite enumeration better authored directly in Python",
    ),
    OperationAdmission(
        "combinatorics.compute.compositions",
        AdmissionDecision.DROP,
        "ordinary scalar or finite enumeration better authored directly in Python",
    ),
    OperationAdmission(
        "combinatorics.compute.derangements",
        AdmissionDecision.NATIVE_ONLY,
        "useful classical number retained without a scalar catalog slot",
        native_symbol="jacobian.math.combinatorics.derangement_number",
    ),
    OperationAdmission(
        "combinatorics.compute.double_factorial",
        AdmissionDecision.NATIVE_ONLY,
        "useful classical number retained without a scalar catalog slot",
        native_symbol="jacobian.math.combinatorics.double_factorial",
    ),
    OperationAdmission(
        "combinatorics.compute.factorial",
        AdmissionDecision.DROP,
        "ordinary scalar or finite enumeration better authored directly in Python",
    ),
    OperationAdmission(
        "combinatorics.compute.fibonacci",
        AdmissionDecision.NATIVE_ONLY,
        "useful classical number retained without a scalar catalog slot",
        native_symbol="jacobian.math.combinatorics.fibonacci_number",
    ),
    OperationAdmission(
        "combinatorics.compute.fibonacci_pair",
        AdmissionDecision.DROP,
        "ordinary scalar or finite enumeration better authored directly in Python",
    ),
    OperationAdmission(
        "combinatorics.compute.lucas",
        AdmissionDecision.NATIVE_ONLY,
        "useful classical number retained without a scalar catalog slot",
        native_symbol="jacobian.math.combinatorics.lucas_number",
    ),
    OperationAdmission(
        "combinatorics.compute.motzkin",
        AdmissionDecision.NATIVE_ONLY,
        "useful classical number retained without a scalar catalog slot",
        native_symbol="jacobian.math.combinatorics.motzkin_number",
    ),
    OperationAdmission(
        "combinatorics.compute.multinomial",
        AdmissionDecision.DROP,
        "ordinary scalar or finite enumeration better authored directly in Python",
    ),
    OperationAdmission(
        "combinatorics.compute.partition_number",
        AdmissionDecision.NATIVE_ONLY,
        "useful classical number retained without a scalar catalog slot",
        native_symbol="jacobian.math.combinatorics.partition_number",
    ),
    OperationAdmission(
        "combinatorics.compute.permutations",
        AdmissionDecision.DROP,
        "ordinary scalar or finite enumeration better authored directly in Python",
    ),
    OperationAdmission(
        "combinatorics.compute.stirling_first",
        AdmissionDecision.NATIVE_ONLY,
        "useful classical number retained without a scalar catalog slot",
        native_symbol="jacobian.math.combinatorics.stirling_first",
    ),
    OperationAdmission(
        "combinatorics.compute.stirling_second",
        AdmissionDecision.NATIVE_ONLY,
        "useful classical number retained without a scalar catalog slot",
        native_symbol="jacobian.math.combinatorics.stirling_second",
    ),
    OperationAdmission(
        "combinatorics.conjugate_partition.compute",
        AdmissionDecision.NATIVE_ONLY,
        "useful deterministic helper retained through the supported native API",
        native_symbol="jacobian.math.algebraic_combinatorics.conjugate_partition",
    ),
    OperationAdmission(
        "combinatorics.cyclic_difference_set.extension.decide",
        AdmissionDecision.KEEP,
        "distinct exact bounded predicate or candidate check with typed semantics",
    ),
    OperationAdmission(
        "combinatorics.cyclic_difference_set.perfect.decide",
        AdmissionDecision.KEEP,
        "distinct exact bounded predicate or candidate check with typed semantics",
    ),
    OperationAdmission(
        "combinatorics.enumerate.integer_partitions",
        AdmissionDecision.NATIVE_ONLY,
        "useful finite enumeration retained without a scalar-family catalog slot",
        native_symbol="jacobian.math.combinatorics.integer_partitions",
    ),
    OperationAdmission(
        "combinatorics.generating_function.coefficients.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "combinatorics.hook_length.compute",
        AdmissionDecision.NATIVE_ONLY,
        "useful deterministic helper retained through the supported native API",
        native_symbol="jacobian.math.algebraic_combinatorics.hook_lengths",
    ),
    OperationAdmission(
        "combinatorics.integer_set.sidon.decide",
        AdmissionDecision.KEEP,
        "distinct exact bounded predicate or candidate check with typed semantics",
    ),
    OperationAdmission(
        "combinatorics.recurrence.linear.evaluate",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "combinatorics.recurrence.p_recursive.evaluate",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "combinatorics.recurrence.p_recursive.table_residuals.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "combinatorics.set_function.evaluate",
        AdmissionDecision.DROP,
        "table lookup that merely echoes one caller-owned value",
    ),
    OperationAdmission(
        "combinatorics.set_function.monotonicity",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "combinatorics.set_function.submodularity",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "combinatorics.standard_young_tableaux.count",
        AdmissionDecision.NATIVE_ONLY,
        "useful deterministic helper retained through the supported native API",
        native_symbol="jacobian.math.algebraic_combinatorics.standard_young_tableaux_count",
    ),
    OperationAdmission(
        "convex.max_affine.evaluate",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "convex.max_affine.subdifferential",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "diophantine.continued_fraction.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "diophantine.convergents.compute",
        AdmissionDecision.NATIVE_ONLY,
        "useful deterministic projection of the retained continued-fraction result",
        native_symbol="jacobian.math.diophantine_approximation.convergents",
    ),
    OperationAdmission(
        "diophantine.pell_equation.solve",
        AdmissionDecision.KEEP,
        "distinct exact or explicitly bounded search outcome with material computational leverage",
    ),
    OperationAdmission(
        "discrepancy.theory.eval.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "discrepancy.theory.optimum.compute",
        AdmissionDecision.KEEP,
        "distinct exact or explicitly bounded search outcome with material computational leverage",
    ),
    OperationAdmission(
        "electrical_network.effective_resistance.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "electrical_network.laplacian.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "electrical_network.node_potentials.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "finite_abelian_group.exact_factorization.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "finite_field.direction_rank_ledger.compute",
        AdmissionDecision.NATIVE_ONLY,
        "useful deterministic helper retained through the supported native API",
        native_symbol="jacobian.math.finite_fields.direction_rank_ledger",
    ),
    OperationAdmission(
        "finite_field.linear_map.rank.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "finite_field.orbit_distribution.compute",
        AdmissionDecision.NATIVE_ONLY,
        "useful deterministic helper retained through the supported native API",
        native_symbol="jacobian.math.finite_fields.orbit_distribution",
    ),
    OperationAdmission(
        "finite_field.polynomial_map.collision.analyze",
        AdmissionDecision.NATIVE_ONLY,
        "useful deterministic helper retained through the supported native API",
        native_symbol="jacobian.math.finite_fields.analyze_collisions",
    ),
    OperationAdmission(
        "finite_field.polynomial_map.fibers.compute",
        AdmissionDecision.NATIVE_ONLY,
        "useful deterministic helper retained through the supported native API",
        native_symbol="jacobian.math.finite_fields.fiber_partition",
    ),
    OperationAdmission(
        "finite_field.polynomial_map.permutation.analyze",
        AdmissionDecision.NATIVE_ONLY,
        "useful deterministic helper retained through the supported native API",
        native_symbol="jacobian.math.finite_fields.analyze_permutation",
    ),
    OperationAdmission(
        "finite_field.polynomial_map.table.compute",
        AdmissionDecision.KEEP,
        "reusable typed mathematical construction or transformation with a distinct discovery intent",
    ),
    OperationAdmission(
        "finite_field.projective_line.enumerate",
        AdmissionDecision.KEEP,
        "distinct exact or explicitly bounded search outcome with material computational leverage",
    ),
    OperationAdmission(
        "finite_field.restrict_scalars.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "finite_set.compute.difference",
        AdmissionDecision.DROP,
        "ordinary deterministic set projection better authored directly in Python",
    ),
    OperationAdmission(
        "finite_set.compute.intersection",
        AdmissionDecision.DROP,
        "ordinary deterministic set projection better authored directly in Python",
    ),
    OperationAdmission(
        "finite_set.compute.intersection_cardinality",
        AdmissionDecision.DROP,
        "ordinary deterministic set projection better authored directly in Python",
    ),
    OperationAdmission(
        "finite_set.compute.left_cardinality",
        AdmissionDecision.DROP,
        "ordinary deterministic set projection better authored directly in Python",
    ),
    OperationAdmission(
        "finite_set.compute.symmetric_difference",
        AdmissionDecision.DROP,
        "ordinary deterministic set projection better authored directly in Python",
    ),
    OperationAdmission(
        "finite_set.compute.union",
        AdmissionDecision.DROP,
        "ordinary deterministic set projection better authored directly in Python",
    ),
    OperationAdmission(
        "finite_set.compute.union_cardinality",
        AdmissionDecision.DROP,
        "ordinary deterministic set projection better authored directly in Python",
    ),
    OperationAdmission(
        "finite_set.decide.disjoint",
        AdmissionDecision.DROP,
        "ordinary deterministic set projection better authored directly in Python",
    ),
    OperationAdmission(
        "finite_set.decide.exact_cover",
        AdmissionDecision.DROP,
        "ordinary deterministic set projection better authored directly in Python",
    ),
    OperationAdmission(
        "finite_set.decide.proper_subset",
        AdmissionDecision.DROP,
        "ordinary deterministic set projection better authored directly in Python",
    ),
    OperationAdmission(
        "finite_set.decide.subset",
        AdmissionDecision.DROP,
        "ordinary deterministic set projection better authored directly in Python",
    ),
    OperationAdmission(
        "formal_series.rational.add.compute",
        AdmissionDecision.NATIVE_ONLY,
        "cheap structural projection of the supplied truncated-series value",
        native_symbol="jacobian.math.formal_power_series.add",
    ),
    OperationAdmission(
        "formal_series.rational.compose.compute",
        AdmissionDecision.KEEP,
        "reusable typed mathematical construction or transformation with a distinct discovery intent",
    ),
    OperationAdmission(
        "formal_series.rational.derivative.compute",
        AdmissionDecision.NATIVE_ONLY,
        "cheap structural projection of the supplied truncated-series value",
        native_symbol="jacobian.math.formal_power_series.derivative",
    ),
    OperationAdmission(
        "formal_series.rational.divide.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "formal_series.rational.from_polynomial.compute",
        AdmissionDecision.NATIVE_ONLY,
        "cheap structural projection of the supplied truncated-series value",
        native_symbol="jacobian.math.formal_power_series.from_polynomial",
    ),
    OperationAdmission(
        "formal_series.rational.identity.check",
        AdmissionDecision.NATIVE_ONLY,
        "cheap structural projection of the supplied truncated-series value",
        native_symbol="jacobian.math.formal_power_series.identity_check",
    ),
    OperationAdmission(
        "formal_series.rational.integral_zero_constant.compute",
        AdmissionDecision.NATIVE_ONLY,
        "cheap structural projection of the supplied truncated-series value",
        native_symbol="jacobian.math.formal_power_series.integral_zero_constant",
    ),
    OperationAdmission(
        "formal_series.rational.inverse.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "formal_series.rational.multiply.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "formal_series.rational.power.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "formal_series.rational.reversion.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "formal_series.rational.scalar_multiply.compute",
        AdmissionDecision.NATIVE_ONLY,
        "cheap structural projection of the supplied truncated-series value",
        native_symbol="jacobian.math.formal_power_series.scalar_multiply",
    ),
    OperationAdmission(
        "formal_series.rational.subtract.compute",
        AdmissionDecision.NATIVE_ONLY,
        "cheap structural projection of the supplied truncated-series value",
        native_symbol="jacobian.math.formal_power_series.subtract",
    ),
    OperationAdmission(
        "formal_series.rational.to_polynomial.compute",
        AdmissionDecision.NATIVE_ONLY,
        "cheap structural projection of the supplied truncated-series value",
        native_symbol="jacobian.math.formal_power_series.to_polynomial",
    ),
    OperationAdmission(
        "formal_series.rational.truncate.compute",
        AdmissionDecision.NATIVE_ONLY,
        "cheap structural projection of the supplied truncated-series value",
        native_symbol="jacobian.math.formal_power_series.truncate",
    ),
    OperationAdmission(
        "game.impartial.birthday.compute",
        AdmissionDecision.KEEP,
        "exact birthday rank certificate for a bounded impartial game position",
    ),
    OperationAdmission(
        "game.impartial.grundy_table.compute",
        AdmissionDecision.KEEP,
        "complete Grundy table over a bounded impartial game state space",
    ),
    OperationAdmission(
        "game.subtraction.grundy_prefix.compute",
        AdmissionDecision.KEEP,
        "complete Grundy prefix with periodic tail certificate for a bounded subtraction game",
    ),
    OperationAdmission(
        "game_theory.best_response.compute",
        AdmissionDecision.DROP,
        "misnamed pure maximin row calculation that is not a best response without an opponent strategy",
    ),
    OperationAdmission(
        "game_theory.nash_equilibrium.compute",
        AdmissionDecision.KEEP,
        "exact primal-dual linear programming returns a complete equilibrium witness for every bounded finite zero-sum game",
    ),
    OperationAdmission(
        "geometry.euclidean.angle_equality.compute",
        AdmissionDecision.DROP,
        "elementary exact formula without material leverage over direct Python",
    ),
    OperationAdmission(
        "geometry.euclidean.segment_ratio.compute",
        AdmissionDecision.DROP,
        "elementary exact formula without material leverage over direct Python",
    ),
    OperationAdmission(
        "geometry.euclidean.triangle_similarity.compute",
        AdmissionDecision.DROP,
        "elementary exact formula without material leverage over direct Python",
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
        "geometry.points.distance_graph.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "geometry.points.distance_profile.compute",
        AdmissionDecision.KEEP,
        "one complete exact multiplicity profile of the pairwise-distance multiset",
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
        "geometry.projective_line_arrangement.flats.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
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
        "graph.coloring.k_colorability.decide",
        AdmissionDecision.KEEP,
        "distinct exact bounded predicate or candidate check with typed semantics",
    ),
    OperationAdmission(
        "graph.complement.compute",
        AdmissionDecision.NATIVE_ONLY,
        "useful deterministic helper retained through the supported native API",
        native_symbol="jacobian.math.graphs.complement",
    ),
    OperationAdmission(
        "graph.cut.minimum_st.compute",
        AdmissionDecision.KEEP,
        "distinct exact or explicitly bounded search outcome with material computational leverage",
    ),
    OperationAdmission(
        "graph.decomposition.biconnected_components.compute",
        AdmissionDecision.NATIVE_ONLY,
        "useful projection of the retained block-cut-tree decomposition",
        native_symbol="jacobian.math.graphs.biconnected_components",
    ),
    OperationAdmission(
        "graph.decomposition.block_cut_tree.compute",
        AdmissionDecision.KEEP,
        "reusable typed mathematical construction or transformation with a distinct discovery intent",
    ),
    OperationAdmission(
        "graph.decomposition.bridge_block_tree.compute",
        AdmissionDecision.KEEP,
        "reusable typed mathematical construction or transformation with a distinct discovery intent",
    ),
    OperationAdmission(
        "graph.decomposition.ear.compute",
        AdmissionDecision.KEEP,
        "reusable typed mathematical construction or transformation with a distinct discovery intent",
    ),
    OperationAdmission(
        "graph.directed.acyclic_order.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "graph.directed.condensation.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "graph.directed.reachability.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "graph.directed.scc.compute",
        AdmissionDecision.NATIVE_ONLY,
        "useful projection of the retained condensation DAG construction",
        native_symbol="jacobian.math.graphs.strongly_connected_components",
    ),
    OperationAdmission(
        "graph.distance_matrix.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "graph.domination.minimum.compute",
        AdmissionDecision.KEEP,
        "distinct exact or explicitly bounded search outcome with material computational leverage",
    ),
    OperationAdmission(
        "graph.encoding.graph6.decode.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "graph.flow.maximum.compute",
        AdmissionDecision.KEEP,
        "distinct exact or explicitly bounded search outcome with material computational leverage",
    ),
    OperationAdmission(
        "graph.hamiltonian_path.decide",
        AdmissionDecision.KEEP,
        "distinct exact bounded predicate or candidate check with typed semantics",
    ),
    OperationAdmission(
        "graph.independent_set.maximal.decide",
        AdmissionDecision.KEEP,
        "distinct exact bounded predicate or candidate check with typed semantics",
    ),
    OperationAdmission(
        "graph.independent_set.maximum.compute",
        AdmissionDecision.KEEP,
        "distinct exact or explicitly bounded search outcome with material computational leverage",
    ),
    OperationAdmission(
        "graph.induced_bipartite.maximum.compute",
        AdmissionDecision.KEEP,
        "distinct exact or explicitly bounded search outcome with material computational leverage",
    ),
    OperationAdmission(
        "graph.induced_forest.maximum.compute",
        AdmissionDecision.KEEP,
        "distinct exact or explicitly bounded search outcome with material computational leverage",
    ),
    OperationAdmission(
        "graph.induced_subgraph.compute",
        AdmissionDecision.NATIVE_ONLY,
        "useful deterministic helper retained through the supported native API",
        native_symbol="jacobian.math.graphs.induced_subgraph",
    ),
    OperationAdmission(
        "graph.induced_tree.maximum.compute",
        AdmissionDecision.KEEP,
        "distinct exact or explicitly bounded search outcome with material computational leverage",
    ),
    OperationAdmission(
        "graph.invariant.chromatic_number.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "graph.invariant.clique_number.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "graph.invariant.diameter.compute",
        AdmissionDecision.NATIVE_ONLY,
        "useful deterministic helper retained through the supported native API",
        native_symbol="jacobian.math.graphs.diameter",
    ),
    OperationAdmission(
        "graph.invariant.edge_connectivity.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "graph.invariant.girth.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "graph.invariant.independence_number.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "graph.invariant.is_eulerian.compute",
        AdmissionDecision.NATIVE_ONLY,
        "useful deterministic helper retained through the supported native API",
        native_symbol="jacobian.math.graphs.is_eulerian",
    ),
    OperationAdmission(
        "graph.invariant.maximum_matching.compute",
        AdmissionDecision.KEEP,
        "distinct exact or explicitly bounded search outcome with material computational leverage",
    ),
    OperationAdmission(
        "graph.invariant.radius.compute",
        AdmissionDecision.NATIVE_ONLY,
        "cheap projection of the retained all-pairs distance matrix",
        native_symbol="jacobian.math.graphs.radius",
    ),
    OperationAdmission(
        "graph.invariant.spanning_tree_count.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "graph.invariant.triangle_count.compute",
        AdmissionDecision.NATIVE_ONLY,
        "useful deterministic helper retained through the supported native API",
        native_symbol="jacobian.math.graphs.triangle_count",
    ),
    OperationAdmission(
        "graph.invariant.vertex_connectivity.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "graph.isomorphism.decide.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded predicate or candidate check with typed semantics",
    ),
    OperationAdmission(
        "graph.k_core.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "graph.line_graph.compute",
        AdmissionDecision.NATIVE_ONLY,
        "useful deterministic helper retained through the supported native API",
        native_symbol="jacobian.math.graphs.line_graph",
    ),
    OperationAdmission(
        "graph.matching.maximal.minimum.compute",
        AdmissionDecision.KEEP,
        "distinct exact or explicitly bounded search outcome with material computational leverage",
    ),
    OperationAdmission(
        "graph.menger.edge_disjoint.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "graph.polynomial.chromatic.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "graph.polynomial.flow.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "graph.polynomial.matching.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "graph.polynomial.tutte.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "graph.power.compute",
        AdmissionDecision.NATIVE_ONLY,
        "useful deterministic helper retained through the supported native API",
        native_symbol="jacobian.math.graphs.graph_power",
    ),
    OperationAdmission(
        "graph.realization.check.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "graph.realization.construct.compute",
        AdmissionDecision.KEEP,
        "reusable typed mathematical construction or transformation with a distinct discovery intent",
    ),
    OperationAdmission(
        "graph.realization.is_graphical.compute",
        AdmissionDecision.DROP,
        "boolean projection already determined by graph.realization.construct.compute",
    ),
    OperationAdmission(
        "graph.spanning_tree.minimum.compute",
        AdmissionDecision.KEEP,
        "distinct exact or explicitly bounded search outcome with material computational leverage",
    ),
    OperationAdmission(
        "graph.spectrum.adjacency.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "graph.spectrum.laplacian.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "graph.symmetry.generator_orbits.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "graphical_model.d_separation.compute",
        AdmissionDecision.KEEP,
        "exact d-separation verdict for a bounded directed acyclic graphical model",
    ),
    OperationAdmission(
        "graphical_model.factor.marginalize",
        AdmissionDecision.KEEP,
        "exact bounded factor marginalization",
    ),
    OperationAdmission(
        "graphical_model.factor.multiply",
        AdmissionDecision.KEEP,
        "exact bounded factor multiplication",
    ),
    OperationAdmission(
        "group.element_order.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "group.orbit.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "group.order.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "integer.compute.absolute_value",
        AdmissionDecision.NATIVE_ONLY,
        "useful deterministic helper retained through the supported native API",
        native_symbol="jacobian.math.arithmetic.absolute_value",
    ),
    OperationAdmission(
        "integer.compute.aliquot_sum",
        AdmissionDecision.DROP,
        "ordinary arithmetic or cheap projection of a retained exact factorization/divisor result",
    ),
    OperationAdmission(
        "integer.compute.decimal_digit_count",
        AdmissionDecision.DROP,
        "ordinary arithmetic or cheap projection of a retained exact factorization/divisor result",
    ),
    OperationAdmission(
        "integer.compute.decimal_digit_sum",
        AdmissionDecision.DROP,
        "ordinary arithmetic or cheap projection of a retained exact factorization/divisor result",
    ),
    OperationAdmission(
        "integer.compute.divisor_count",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "integer.compute.divisor_sum",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "integer.compute.divisors",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "integer.compute.euler_totient",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "integer.compute.extended_gcd",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "integer.compute.floor_square_root",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "integer.compute.gcd",
        AdmissionDecision.DROP,
        "ordinary arithmetic or cheap projection of a retained exact factorization/divisor result",
    ),
    OperationAdmission(
        "integer.compute.lcm",
        AdmissionDecision.DROP,
        "ordinary arithmetic or cheap projection of a retained exact factorization/divisor result",
    ),
    OperationAdmission(
        "integer.compute.mobius",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "integer.compute.next_prime",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "integer.compute.nth_prime",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "integer.compute.nth_root",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "integer.compute.previous_prime",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "integer.compute.prime_count",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "integer.compute.prime_factorization",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "integer.compute.primorial",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "integer.compute.proper_divisors",
        AdmissionDecision.DROP,
        "ordinary arithmetic or cheap projection of a retained exact factorization/divisor result",
    ),
    OperationAdmission(
        "integer.compute.radical",
        AdmissionDecision.DROP,
        "ordinary arithmetic or cheap projection of a retained exact factorization/divisor result",
    ),
    OperationAdmission(
        "integer.compute.sign",
        AdmissionDecision.NATIVE_ONLY,
        "useful deterministic helper retained through the supported native API",
        native_symbol="jacobian.math.arithmetic.sign",
    ),
    OperationAdmission(
        "integer.compute.valuation",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "integer.counting.congruence_box.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "integer.counting.floor_sum.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "integer.decide.abundant",
        AdmissionDecision.DROP,
        "ordinary arithmetic or cheap projection of a retained exact factorization/divisor result",
    ),
    OperationAdmission(
        "integer.decide.coprime",
        AdmissionDecision.DROP,
        "ordinary arithmetic or cheap projection of a retained exact factorization/divisor result",
    ),
    OperationAdmission(
        "integer.decide.deficient",
        AdmissionDecision.DROP,
        "ordinary arithmetic or cheap projection of a retained exact factorization/divisor result",
    ),
    OperationAdmission(
        "integer.decide.divides",
        AdmissionDecision.DROP,
        "ordinary arithmetic or cheap projection of a retained exact factorization/divisor result",
    ),
    OperationAdmission(
        "integer.decide.even",
        AdmissionDecision.DROP,
        "ordinary arithmetic or cheap projection of a retained exact factorization/divisor result",
    ),
    OperationAdmission(
        "integer.decide.odd",
        AdmissionDecision.DROP,
        "ordinary arithmetic or cheap projection of a retained exact factorization/divisor result",
    ),
    OperationAdmission(
        "integer.decide.perfect",
        AdmissionDecision.DROP,
        "ordinary arithmetic or cheap projection of a retained exact factorization/divisor result",
    ),
    OperationAdmission(
        "integer.decide.powerful",
        AdmissionDecision.DROP,
        "ordinary arithmetic or cheap projection of a retained exact factorization/divisor result",
    ),
    OperationAdmission(
        "integer.decide.prime",
        AdmissionDecision.KEEP,
        "distinct exact bounded predicate or candidate check with typed semantics",
    ),
    OperationAdmission(
        "integer.decide.square",
        AdmissionDecision.DROP,
        "ordinary arithmetic or cheap projection of a retained exact factorization/divisor result",
    ),
    OperationAdmission(
        "integer.decide.squarefree",
        AdmissionDecision.DROP,
        "ordinary arithmetic or cheap projection of a retained exact factorization/divisor result",
    ),
    OperationAdmission(
        "integer.transform.base_digits",
        AdmissionDecision.DROP,
        "ordinary arithmetic or cheap projection of a retained exact factorization/divisor result",
    ),
    OperationAdmission(
        "lattice.basis.reduce",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "lattice.hermite_normal_form.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "lean.check",
        AdmissionDecision.KEEP,
        "distinct exact bounded predicate or candidate check with typed semantics",
    ),
    OperationAdmission(
        "linear.rational_inconsistency.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "linear.rational_solution.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "matrix.adjugate.compute",
        AdmissionDecision.NATIVE_ONLY,
        "useful deterministic helper retained through the supported native API",
        native_symbol="jacobian.math.matrices.adjugate",
    ),
    OperationAdmission(
        "matrix.characteristic_polynomial.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "matrix.determinant.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "matrix.farkas_certificate.check",
        AdmissionDecision.KEEP,
        "distinct exact bounded predicate or candidate check with typed semantics",
    ),
    OperationAdmission(
        "matrix.inertia.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "matrix.inverse.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "matrix.kronecker_product.compute",
        AdmissionDecision.NATIVE_ONLY,
        "useful deterministic helper retained through the supported native API",
        native_symbol="jacobian.math.matrices.kronecker_product",
    ),
    OperationAdmission(
        "matrix.minimal_polynomial.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "matrix.multiply.compute",
        AdmissionDecision.NATIVE_ONLY,
        "useful deterministic helper retained through the supported native API",
        native_symbol="jacobian.math.matrices.multiply",
    ),
    OperationAdmission(
        "matrix.normal_form.rref.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "matrix.normal_form.smith.certified.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "matrix.normal_form.smith.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "matrix.nullspace.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "matrix.partial_trace.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "matrix.permanent.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "matrix.primary_decomposition.compute",
        AdmissionDecision.KEEP,
        "reusable typed mathematical construction or transformation with a distinct discovery intent",
    ),
    OperationAdmission(
        "matrix.rank.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "matrix.rational_canonical_form.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "matrix.rational_linear_system.solve",
        AdmissionDecision.KEEP,
        "distinct exact or explicitly bounded search outcome with material computational leverage",
    ),
    OperationAdmission(
        "matrix.symbolic.characteristic_polynomial.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "matrix.symbolic.determinant.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "matrix.symbolic.eigenvalues.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "matrix.symbolic.rank.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "matrix.trace.compute",
        AdmissionDecision.NATIVE_ONLY,
        "useful deterministic helper retained through the supported native API",
        native_symbol="jacobian.math.matrices.trace",
    ),
    OperationAdmission(
        "metric_space.ball.compute",
        AdmissionDecision.NATIVE_ONLY,
        "direct row filter on a caller-supplied finite distance matrix",
        native_symbol="jacobian.math.finite_metric_spaces.ball",
    ),
    OperationAdmission(
        "metric_space.gromov_hyperbolicity.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "metric_space.profile.compute",
        AdmissionDecision.KEEP,
        "one complete exact metric profile whose mutually bound fields form a reusable invariant family",
    ),
    OperationAdmission(
        "modular.compute.discrete_logarithm",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "modular.compute.inverse",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "modular.compute.multiplicative_order",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "modular.enumerate.quadratic_residues",
        AdmissionDecision.KEEP,
        "distinct exact or explicitly bounded search outcome with material computational leverage",
    ),
    OperationAdmission(
        "modular.polynomial_identity.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "modular.polynomial_residue_image.assignments.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "modular.polynomial_residue_image.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "modular.solve.chinese_remainder",
        AdmissionDecision.KEEP,
        "distinct exact or explicitly bounded search outcome with material computational leverage",
    ),
    OperationAdmission(
        "number_field.discriminant.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "number_theory.compute.factorial_valuation",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "number_theory.compute.jacobi_symbol",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "number_theory.compute.legendre_symbol",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "number_theory.numerical_semigroup.betti_elements.compute",
        AdmissionDecision.KEEP,
        "exact complete Betti enumeration replacing the capped heuristic search after the #1977 contract repair",
    ),
    OperationAdmission(
        "number_theory.numerical_semigroup.catenary_degree.compute",
        AdmissionDecision.KEEP,
        "complete global catenary-degree invariant rebuilt on the repaired Betti basis after the #1977 contract repair",
    ),
    OperationAdmission(
        "number_theory.numerical_semigroup.delta_set.compute",
        AdmissionDecision.KEEP,
        "complete global delta-set invariant rebuilt on the repaired Betti basis after the #1977 contract repair",
    ),
    OperationAdmission(
        "number_theory.numerical_semigroup.factorization_graph.compute",
        AdmissionDecision.KEEP,
        "reusable factorization graph and component construction",
    ),
    OperationAdmission(
        "number_theory.numerical_semigroup.factorizations.compute",
        AdmissionDecision.KEEP,
        "complete bounded factorization family for one element",
    ),
    OperationAdmission(
        "number_theory.numerical_semigroup.membership.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "number_theory.numerical_semigroup.minimal_presentation.compute",
        AdmissionDecision.KEEP,
        "minimal presentation rebuilt on the exact Betti basis after the #1977 contract repair",
    ),
    OperationAdmission(
        "number_theory.numerical_semigroup.presentation_binomials.compute",
        AdmissionDecision.KEEP,
        "unit binomial coefficients of the repaired minimal presentation after the #1977 contract repair",
    ),
    OperationAdmission(
        "number_theory.numerical_semigroup.summary.compute",
        AdmissionDecision.KEEP,
        "one complete exact finite gap profile with its mutually determined canonical invariants",
    ),
    OperationAdmission(
        "optimization.linear.rational_optimum.compute",
        AdmissionDecision.KEEP,
        "distinct exact or explicitly bounded search outcome with material computational leverage",
    ),
    OperationAdmission(
        "petri_net.fire_transition.compute",
        AdmissionDecision.KEEP,
        "exact transition firing with typed marking semantics",
    ),
    OperationAdmission(
        "petri_net.reachability_graph.compute",
        AdmissionDecision.KEEP,
        "bounded reachability construction with an explicit typed completeness frontier after the #1978 contract repair",
    ),
    OperationAdmission(
        "polynomial.compute.discriminant",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "polynomial.compute.gcd",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "polynomial.compute.resultant",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "polynomial.compute.square_free_decomposition",
        AdmissionDecision.KEEP,
        "reusable typed mathematical construction or transformation with a distinct discovery intent",
    ),
    OperationAdmission(
        "polynomial.factor.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "polynomial.integer.compute.compose",
        AdmissionDecision.DROP,
        "ordinary polynomial projection or composition better expressed through the native SymPy-valued API",
    ),
    OperationAdmission(
        "polynomial.integer.compute.content",
        AdmissionDecision.DROP,
        "ordinary polynomial projection or composition better expressed through the native SymPy-valued API",
    ),
    OperationAdmission(
        "polynomial.integer.compute.evaluate",
        AdmissionDecision.DROP,
        "ordinary polynomial projection or composition better expressed through the native SymPy-valued API",
    ),
    OperationAdmission(
        "polynomial.integer.compute.gcd",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "polynomial.integer.compute.primitive_part",
        AdmissionDecision.DROP,
        "ordinary polynomial projection or composition better expressed through the native SymPy-valued API",
    ),
    OperationAdmission(
        "polynomial.integer.compute.shift",
        AdmissionDecision.DROP,
        "ordinary polynomial projection or composition better expressed through the native SymPy-valued API",
    ),
    OperationAdmission(
        "polynomial.jacobian_syzygy.coefficients.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "polynomial.jacobian_syzygy.minimum_degree.compute",
        AdmissionDecision.KEEP,
        "distinct exact or explicitly bounded search outcome with material computational leverage",
    ),
    OperationAdmission(
        "polynomial.map.compose",
        AdmissionDecision.DROP,
        "ordinary polynomial projection or composition better expressed through the native SymPy-valued API",
    ),
    OperationAdmission(
        "polynomial.map.evaluate",
        AdmissionDecision.DROP,
        "ordinary polynomial projection or composition better expressed through the native SymPy-valued API",
    ),
    OperationAdmission(
        "polynomial.map.jacobian",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "polynomial.multivariate.divide.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "polynomial.multivariate.gcd.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "polynomial.multivariate.resultant.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "polynomial.rational.compute.derivative",
        AdmissionDecision.NATIVE_ONLY,
        "useful deterministic helper retained through the supported native API",
        native_symbol="jacobian.math.polynomials.derivative",
    ),
    OperationAdmission(
        "polynomial.rational.compute.evaluate",
        AdmissionDecision.NATIVE_ONLY,
        "useful deterministic helper retained through the supported native API",
        native_symbol="jacobian.math.polynomials.evaluate",
    ),
    OperationAdmission(
        "polynomial.rational.compute.integral",
        AdmissionDecision.NATIVE_ONLY,
        "useful deterministic helper retained through the supported native API",
        native_symbol="jacobian.math.polynomials.integral",
    ),
    OperationAdmission(
        "polynomial.rational.compute.partial_fraction_decomposition",
        AdmissionDecision.NATIVE_ONLY,
        "useful deterministic helper retained through the supported native API",
        native_symbol="jacobian.math.polynomials.partial_fractions",
    ),
    OperationAdmission(
        "polynomial.rational.compute.quotient_remainder",
        AdmissionDecision.NATIVE_ONLY,
        "useful deterministic helper retained through the supported native API",
        native_symbol="jacobian.math.polynomials.divide",
    ),
    OperationAdmission(
        "polynomial.root_count.compute",
        AdmissionDecision.KEEP,
        "distinct exact or explicitly bounded search outcome with material computational leverage",
    ),
    OperationAdmission(
        "polynomial.roots.isolate",
        AdmissionDecision.KEEP,
        "distinct exact or explicitly bounded search outcome with material computational leverage",
    ),
    OperationAdmission(
        "polynomial.sturm_chain.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "poset.finite.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "poset.linear_extensions.count",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "poset.mobius_function.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "poset.width.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "probability.bh_step_up.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "probability.fdp.compute",
        AdmissionDecision.DROP,
        "cheap deterministic projection of two caller-supplied finite sets",
    ),
    OperationAdmission(
        "probability.finite_distribution.condition.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "probability.finite_distribution.convolution.compute",
        AdmissionDecision.KEEP,
        "reusable typed mathematical construction or transformation with a distinct discovery intent",
    ),
    OperationAdmission(
        "probability.finite_distribution.event_probability.compute",
        AdmissionDecision.DROP,
        "cheap deterministic projection of a caller-supplied finite distribution",
    ),
    OperationAdmission(
        "probability.finite_distribution.pushforward.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "probability.finite_distribution.raw_moment.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "probability.gaussian_polynomial.moment.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "probability.graph_reliability.connection_probability.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "probability.joint.mutual_information.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "probability.markov_chain.ergodic.decide",
        AdmissionDecision.KEEP,
        "distinct exact bounded predicate or candidate check with typed semantics",
    ),
    OperationAdmission(
        "probability.markov_chain.stationary_distribution.compute",
        AdmissionDecision.KEEP,
        "returns every canonical extreme point of the complete stationary-distribution simplex",
    ),
    OperationAdmission(
        "rational.compute.absolute_value",
        AdmissionDecision.DROP,
        "ordinary Fraction operation without catalog-level leverage",
    ),
    OperationAdmission(
        "rational.compute.ceiling",
        AdmissionDecision.DROP,
        "ordinary Fraction operation without catalog-level leverage",
    ),
    OperationAdmission(
        "rational.compute.continued_fraction",
        AdmissionDecision.DROP,
        "ordinary Fraction operation without catalog-level leverage",
    ),
    OperationAdmission(
        "rational.compute.difference",
        AdmissionDecision.DROP,
        "ordinary Fraction operation without catalog-level leverage",
    ),
    OperationAdmission(
        "rational.compute.floor",
        AdmissionDecision.DROP,
        "ordinary Fraction operation without catalog-level leverage",
    ),
    OperationAdmission(
        "rational.compute.maximum",
        AdmissionDecision.DROP,
        "ordinary Fraction operation without catalog-level leverage",
    ),
    OperationAdmission(
        "rational.compute.minimum",
        AdmissionDecision.DROP,
        "ordinary Fraction operation without catalog-level leverage",
    ),
    OperationAdmission(
        "rational.compute.negation",
        AdmissionDecision.DROP,
        "ordinary Fraction operation without catalog-level leverage",
    ),
    OperationAdmission(
        "rational.compute.product",
        AdmissionDecision.DROP,
        "ordinary Fraction operation without catalog-level leverage",
    ),
    OperationAdmission(
        "rational.compute.quotient",
        AdmissionDecision.NATIVE_ONLY,
        "useful deterministic helper retained through the supported native API",
        native_symbol="jacobian.math.arithmetic.quotient",
    ),
    OperationAdmission(
        "rational.compute.reciprocal",
        AdmissionDecision.NATIVE_ONLY,
        "useful deterministic helper retained through the supported native API",
        native_symbol="jacobian.math.arithmetic.reciprocal",
    ),
    OperationAdmission(
        "rational.compute.sum",
        AdmissionDecision.NATIVE_ONLY,
        "useful deterministic helper retained through the supported native API",
        native_symbol="jacobian.math.arithmetic.sum_rationals",
    ),
    OperationAdmission(
        "rational.decide.equal",
        AdmissionDecision.DROP,
        "ordinary Fraction operation without catalog-level leverage",
    ),
    OperationAdmission(
        "rational.decide.less_than",
        AdmissionDecision.DROP,
        "ordinary Fraction operation without catalog-level leverage",
    ),
    OperationAdmission(
        "regular_language.complement.compute",
        AdmissionDecision.NATIVE_ONLY,
        "cheap deterministic accepting-state projection of a supplied complete DFA",
        native_symbol="jacobian.math.regular_languages.dfa_complement",
    ),
    OperationAdmission(
        "regular_language.count_words.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "regular_language.run.check",
        AdmissionDecision.KEEP,
        "distinct exact bounded predicate or candidate check with typed semantics",
    ),
    OperationAdmission(
        "sat.assignment.check",
        AdmissionDecision.KEEP,
        "distinct exact bounded predicate or candidate check with typed semantics",
    ),
    OperationAdmission(
        "sat.cnf.canonicalize",
        AdmissionDecision.KEEP,
        "reusable typed mathematical construction or transformation with a distinct discovery intent",
    ),
    OperationAdmission(
        "sat.solve",
        AdmissionDecision.KEEP,
        "distinct exact or explicitly bounded search outcome with material computational leverage",
    ),
    OperationAdmission(
        "sequence.compute.distinct_count",
        AdmissionDecision.DROP,
        "ordinary deterministic finite-sequence projection better authored directly in Python",
    ),
    OperationAdmission(
        "sequence.compute.first_differences",
        AdmissionDecision.DROP,
        "ordinary deterministic finite-sequence projection better authored directly in Python",
    ),
    OperationAdmission(
        "sequence.compute.frequencies",
        AdmissionDecision.DROP,
        "ordinary deterministic finite-sequence projection better authored directly in Python",
    ),
    OperationAdmission(
        "sequence.compute.gcd",
        AdmissionDecision.DROP,
        "ordinary deterministic finite-sequence projection better authored directly in Python",
    ),
    OperationAdmission(
        "sequence.compute.lcm",
        AdmissionDecision.DROP,
        "ordinary deterministic finite-sequence projection better authored directly in Python",
    ),
    OperationAdmission(
        "sequence.compute.maximum",
        AdmissionDecision.DROP,
        "ordinary deterministic finite-sequence projection better authored directly in Python",
    ),
    OperationAdmission(
        "sequence.compute.mean",
        AdmissionDecision.DROP,
        "ordinary deterministic finite-sequence projection better authored directly in Python",
    ),
    OperationAdmission(
        "sequence.compute.median",
        AdmissionDecision.DROP,
        "ordinary deterministic finite-sequence projection better authored directly in Python",
    ),
    OperationAdmission(
        "sequence.compute.minimum",
        AdmissionDecision.DROP,
        "ordinary deterministic finite-sequence projection better authored directly in Python",
    ),
    OperationAdmission(
        "sequence.compute.prefix_gcds",
        AdmissionDecision.DROP,
        "ordinary deterministic finite-sequence projection better authored directly in Python",
    ),
    OperationAdmission(
        "sequence.compute.prefix_lcms",
        AdmissionDecision.DROP,
        "ordinary deterministic finite-sequence projection better authored directly in Python",
    ),
    OperationAdmission(
        "sequence.compute.prefix_maxima",
        AdmissionDecision.DROP,
        "ordinary deterministic finite-sequence projection better authored directly in Python",
    ),
    OperationAdmission(
        "sequence.compute.prefix_minima",
        AdmissionDecision.DROP,
        "ordinary deterministic finite-sequence projection better authored directly in Python",
    ),
    OperationAdmission(
        "sequence.compute.prefix_products",
        AdmissionDecision.DROP,
        "ordinary deterministic finite-sequence projection better authored directly in Python",
    ),
    OperationAdmission(
        "sequence.compute.prefix_sums",
        AdmissionDecision.DROP,
        "ordinary deterministic finite-sequence projection better authored directly in Python",
    ),
    OperationAdmission(
        "sequence.compute.product",
        AdmissionDecision.DROP,
        "ordinary deterministic finite-sequence projection better authored directly in Python",
    ),
    OperationAdmission(
        "sequence.compute.range",
        AdmissionDecision.DROP,
        "ordinary deterministic finite-sequence projection better authored directly in Python",
    ),
    OperationAdmission(
        "sequence.compute.second_differences",
        AdmissionDecision.DROP,
        "ordinary deterministic finite-sequence projection better authored directly in Python",
    ),
    OperationAdmission(
        "sequence.compute.sum",
        AdmissionDecision.DROP,
        "ordinary deterministic finite-sequence projection better authored directly in Python",
    ),
    OperationAdmission(
        "sequence.compute.zero_indices",
        AdmissionDecision.DROP,
        "ordinary deterministic finite-sequence projection better authored directly in Python",
    ),
    OperationAdmission(
        "sequence.decide.arithmetic",
        AdmissionDecision.DROP,
        "ordinary deterministic finite-sequence projection better authored directly in Python",
    ),
    OperationAdmission(
        "sequence.decide.geometric",
        AdmissionDecision.DROP,
        "ordinary deterministic finite-sequence projection better authored directly in Python",
    ),
    OperationAdmission(
        "sequence.decide.nondecreasing",
        AdmissionDecision.DROP,
        "ordinary deterministic finite-sequence projection better authored directly in Python",
    ),
    OperationAdmission(
        "sequence.decide.strictly_increasing",
        AdmissionDecision.DROP,
        "ordinary deterministic finite-sequence projection better authored directly in Python",
    ),
    OperationAdmission(
        "sequence.recurrence.closed_form.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "sequence.recurrence.find",
        AdmissionDecision.KEEP,
        "distinct exact or explicitly bounded search outcome with material computational leverage",
    ),
    OperationAdmission(
        "sequence.transform.parities",
        AdmissionDecision.DROP,
        "ordinary deterministic finite-sequence projection better authored directly in Python",
    ),
    OperationAdmission(
        "sequence.transform.reverse",
        AdmissionDecision.DROP,
        "ordinary deterministic finite-sequence projection better authored directly in Python",
    ),
    OperationAdmission(
        "sequence.transform.signs",
        AdmissionDecision.DROP,
        "ordinary deterministic finite-sequence projection better authored directly in Python",
    ),
    OperationAdmission(
        "sequence.transform.sort",
        AdmissionDecision.DROP,
        "ordinary deterministic finite-sequence projection better authored directly in Python",
    ),
    OperationAdmission(
        "sequence.transform.sorted_unique",
        AdmissionDecision.DROP,
        "ordinary deterministic finite-sequence projection better authored directly in Python",
    ),
    OperationAdmission(
        "smt.solve",
        AdmissionDecision.KEEP,
        "distinct exact or explicitly bounded search outcome with material computational leverage",
    ),
    OperationAdmission(
        "symbolic_dynamics.block_language.compute",
        AdmissionDecision.KEEP,
        "complete block language of a bounded shift presentation",
    ),
    OperationAdmission(
        "symbolic_dynamics.finite_type_shift.construct",
        AdmissionDecision.KEEP,
        "exact finite-type shift presentation from a bounded edge label set",
    ),
    OperationAdmission(
        "symbolic_dynamics.higher_block.compute",
        AdmissionDecision.KEEP,
        "exact higher-block presentation of a bounded shift",
    ),
    OperationAdmission(
        "symbolic_dynamics.periodic_point_profile.compute",
        AdmissionDecision.KEEP,
        "complete periodic point profile of a bounded shift",
    ),
    OperationAdmission(
        "term_rewriting.matching.compute",
        AdmissionDecision.KEEP,
        "typed first-order term matching with a complete substitution result",
    ),
    OperationAdmission(
        "term_rewriting.rewrite_step.compute",
        AdmissionDecision.KEEP,
        "explicit one-step rewrite choices with a typed rewrite application result",
    ),
    OperationAdmission(
        "term_rewriting.unification.compute",
        AdmissionDecision.KEEP,
        "most general unifier certificate for a bounded first-order unification problem",
    ),
    OperationAdmission(
        "topology.beat_points.compute",
        AdmissionDecision.KEEP,
        "complete beat-point witness family for a bounded finite topology",
    ),
    OperationAdmission(
        "topology.connected_components.compute",
        AdmissionDecision.KEEP,
        "complete connected component partition of a bounded finite topology",
    ),
    OperationAdmission(
        "topology.is_continuous.compute",
        AdmissionDecision.KEEP,
        "exact continuity verdict between bounded finite topologies",
    ),
    OperationAdmission(
        "topology.simplicial_complex.canonicalize",
        AdmissionDecision.KEEP,
        "reusable typed mathematical construction or transformation with a distinct discovery intent",
    ),
    OperationAdmission(
        "topology.simplicial_complex.chain_complex.compute",
        AdmissionDecision.KEEP,
        "reusable typed mathematical construction or transformation with a distinct discovery intent",
    ),
    OperationAdmission(
        "topology.simplicial_homology.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "topology.simplicial_homology.integral.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "topology.specialization_preorder.compute",
        AdmissionDecision.KEEP,
        "complete specialization preorder of a bounded finite topology",
    ),
    OperationAdmission(
        "transducer.relation.path.replay.compute",
        AdmissionDecision.KEEP,
        "exact path replay over a rational transducer relation",
    ),
    OperationAdmission(
        "transducer.subsequential.compose.compute",
        AdmissionDecision.KEEP,
        "exact composition of bounded subsequential transducers",
    ),
    OperationAdmission(
        "transducer.subsequential.run.compute",
        AdmissionDecision.KEEP,
        "exact subsequential transducer run with complete output",
    ),
    OperationAdmission(
        "tree_automaton.accepted_tree_count.compute",
        AdmissionDecision.KEEP,
        "exact bounded accepted-tree enumeration for a deterministic bottom-up tree automaton",
    ),
    OperationAdmission(
        "tree_automaton.run.compute",
        AdmissionDecision.KEEP,
        "typed bottom-up tree automaton run with a complete accepted or rejected verdict",
    ),
    OperationAdmission(
        "word.factors.length.compute",
        AdmissionDecision.KEEP,
        "complete bounded factor table of a finite word",
    ),
    OperationAdmission(
        "word.periods.compute",
        AdmissionDecision.KEEP,
        "complete period set with border certificate for a finite word",
    ),
    OperationAdmission(
        "word_morphism.incidence_matrix.compute",
        AdmissionDecision.KEEP,
        "exact incidence matrix of a bounded word morphism",
    ),
)


def curate_public_tools(candidates: MathTools) -> MathTools:
    """Return only reviewed public operations and fail closed on ledger drift."""

    records = {record.operation_id: record for record in OPERATION_ADMISSIONS}
    if len(records) != len(OPERATION_ADMISSIONS):
        raise ValueError("operation admission IDs must be unique")
    candidate_sequence = tuple(tool.operation_id for tool in candidates)
    candidate_ids = set(candidate_sequence)
    if len(candidate_ids) != len(candidate_sequence):
        raise ValueError("candidate operation IDs must be unique")
    record_ids = set(records)
    if candidate_ids != record_ids:
        missing = sorted(candidate_ids - record_ids)
        stale = sorted(record_ids - candidate_ids)
        raise ValueError(
            "operation admission ledger does not match candidates: "
            f"missing={missing}, stale={stale}"
        )
    admitted = {AdmissionDecision.KEEP}
    return tuple(
        tool for tool in candidates if records[tool.operation_id].decision in admitted
    )


def admission_by_id() -> dict[str, OperationAdmission]:
    return {record.operation_id: record for record in OPERATION_ADMISSIONS}


__all__ = [
    "OPERATION_ADMISSIONS",
    "REVIEWED_BASE_REVISION",
    "AdmissionDecision",
    "OperationAdmission",
    "admission_by_id",
    "curate_public_tools",
]
