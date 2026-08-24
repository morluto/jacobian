# Jacobian Scale/Backend Audit Report

## Scope

- **532 admitted operations** across **87 mathematical domains**
- **501 fixed caps** enumerated from source
- **35+ domains** audited in depth via 5 parallel explorers
- **19 GitHub issues** created for validated, non-duplicate findings

## Methodology

1. Enumerated all fixed caps (`MAX_*`, `_MAX_*`) across `src/jacobian/math/`
2. For each cap, traced the operations it gates, the algorithm used, and the true bottleneck
3. Reproduced public rejections via Pydantic model validation
4. Benchmarked existing kernels and maintained backends (SymPy, Z3, FLINT)
5. Classified each cap as: conservative (scale/backend gap), genuine (output-size or mathematical boundary), or algorithmic (suboptimal algorithm masked by conservative cap)

## Findings Summary

### Category 1: Algorithm-regime change needed (brute-force where better algorithms exist)

| Issue | Operation | Current Cap | Algorithm | Better Algorithm | Safe Envelope |
|-------|-----------|-------------|-----------|-------------------|---------------|
| #2399 | `integer.counting.floor_sum.compute` | n≤1M | O(n) loop | O(log m) Euclidean | n≤10^18 |
| #2420 | `discrepancy.theory.optimum.compute` | n≤20 | 2^n brute-force | Exact ILP | n≤100 |
| #2409 | `combinatorics.set_function.submodularity` | n≤12 | O(4^n) all-pairs | O(n²×2^n) diminishing returns | n≤20 |

### Category 2: Brute-force algorithm where maintained backends exist

| Issue | Operation | Current Cap | Algorithm | Backend Available | Safe Envelope |
|-------|-----------|-------------|-----------|-------------------|---------------|
| #2405 | `polytope.volume.compute` | dim≤6, vertices≤64 | O(C(n,d)) brute-force hull | scipy/Qhull, cddlib, Normaliz | dim≤10, vertices≤500 |

### Category 3: Enumeration cap applied to non-enumerating operations

| Issue | Operation | Current Cap | True Cost | Safe Envelope |
|-------|-----------|-------------|-----------|---------------|
| #2403 | `code.linear.*` (8 of 9 ops) | q^k≤4096 | O(k²×n) RREF | Remove cap; length≤128 |
| #2400 | `graph.coloring.independent_set.maximal.decide` | 20 vertices | O(V+E) linear scan | 64+ vertices |
| #2400 | `graph.edge_coloring.check` | 20 vertices | O(E²) check | 128+ vertices |
| #2430 | `graph.cycle.fixed_length.decide` | 20 vertices | Has work-budget admission | 64 vertices |

### Category 4: Pure cap conservatism (optimal algorithm, cap set too low)

| Issue | Operation | Current Cap | True Cost | Safe Envelope |
|-------|-----------|-------------|-----------|---------------|
| #2406 | `combinatorics.compute.*` | n≤1000 | O(1) SymPy | n≤10,000 |
| #2407 | `quadratic_form.evaluate/discriminant/signature` | dim≤10 | O(n²)–O(n³) | dim≤50 |
| #2412 | `finite_set.*` | 128 elements | O(n) set ops | 1,024 elements |
| #2414 | `game_theory.nash_equilibrium` | strategies≤8 | O(LP) exact | strategies≤50 |
| #2415 | `algebra.center.compute` | dim≤32 | O(n³) Gaussian elim | dim≤128 |
| #2416 | `convex.max_affine.*` | dim≤20, pieces≤100 | O(p×d) linear scan | dim≤100, pieces≤10,000 |
| #2418 | `code.nonlinear.distance_profile.*` | length≤16 | O(n²×L) integer ops | length≤64 |
| #2419 | `additive.representation_profile.compute` | set_size≤256 | O(|A|×|B|) | set_size≤4,096 |
| #2421 | `poset.linear_extensions.count` | n≤14 | O(2^n) DP | n≤20 |
| #2422 | `group_action.element_cycles/cycle_index/burnside` | |G|≤720 | O(|G|×n) | |G|≤10,000 |
| #2424 | `electrical_network.*` | vertices≤64 | O(n³) rational solve | vertices≤200 |
| #2425 | `probability.gaussian_polynomial.moment` | variables≤8 | FLINT fmpq_mpoly | variables≤20 |
| #2426 | `finite_field.*` | order≤4096 | Work-capped (1M) | order≤65,536 |
| #2427 | `formal_context.derivation/closure/concept` | objects/attrs≤64 | O(|incidence|) linear | objects/attrs≤500 |
| #2428 | `cubical.f_vector/face_closure` | dim≤5, cells≤200 | O(cells×2^dim) | dim≤10, cells≤5,000 |
| #2429 | `quiver.adjacency/profiles/paths` | vertices≤32 | O(n²)–O(length×n³) | vertices≤200 |
| #2431 | `polynomial.ideal.*` | vars≤6, generators≤16 | Gröbner (wall-timed) | vars≤8, generators≤32 |

### Caps confirmed as NOT conservative (mathematical boundaries or well-reasoned)

- `root_systems.MAX_RANK = 8` — mathematical boundary for finite crystallographic root systems
- `symbolic_dynamics.*` — exponential state space (|A|^memory) is genuine
- `lattice_polytopes.MAX_DIMENSION = 4` — honest for brute-force box scanning
- `electrical_networks.MAX_CONDUCTANCE_DIGITS` — well-reasoned spanning-tree polynomial bound
- `arithmetic_dynamics.MAX_ITERATE = 20` — degree cap (1024) binds first; genuine output-size limit
- `finite_metric_spaces.gromov_hyperbolicity` — O(n⁴) is the real barrier

## Reproduction Evidence

```
floor_sum n=1,000,001: REJECTED (cap=1,000,000)
floor_sum n=1,000,000 brute-force: 0.21s
SymPy fibonacci(100000): 0.006s (cap rejects n>1000)
Z3 3-colorability n=50: 0.037s (cap rejects n>20)
Quadratic form evaluate dim=20: 0.0007s (cap rejects dim>10)
SymPy det dim=50: 2.9s (cap rejects dim>10)
Monotonicity check n=20: 3.4s (cap rejects n>12)
Python set union 10K elements: 0.001s (cap rejects >128)
```
