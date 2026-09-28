# Petri-net state equation

[Documentation home](../../index.md) · [Operation references](index.md)

`petri_net.state_equation.target.compute` computes the formal integer vector

```text
M_target = M_source + (Post - Pre) y
```

for a supplied nonnegative transition-count vector `y`. The output is an
integer vector on the net's place axis, so negative coordinates are retained
as formal results. The operation does not check whether any transition is
enabled and does not assert that the vector is a reachable marking or that
`y` can be ordered into a firing sequence. In Petri-net theory, the state
equation gives a necessary condition for reachability; it is not sufficient in
general. Murata's survey states this condition and its limitation in
[“Petri Nets: Properties, Analysis, and Applications,” §3.2](https://www.dsc.ufcg.edu.br/~abrantes/CursosAnteriores/MVSRP/murata89.pdf).

The transition-count vector must match the transition axis, contain
nonnegative integers, and have total multiplicity at most 1,000. The net's
existing limits (64 places, 64 transitions, arc weights at most 1,000) bound
the multiplication work and result size. The result retains the exact net and
source marking context, including an explicit marking parent when supplied.

`petri_net.marking_equation.compute` compares a supplied target marking with
the formal target. It returns that formal target, the full signed residual
`M_target - M_source - (Post - Pre)y`, and whether every residual coordinate is
zero. It uses the same 1,000 total-count bound. The formal-target and residual
coordinates are bounded by 1,001,000 from the marking, arc-weight, and
multiplicity limits; output admission occurs before either vector is built.
Even a zero residual only establishes the algebraic state equation. It does not
establish that the counts can be ordered as enabled firings or that the target
is reachable.

For example, if two transitions require opposite places and produce into each
other's required place, `y = (1, 1)` may satisfy the state equation at the
empty marking although neither transition is enabled. The result reports the
formal target only; sequence feasibility belongs to firing-sequence or
reachability operations.
