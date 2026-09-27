# Petri net reversal

`petri_net.reverse.compute` returns the Petri net on the same ordered place and
transition axes with each transition's input and output arcs exchanged:

\[
\operatorname{pre}_{\mathrm{rev}}(p,t)=\operatorname{post}(p,t),\qquad
\operatorname{post}_{\mathrm{rev}}(p,t)=\operatorname{pre}(p,t).
\]

If transition `t` fires from marking `M` to `M'` in the source net, then `t`
fires from `M'` back to `M` in the reversed net. Indeed,
`M' = M - pre_t + post_t`; the reverse transition is enabled because
`M' >= post_t`, and its result is `M' - post_t + pre_t = M`. Reversal is
involutive. These are transition-local laws; the operation makes no claim
about reachability or reversing an entire execution history.

The kernel admits and canonicalizes the input first. The existing carrier
bounds the incidence matrices to at most 64 places by 64 transitions; the
kernel checks the serialized input-size envelope before returning the direct
`PetriNet` value. Its immutable incidence tuples are swapped without expanding
another matrix.
