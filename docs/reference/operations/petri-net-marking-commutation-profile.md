# Petri-net sequential commutation profile

`petri_net.marking.commutation_profile.compute` takes a marking and an
ordered pair of distinct transitions, then replays both two-step orders:

```text
t, u
u, t
```

Each nested `FiringSequenceReplayResult` returns every prefix marking and the
final marking when that order fires, or its first blocked transition and
complete place-deficit vector. `both_orders_fire` states whether both orders
execute. `same_target` is true only when both execute and their exact final
markings agree.

This is a local sequential diamond at one marking. It does not test whether
the transitions are enabled simultaneously, and it does not establish global
confluence across other markings. Use
[`petri_net.marking.conflict_profile.compute`](petri-net-marking-conflict-profile.md)
for pairwise simultaneous-step compatibility.

The operation runs two fixed-length replays and admits their bounded output
before constructing either result. If a replay leaves the marking token
envelope, the operation returns an admission failure rather than a partial
commutation claim.
