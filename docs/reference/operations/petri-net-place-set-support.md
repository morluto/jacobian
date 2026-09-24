# Petri-net place-set support profiles

`petri_net.place_set.support_profile.compute` takes a finite ordinary weighted
place/transition net and a subset `S` of its place axis. It returns the exact
sets

- `producers_into`: transitions with a positive post-arc into at least one
  place in `S`;
- `consumers_from`: transitions with a positive pre-arc from at least one
  place in `S`;
- `siphon_offenders = producers_into \\ consumers_from`; and
- `trap_offenders = consumers_from \\ producers_into`.

The subset is a siphon exactly when its offender set is empty, and is a trap
exactly when its trap offender set is empty. These use structural arc support;
arc weights do not change either predicate. The empty set satisfies both
predicates vacuously.

This operation checks one selected subset and reports every supporting
transition. It does not enumerate subsets or make claims about markings or
reachability. The net and subset must use matching place axes.

`petri_net.place_set.initial_marking_profile.compute` combines this exact
support profile with a source marking and returns the selected places' initial
token total. Its `preservation_implications` data lists `EMPTY_SIPHON_REMAINS_EMPTY`
when the selected set is an initially empty siphon, and
`MARKED_TRAP_REMAINS_MARKED` when it is a trap containing at least one initial
token. Each statement applies along every finite firing sequence from that
marking. The result retains the source net, selected places, marking, and full
support profile; it does not claim to enumerate reachable markings.

`petri_net.siphon_trap_family.enumerate.compute` enumerates every **nonempty**
siphon and trap on the exact place axis. Siphon membership is the condition
that every transition producing into the subset also consumes from it; trap
membership is the reverse condition. Arc weights matter only through whether
an arc is positive. The operation admits the complete subset scan and a
worst-case output bound before enumeration. It includes nonminimal families;
`petri_net.siphon_trap.check` continues to return only inclusion-minimal
families.

## Example

For a one-place net with one transition consuming and producing one token,
`places = [0]` has `producers_into = [0]`, `consumers_from = [0]`, and no
offenders, so it is both a siphon and a trap.
