# Positive-root length profiles

`root_system.root_length_profile.compute` returns every positive root grouped by
its exact squared length, separately within each irreducible component. Each
component's positive symmetrizer is normalized so its first simple root has
squared length 2. The result gives both those exact lengths and each class's
squared-length ratio to the shortest class in the same component. Ratios are
component-local because a reducible root datum has no canonical relative scale
between distinct factors.

For a symmetrizer `D = diag(d_i)` and Cartan matrix `A`, the convention is
`(alpha, alpha) = c^T D A c` for root coordinates `c`. The exact class
partition agrees with the squared lengths returned alongside roots by
`root_system.coroots.compute`; it is also the same bilinear form used by
`root_system.root_to_coroot.compute`.

Finite Cartan matrices through rank 8 are admitted. The complete positive-root
family has at most 120 entries; work and worst-case serialized output are
admitted before root enumeration. No partial profile is returned.
