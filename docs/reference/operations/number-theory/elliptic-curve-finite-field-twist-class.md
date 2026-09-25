# Finite-field elliptic-curve twist-class decision

`elliptic_curve.finite_field.twist_class.decide` compares nonsingular short
Weierstrass curves over the same exact finite-field presentation, in
characteristic greater than three.

For unequal `j`-invariants it returns `DIFFERENT_J`. For equal `j`, it accepts
only the generic case `j` outside `{0, 1728}`. Such a curve has no extra
automorphisms over the algebraic closure, so its finite-field forms have two
classes: the isomorphism class and its nontrivial quadratic twist. The
operation performs a complete bounded scaling search to choose between them.
An isomorphic result includes the exact scaling. A quadratic-twist result
includes the canonical nonsquare twist relation and an exact scaling from the
twisted model to the target.

The searches require field order at most 4096 and are admitted against one
aggregate exact-work and output-size bound. Curves with `j = 0` or `j = 1728`
are rejected because their larger automorphism groups require explicit
classification branches. Curves over different field presentations are also
rejected; callers must first provide a supported field transport.

The two-class statement follows from the classification of twists by
`F_q^*/(F_q^*)^d`, where `d=2` for `j` outside `{0,1728}`. See Lidl and
Niederreiter (eds.), *Handbook of Finite Fields*, §12.2.57, for this
classification and the corresponding short-Weierstrass model:
<https://archive.ymsc.tsinghua.edu.cn/pacm_download/672/12637-dingjt-p2.pdf>.

This operation decides the relation between two supplied models. It does not
enumerate all twist classes or classify the exceptional `j` values.
