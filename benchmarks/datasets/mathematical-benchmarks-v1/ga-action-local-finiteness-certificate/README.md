# Additive-group local-finiteness certificate

Hard-provisional Regression benchmark derived from `Jiahao004/DeepTheorem`
train row 10019 at revision
`f5935720f176cedff4ecd8ebf83d1696e31cfac8` (MIT). The canonical source-row
SHA-256 is `bce6d08418d3828af667b64fbc99b9701f2d6388ddf0e0e26a7900409ac7d445`.

The source discusses the local-finiteness argument for a coordinate-ring
coaction. This transformed case asks for an explicit invariant-subspace
certificate for the additive-group action `(x,y) -> (x+t*y,y)`. It tests
constructive use of the coaction identity rather than acceptance of the
unsupported sentence that the finite coefficient span is invariant merely
because the coaction is a ring map.

The verifier independently reconstructs sparse polynomials over `QQ`, checks
that the submitted degree-four basis is independent, reconstructs the frozen
polynomial, replays every basis action, and checks the identity and additive
group laws for the submitted polynomial matrix. Any valid rational basis and
its induced action matrix are accepted.

Family: **Regression**. Primary reasoning objective: **construct a replayable
finite invariant-subspace certificate**. Difficulty is Hard (provisional): the
agent must coordinate a non-unique basis, an exact symbolic representation,
and two polynomial matrix identities; empirical baseline calibration is still
pending.

Shortcut audit: the source contains the abstract coefficient-span idea but not
this frozen action, polynomial, basis, or matrix certificate. The verifier does
not compare with the Oracle basis. The task excludes dimensions below five, so
a tiny one-vector invariant witness cannot satisfy the contract.

Assurance is capped at `COMPUTED`: the exact frozen polynomial identities are
replayed, but the general local-finiteness theorem for arbitrary affine
algebraic groups is not formalized or certified.
