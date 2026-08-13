# Integer prime-factorization verification

[Documentation home](../../../index.md)

`integer.compute.prime_factorization` retains its existing request and result
contracts and returns `COMPUTED` assurance. The operator-authorized
`integer.prime_factorization.verify` capability independently replays one
submitted `{input, candidate}` pair with Python-FLINT and may promote that exact claim to
`VERIFIED`.

## Exact claim and scope

The verifier checks:

> The submitted ascending prime-power list is the complete prime factorization of
> the absolute value of the exact submitted nonzero integer.

The producer accepts one canonical decimal integer string of at most 256
characters and an explicit wall-clock budget. Zero remains not applicable.
Both `1` and `-1` have an empty factor list, and negative integers use the
factorization of their absolute value.

The verification request supplies the exact producer input plus the complete
typed candidate value inline. The verification record binds their canonical
digests, the number-theory semantics, checker identity, checker source digest,
and Python-FLINT runtime.

## Independent replay

The producer runs SymPy `factorint` in an isolated bounded process. The checker
uses the separately maintained Python-FLINT runtime and does not import the
producer, its worker, or SymPy.

Before backend replay, the checker requires canonical integer encoding, a
valid submitted producer budget, exact result fields, positive exponents, prime
bases greater than one, strict ascending order, no duplicates, and equality
between the declared prime-power product and the input absolute value. It then
compares the complete list with Python-FLINT's independent factorization.

## Verification obligation ledger

| Obligation | Independent replay | Failure meaning |
| --- | --- | --- |
| Artifact binding | Recompute and compare claim, semantics, candidate, lineage, witness-envelope, and payload digests. | Reject this evidence; no mathematical conclusion. |
| Input domain | Require one canonical nonzero integer and the exact bounded producer budget shape. | Reject malformed, zero, or unsupported evidence. |
| Result normalization | Require exact fields, canonical positive bases, positive exponents, strict ascending order, and no duplicates. | Reject noncanonical evidence. |
| Reconstruction | Multiply every declared prime power with bounded exact integer arithmetic and require the product to equal the input absolute value. | Reject an incomplete or incorrect factor list. |
| Primality and completeness | Independently factor the absolute value with Python-FLINT and compare every base and exponent. | Reject a composite base, missing factor, wrong power, or extra factor. |
| Sign and unit convention | Replay `±1` as the empty factorization and ignore only the input sign for other nonzero integers. | Reject a mismatched unit or sign convention. |
| Authorization and runtime | Dispatch only the operator-authorized checker matching the schemas, semantics, format, source digest, and pinned Python-FLINT runtime. | Unavailable, timeout, cancellation, or error remains non-conclusive. |

A rejected candidate returns `UNKNOWN`; it does not assert a different
factorization or a mathematical claim about another integer.
