# Additive-combinatorics decisions

[Documentation home](../index.md)

- Status: Current implementation reference; contracts are pre-stable
- Related reference: [Domain operation library](domain-operation-library.md)

Jacobian exposes three bounded, atomic outcomes for Sidon and cyclic
perfect-difference-set work. They support exact finite reasoning without
claiming a general design-theory solver or a universal obstruction theorem.

## Integer Sidon decision

`combinatorics.integer_set.sidon.decide` accepts at most 32 distinct canonical
integer strings. Its result sorts the set and records one row for every ordered
pair of different elements. The result is Sidon exactly when every recorded
difference is distinct. The producer is deterministic standard-library code
and returns `COMPUTED` assurance.

When authorized,
`combinatorics.integer_set.sidon.verify` independently reconstructs the whole
ordered profile in a bounded checker process. The inline verification request
contains the original typed request and the candidate result.

## Cyclic perfect-difference-set decision

`combinatorics.cyclic_difference_set.perfect.decide` accepts a canonical
residue set and a modulus no larger than 4096. It returns the multiplicity of
every residue from 1 through `modulus-1`, the missing and repeated residues,
the set order `k`, and the expected perfect modulus `k(k-1)+1`.

The decision is true only when the submitted modulus equals that expected
modulus and every nonzero residue occurs exactly once. The independent
`combinatorics.cyclic_difference_set.perfect.verify` capability rebuilds this
complete profile rather than accepting the Boolean field alone.

## Fixed-order extension decision

`combinatorics.cyclic_difference_set.extension.decide` asks whether the
reduced residues of one integer base set are directly contained in a perfect
difference set of one specified order `k`. The modulus is not arbitrary: it is
derived as `k(k-1)+1`. Translation, affine equivalence, alternative embeddings,
and universal quantification over `k` are outside this contract.

The request is accepted only when all of these bounds hold:

- `2 <= k <= 64` and the derived modulus is at most 4096;
- at most three residues must be added; and
- the complete candidate space contains at most 50,000 combinations.

A positive result carries the complete extension witness. A negative result
carries `coverage = ALL_CANDIDATES`, an empty witness, and the exact binomial
candidate count. The result is a typed artifact, so discovery can match its
schema to the verifier's `TYPED_ARTIFACT` input contract.

The producer uses difference-conflict pruning. The independent
`combinatorics.cyclic_difference_set.extension.verify` checker does not import
that code or reuse its search: it enumerates the declared combination space
directly and checks each candidate's complete residue profile. Only an
operator-authorized successful replay is eligible for product `VERIFIED`.
Unavailable authority, timeout, malformed output, scope mismatch, and checker
rejection remain non-conclusions.

## Public research diagnostic

`jacobian/jcb-postdoc-015` is a public answer-visible reproduction for the
five-element set `{1,2,4,8,13}`. Its clean-room Harbor verifier checks all 20
integer differences and exhausts orders 5, 6, and 7. The task explicitly marks
the public universal obstruction as not replayed: those three finite checks do
not prove that the set is absent from every finite perfect difference set.

The public Oracle validates the benchmark contract and the bounded operations.
It is not held-out evidence of model improvement. Protected transformed cases
and matched Jacobian-on/off runs remain a separate evaluation stage.
