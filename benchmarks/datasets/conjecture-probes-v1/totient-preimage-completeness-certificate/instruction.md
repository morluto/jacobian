# Classify every integer with Euler totient 48

Submit a complete, duplicate-free classification of all positive integers
`n` satisfying `phi(n)=48`. The certificate must not rely on an arbitrary
search cutoff. Provide the factorization and exact totient contribution for
every accepted integer. You may optionally include a candidate-prime set,
prime-power exponent options, and a branch count as supporting completeness
evidence, but these fields are not required: any complete classification that
matches the independently reconstructed inverse image is accepted.

The verifier independently tests primality, reconstructs the candidate primes,
derives exponent options, enumerates the finite search internally, recomputes
every totient, and checks exact equality with the submitted classification.
Candidate-prime, exponent-option, factorization, and solution lists are treated
as mathematical collections, so equivalent orderings are accepted. This is a closed
finite preimage audit and makes no claim about whether infinitely many totient
values have a unique preimage.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
