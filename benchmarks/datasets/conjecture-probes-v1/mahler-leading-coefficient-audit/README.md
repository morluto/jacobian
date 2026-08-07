# mahler-leading-coefficient-audit

An Assurance benchmark derived from retained row C-018. It diagnoses a trust-boundary error in a partial Mahler-measure formula and repairs it with exact factorization and arithmetic in `Q(sqrt(5))`.

The verifier does not use floating roots or compare a memorized scalar: it reconstructs the polynomial, validates every primitive factor, derives the outside-root contributions, and multiplies radical pairs exactly. The task excludes tiny linear examples and caps assurance at one frozen polynomial. Difficulty is provisionally Hard because factor discovery, root classification, and exact normalization must all agree; baseline calibration is pending.
