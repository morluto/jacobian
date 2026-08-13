# illumination-strictness-audit

An Assurance benchmark from retained row C-083. It detects a boundary strictness bug in an illumination evaluator, exhaustively records every false positive, repairs the certificate, and proves the exact eight-direction minimum for the cube by sign-pattern injectivity.

The verifier checks all vertex–direction pairs and accepts arbitrary orderings and alternative valid tangent/strict direction sets. The four-direction flawed pass excludes the zero-vector shortcut; the eight-direction repair must provide a bijective coverage map. Difficulty is provisionally Hard because semantic diagnosis, exhaustive evidence, construction, and minimality must all align; calibration is pending.
