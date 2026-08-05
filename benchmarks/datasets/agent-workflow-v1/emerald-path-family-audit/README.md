# Emerald path family audit

Hard-provisional Assurance benchmark from ProofBench APMO-2025-2. A generated proof
incorrectly concludes that only `(1,1)` works. The verifier accepts any normalized positive
rational pair `alpha > beta` with sum two and independently checks the general parity offsets,
the band constraint, and a frozen exact trace.

This tests recovery of a missed parameter family rather than another isolated counterexample.
The public model answer is not a shortcut because it is the object being refuted. Assurance is
`COMPUTED`; necessity for every possible trip is outside the certificate's declared scope.
