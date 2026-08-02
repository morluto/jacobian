# Audit RSA exponent reduction over every residue class

The frozen source argues that reducing a positive RSA private exponent `d`
modulo `p-1` preserves `C^d mod p`. Its displayed manipulation introduces a
negative exponent and silently relies on `C` being invertible modulo `p`.
RSA ciphertext residues need not be units.

Submit a domain-complete repair certificate for an odd prime `p`, positive
`d` with `gcd(d,p-1)=1`, and least nonnegative remainder `d_p`.

Your certificate must address the domain gap and support the universal conclusion with an independently replayable symbolic certificate. It must:

1. diagnose why the inverse-based step is not defined for nonunits;
2. derive `1 <= d_p <= p-2` from the stated assumptions;
3. prove both residue cases without introducing an undefined inverse;
4. state the exhaustive domain split and why each branch is valid;
6. provide freely chosen unit and nonunit numeric witnesses satisfying the
   frozen bounds; and
7. distinguish the symbolic repair from bounded sanity checks.

The verifier independently checks the witnesses and exhaustively tests all
eligible residues for odd primes up to 43 and exponents up to 80. Those tests
are sanity evidence only; acceptance also requires the symbolic branch
certificate.

Write `/app/submission.json` and bind a concise explanation at
`/app/evidence/answer.txt` by SHA-256. Do not claim `VERIFIED`: this task does
not replay a proof assistant or certify the universal theorem beyond the
explicit certificate checker.
