# RSA exponent-reduction domain audit

This Regression-family task is derived from `Jiahao004/DeepTheorem` train row
7 (source id 780), pinned at commit
`f5935720f176cedff4ecd8ebf83d1696e31cfac8` under the MIT license.

The primary objective is proof diagnosis: repair a unit-only exponent argument
so that it covers every ciphertext residue. A complete answer must separate
unit and nonunit cases, derive positivity of the reduced exponent, avoid
undefined inverse powers, and bind alternative numeric witnesses. A tiny
counterexample or a copied source proof cannot pass because neither supplies
the universal two-branch certificate.

Difficulty is **Hard (provisional)**: the reasoning chain combines assumption
recovery, modular-domain analysis, a two-case repair, and assurance
calibration. We expect weaker agents to repeat Fermat's theorem without
handling nonunits, stronger agents to construct the full split, and tool-less
agents to remain viable but error-prone. No empirical baseline is available.

The standard-library verifier checks witness arithmetic, the exact symbolic
contract, and all eligible residue classes for odd primes through 43 and
exponents through 80. This is `COMPUTED`, not a proof-assistant verification of
the universal theorem. The source row supplies provenance, not authority.
