# Repair a quartic Chebotarev density proof

Audit the frozen proof for `f(x)=x^4-4x+1`. Submit one exact certificate that repairs irreducibility, the discriminant, the Galois-group argument, and the fixed-point count used by Chebotarev.

Your result must include:

- the ascending coefficients of `f(x+1)` and an Eisenstein prime;
- the exact discriminant and prime-power factorization;
- an unramified prime and a monic linear-times-irreducible-cubic factorization modulo that prime (coefficients ascending and interpreted modulo the prime);
- the resulting transitive Galois group;
- a complete `S4` fixed-point cycle count;
- the reduced density and encoded answer; and
- every concrete defect in the frozen explanation.

Write `/app/submission.json` using the supplied schema and bind `/app/evidence/answer.txt`. Do not claim `VERIFIED`: the checker replays exact algebra and finite group enumeration, while Chebotarev and the stated elementary Galois lemmas remain trusted.
