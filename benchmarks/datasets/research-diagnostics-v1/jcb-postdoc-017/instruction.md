# Exact LP certificate for the pairwise-independent correlation-gap counterexample

Let N={1,2,3,4,5}. Elements cover E1={A,B}, E2={A,B,C,D}, E3={C,D}, E4={A,C}, E5={B,D}, and f(S) is the number of covered features. Let x=(3/10,7/20,3/10,7/20,7/20). For distributions with these marginals let f+(x) be the maximum E[f(S)]; for pairwise-independent distributions add constraints P(i,j in S)=x_i x_j and call the optimum f++(x). Verify f+(x)=4 using masses 3/10 on {1,3}, 7/20 on {2}, and 7/20 on {4,5}. Verify the dual upper certificate lambda0=1/2; lambda1=lambda3=lambda4=lambda5=3/2; lambda2=7/2; lambda12=lambda23=-1; lambda24=lambda25=-3/2; lambda13=lambda45=1/2; lambda14=lambda15=lambda34=lambda35=-1/2; all unspecified pair coefficients zero. Prove lambda(S)>=f(S) for all 32 subsets, compute its objective 479/160, and conclude f+(x)/f++(x) >= 640/479 > 4/3.

This is a public answer-visible diagnostic: the expected conclusion and oracle summary are
public. Reproduce the answer-visible conclusion, state the relevant answer-visible facts, and
report the capability boundary honestly. Generic LP evidence still needs an inspectable encoding that binds variables and constraints to the coverage-function and pairwise-independence semantics. Solver status alone is not VERIFIED.

Write `submission.json` to the exact agent-visible `submission_schema.json`, record your
reasoning in `evidence/answer.txt`, and include that file's SHA-256 digest in the evidence
list. In `result.key_facts`, use a nonempty object with lower-snake-case fact
names and nonempty string values; do not rely on prose or numeric/boolean
coercion. You may claim `UNVERIFIED` or `COMPUTED`; only `COMPUTED`
receives full aggregate credit. Do not claim `VERIFIED`. Treat timeout,
error, or incomplete search as a non-conclusion, not as evidence for or
against the claim.
