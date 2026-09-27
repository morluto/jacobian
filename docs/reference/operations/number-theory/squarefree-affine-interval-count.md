# Simultaneous square-free affine interval counts

For a bounded family of integer affine forms \(L_j(n)=a_jn+b_j\), this
operation counts integers \(n\in[\ell,u]\) for which every \(L_j(n)\) is
square-free. An optional ledger lists matching integers and the least prime
whose square divides a form value at each rejected integer.

The exact prime-square sieve uses primes \(p\le\sqrt{\max_{j,n}|L_j(n)|}\).
Admission bounds both the number of congruence classes and the maximum number
of interval points traversed along those classes before the sieve expands
them. The current limits are 4,000,000 classes and 2,000,000 bounded visits.
Classes on which a form is identically zero modulo \(p^2\) are charged once
for each interval point because the kernel marks the interval directly.

This finite interval count makes no density or infinitude claim.
