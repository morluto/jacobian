# Triplewise-empty extremal audit

Hard provisional Regression benchmark from Xerv-AI/GRAD (MIT), train row 59, revision `71595210590450202b7b69225bc07e9e01b13c5c`. Score: 88/100.

The source's `2n` answer ignores the cost of keeping subsets distinct. An extremal family contains the empty set, all singletons, and a maximum matching of pairs, giving `1+n+floor(n/2)`. Copying the public answer fails. Alternative matchings are accepted across odd/even probes. The workflow adds general incidence-bound reasoning plus constructive extremality, with a `COMPUTED` ceiling.

