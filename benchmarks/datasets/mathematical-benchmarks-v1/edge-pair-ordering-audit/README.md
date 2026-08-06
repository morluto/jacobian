# Edge-pair ordering audit

Hard provisional Regression benchmark derived from Xerv-AI/GRAD (MIT), train row 98, pinned revision `71595210590450202b7b69225bc07e9e01b13c5c`. The public solution derives an ordered-pair factor and then incorrectly halves it by reinterpreting the defining double sum as unordered.

Score: 89/100. The workflow adds semantic interpretation of mathematical notation, local proof diagnosis, symbolic counting, and independent exhaustive finite replay. Copying the public final answer fails. The verifier checks all graphs through six vertices, rather than a tiny fixed witness, but the all-`n` conclusion remains a combinatorial argument outside machine proof; hence `COMPUTED`.
