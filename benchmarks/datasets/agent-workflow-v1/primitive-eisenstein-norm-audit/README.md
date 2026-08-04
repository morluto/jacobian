# Primitive Eisenstein norm audit

Assurance benchmark derived from Xerv-AI/GRAD train row 90 at immutable revision `71595210590450202b7b69225bc07e9e01b13c5c` (MIT), canonical row digest `sha256:4185b4092ad9c58e3ed2a5e8d88246823f52c03dc6a2391ae624843f851a25d4`.

The source proof gives an incorrect factorization criterion for primitive values of `x^2+xy+y^2`: it excludes the ramified prime 3 and permits even powers of inert primes. The task asks for two independently replayable local certificates exposing those distinct errors.

Family: Assurance. Primary objective: proof diagnosis. Quality score: 89/100. Difficulty: Hard (provisional), based on two local arguments, exact residue reasoning, and repair of a prime-factor classification; baseline calibration remains pending.

Shortcut audit: a copied corrected criterion, a single tiny witness, or a factorization label cannot pass. The verifier accepts varied primitive ramified witnesses, requires a freely chosen inert prime, independently enumerates its full residue square, and binds the evidence bytes. It does not adjudicate the source's separate cubic-form intersection count.

Assurance is `COMPUTED`: the checker proves the submitted finite local certificates and their exact arithmetic, but no proof assistant certifies the general algebraic-number-theory classification.
