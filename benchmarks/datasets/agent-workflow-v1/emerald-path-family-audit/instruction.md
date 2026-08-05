# Audit the claimed uniqueness of an APMO path construction

The frozen ProofBench model response concludes that the only possible parameters are `alpha=beta=1`. Audit that conclusion.

Submit a nontrivial positive rational pair with `alpha >= beta`, `alpha+beta=2`, and `alpha != beta`. Use the path `x_n=ceil(n/2)`, `y_n=floor(n/2)`. Provide an exact parity certificate showing that `floor(x_n alpha+y_n beta)=n` for every nonnegative integer `n`, and a 16-row exact trace. Bind one explanatory evidence file.

Do not claim `VERIFIED`. The checker replays exact rational parity identities and the finite trace, but does not formalize the full original olympiad theorem.
