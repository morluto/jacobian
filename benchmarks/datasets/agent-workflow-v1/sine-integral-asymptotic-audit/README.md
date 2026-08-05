# Sine-integral asymptotic audit

Hard provisional Assurance benchmark derived from HARDMath2 train row 2. The published
answer has the wrong sign on `sin(x)/x^2`. The task requires a five-term symbolic tail
identity, its `Si` negation, and an exact remainder bound. The verifier differentiates the
submitted expression coefficient-by-coefficient and accepts reordered equivalent term lists.

This adds dataset-answer auditing for a subtle symbolic sign error rather than another
bounded witness or fixed proof-label task. The public two-term answer is not a shortcut: it is
incorrect, while the benchmark requires a longer independently replayable certificate.

Assurance is capped at `COMPUTED`: the verifier replays the formal identity and bound under
standard differentiation, FTC, and oscillatory-tail lemmas; it does not machine-prove those
calculus lemmas or arbitrary transcendental asymptotics.
