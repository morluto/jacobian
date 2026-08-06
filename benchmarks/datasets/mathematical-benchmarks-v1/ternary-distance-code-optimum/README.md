# Ternary distance-code optimum

This Hard (provisional) Regression benchmark turns a six-question, three-choice
answer-pattern problem into a ternary coding-theory certificate. The agent must
construct 18 words whose distinct pairs agree in zero or two coordinates and
must independently certify optimality with two exact Krawtchouk inequalities
and a nonnegative dual combination.

The verifier recomputes every pair distance, the average distance distribution,
the Krawtchouk values, and the dual coefficient identity. It accepts any valid
18-word code. It does not invoke a formal proof assistant, so assurance is
capped at `COMPUTED`.
