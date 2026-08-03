# Certify a cyclic Lipschitz optimum

Read `/app/input.json`. Maximize the sum at the five marked positions over real cyclic sequences satisfying the zero-sum and adjacent-difference constraints.

Submit the standard envelope. Give a canonical rational 60-entry feasible sequence and a canonical rational 60-edge circulation `q`. With indices modulo 60, require `q_i-q_(i-1)=w_i`, where `w_i=11/12` at a marked position and `-1/12` otherwise. Its `L1` norm is the dual value. Bind a concise derivation in `evidence/answer.txt`.

The verifier checks primal feasibility/value, flow divergence, exact `L1` cost, and independently recomputes the minimum circulation cost from cumulative imbalances and their median. Assurance remains `COMPUTED`.
