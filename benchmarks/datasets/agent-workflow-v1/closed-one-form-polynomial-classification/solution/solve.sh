#!/bin/sh
set -eu
mkdir -p /app/evidence
cat > /app/evidence/answer.txt <<'EOF'
CHAIN_RULE: d/dx f(y,x) = f_y(y,x)
CONSTRAINTS: a_11 - 2*a_02 = 0; a_21 - 3*a_03 = 0
RANK: 2
DIMENSION: 8
POTENTIALS: Every listed potential satisfies F_x=f(x,y) and F_y=f(y,x).
LIMITATION: The analytic Poincare lemma and arbitrary smooth forms are not checked.
EOF
cp /solution/submission.json /app/submission.json
