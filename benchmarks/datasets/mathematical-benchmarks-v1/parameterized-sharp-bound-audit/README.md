# Parameterized sharp-bound audit

This task freezes IneqMath train row 53 at Hugging Face revision
`3c7c32c786eb77117f3476d7f6d9af8419fa6ecc`. The published answer gives the
correct piecewise bound, but its explanation blurs an attained symmetric
minimum with a boundary infimum that is unavailable under strict positivity.

The verifier independently checks the transition value, the two affine
polynomial decompositions, the symmetric equality case, and a submitted
permutation of the boundary limiting family. It does not trust the dataset's
answer or claim a machine-checked theorem beyond this exact certificate family.

The dataset contribution is CC BY-SA 4.0. The frozen record and its upstream
problem content retain the attribution and use conditions recorded by
IneqMath.
