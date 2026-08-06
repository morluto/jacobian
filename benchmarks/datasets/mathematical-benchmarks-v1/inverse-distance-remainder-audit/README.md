# Inverse-distance remainder audit

This task freezes DeepTheorem default train row 5001 at Hugging Face revision
`f5935720f176cedff4ecd8ebf83d1696e31cfac8`. The negative variant reaches the
right conclusion that a cubic remainder is false, but its derivation
incorrectly treats `epsilon^2` as fourth order and drops an explicit
second-order contribution. The canonical frozen source row digest is
`sha256:c5bfe234c517c99357fbabc3325bb1289829822aa3db7908ff40a9e191e76497`.

The verifier independently derives the normalized directional coefficient
`(3*u_1^2-1)/2`, checks the invariant second-order term, and accepts any two
rational unit directions producing opposite nonzero coefficient signs. This
makes the certificate answer-flexible while proving that the residual is
generically quadratic. The checker does not trust the dataset response and
does not claim external proof-assistant verification.

DeepTheorem is distributed under the MIT license. The frozen row records
`stackexchange-math` as its upstream source; downstream users remain
responsible for any upstream attribution requirements.
