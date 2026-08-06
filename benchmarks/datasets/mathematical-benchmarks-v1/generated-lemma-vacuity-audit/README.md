# Generated lemma vacuity audit

This task freezes two generated intermediate lemmas from `Tencent-IMO/IMO-Lemmas`
at revision `ad57c36fc5d99010b4ff71d98616b3b6c2b11ba3` (train rows 7 and 11).
It measures whether an agent can distinguish a true or satisfiable lemma from a
lemma that advances the source proof. The verifier independently evaluates the
submitted integer witnesses and corrected quantifier structure. It does not run
Lean and does not assess the truth of either original IMO theorem.

The source `lemmas.jsonl` file has SHA-256
`c92fbbe3e5179cb2d6430a04c5ebb96def08c94fbd2f76dd5e1a9a09decd0a9f`
and is licensed Apache-2.0.
