# Necklace Burnside certificate

This **Regression** benchmark transforms BeyondAIME row 23 into a complete finite orbit certificate. Its single primary objective is algorithmic symmetry reasoning: the agent must enumerate constrained cyclic words, derive all 32 fixed-point counts, apply Burnside's lemma, and expose the complete canonical orbit partition.

## Curation and quality

- **Selected because:** it adds executable group-action reasoning rather than another proof-label or bounded-witness task.
- **Workflow novelty:** no existing benchmark combines a cyclic local constraint, dihedral actions, a fixed-point table, and a complete orbit partition.
- **Shortcut audit:** the public answer `88` is insufficient; the verifier rejects a correct count with a corrupt fixed table or missing representative.
- **Difficulty:** Hard, provisional. The state space is manageable but requires a correct cyclic predicate, two action families, orbit canonicalization, and consistency across four certificate layers.
- **Expected discrimination:** weak agents often omit wraparound triples or conflate reflection conventions; stronger agents can implement and cross-check the complete action.
- **Quality score:** 87/100.

The hidden verifier uses only the frozen finite instance. It reports `COMPUTED`, not a general theorem proof.
