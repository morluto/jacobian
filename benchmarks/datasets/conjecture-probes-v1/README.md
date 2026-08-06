# Jacobian conjecture-probes-v1

This Harbor dataset contains bounded finite probes of open mathematical
conjectures. Each probe is one self-contained task bundle under this dataset
directory as `<task-id>/`.

A conjecture probe fixes a small finite instance family, asks the agent to
compute exact witnesses (counts, domination numbers, bound checks) over that
family, and binds the result through a digest-bound evidence artifact. The
verifier independently recomputes every value from the frozen input using only
the Python standard library and rejects malformed submissions, false
certification, and incomplete scope. Probes are capped at `CHECKED` assurance:
the verifier checks the finite certificate, but a bounded probe is never a proof
of the open conjecture.

The prompt and permitted runtime context are agent-visible; source answers,
Oracle summaries, and verifier material remain outside the agent image.
`suite.toml` owns stable policy, member records own membership, and immutable
snapshot locks record Harbor task digests when an evaluation is intentionally
frozen. Run the Oracle contract gate with:

```sh
make harbor-oracle DATASET=conjecture-probes-v1 FULL=1
```

Oracle success establishes solvability and verifier integrity for the selected
probe. Results are case-level computational evidence only, never held-out
evidence or a comparative model performance claim.
