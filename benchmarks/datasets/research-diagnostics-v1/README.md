# Jacobian research-diagnostics-v1

This Harbor dataset contains public, answer-visible research challenges. Each
challenge is one self-contained task under
this dataset directory as `jcb-postdoc-<id>/` task bundles.

The prompt and permitted runtime context are agent-visible; source answers,
Oracle summaries, and verifier material remain outside the agent image.
Per-challenge portfolio status (`historical_fit`, `current_status`,
`evaluation_status`, and `next_action`) is retained in each task's metadata and
maintainer README. `suite.toml` owns stable policy, member records own
membership, and immutable snapshot locks record Harbor task digests when an
evaluation is intentionally frozen. Run the Oracle
contract gate with:

```sh
make harbor-oracle DATASET=research-diagnostics-v1
```

Oracle success establishes solvability and verifier integrity. Results are
case-level diagnostics only, never held-out evidence or a comparative model
performance claim; all tasks are capped at `COMPUTED` assurance.
