# Repository action policy

## Search ownership first

Search current main, branches, issues and comments, PRs of every state, saved
evaluations, and copied implementation lineages.

## Outcomes

- **No action:** product behaved correctly or failure is model-local.
- **Existing issue evidence:** independent evidence strengthens an owned root cause.
- **Documentation clarification:** factual guidance is missing but behavior is sound.
- **Maintainer issue:** strong architectural question, cross-family operation gap, or policy choice.
- **Localized draft PR:** small reproducible defect with an unambiguous safe correction.

## Draft-PR gate

Open a draft PR only when:

- the failure reproduces on current main;
- no existing PR owns the correction;
- the change is localized and low risk;
- verification, assurance, and binding remain intact;
- focused tests cover the failure and safe negative cases;
- the change-aware plan and relevant checks pass.

Use an isolated branch or worktree, commit only scoped files, describe source
independence and overlap, and never merge.

## Operation-gap threshold

Require independent evidence from at least two mathematical families unless
the missing primitive is already accepted repository architecture. Do not
count repeated executions, parameter variants, copied adapters, or one
benchmark family as independent.

## Attribution guard

Do not file against Jacobian for a model-transcribed wrong input, local fallback
mistake, honest stop on an unsupported domain, unreplicated transport
cancellation, model-authored false `VERIFIED` label, or benchmark-specific
convenience request.
