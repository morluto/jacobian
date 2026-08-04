# jacobian/jcb-postdoc-016

Research diagnostic for Erdős Problem 364. The task requires exact
factorizations and powerful-number decisions for 8 through 16, followed by
structural checks of all seven consecutive triples in that window.

## Provenance

- case_version: research-diagnostics-v1
- contamination_class: public-answer-visible-diagnostic
- fixture_digest: sha256:1e8f2277d84016562c4fea73ca77e0beb1304f2628614be160143b43533ba1b6
- derivation: The open universal conjecture and public 10^14 reference are separated from an independently replayable local window with complete factorizations and seven consecutive-triple checks.

## Portfolio status

- historical_fit: `MISSING`
- current_status: `PARTIAL`
- evaluation_status: `RUNNABLE_PUBLIC_REPRODUCTION`
- next_action: Run public Oracle validation; keep the 10^14 artifact replay as a separate provider- and format-bound proposal.

## Contract

- schema_version: 1.4
- difficulty: hard
- maximum_assurance: COMPUTED
- agent-visible verification record schema: no
- timeout_sec (agent): 600.0
- timeout_sec (verifier): 120.0
- environment_mode: separate

The clean-room verifier independently factors every value and reconstructs
each triple. The public `10^14` artifact is not bundled or replayed, and the
unbounded conjecture remains explicitly open. This is a public regression and
scope-calibration diagnostic, not held-out causal evidence.
