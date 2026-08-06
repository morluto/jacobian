# Marginal versus joint product-convergence audit

This Assurance benchmark transforms a public probability discussion into an exact finite-law countermodel. The primary reasoning objective is to diagnose why marginal convergence and independence of each prelimit pair do not, by themselves, identify the joint law of the named limit pair.

The agent must construct a non-product coupling with the frozen four-point marginals, replay the independent prelimit product law, and demonstrate that the two induced product distributions differ. The verifier independently parses every rational mass, checks both marginals, checks prelimit independence, recomputes both pushforward product laws, and accepts any valid non-product coupling.

## Curation

- **Family:** Assurance.
- **Quality score:** 87/100.
- **Difficulty:** Hard (provisional): the task requires a four-stage coupling argument, exact 16-cell probability bookkeeping, and pushforward-law comparison. Baseline calibration may revise this label.
- **Portfolio value:** adds marginal-versus-joint convergence and dependence calibration; it is not another convergence-mode implication, finite witness search, or fixed proof replay.
- **Shortcut audit:** a two-point Rademacher witness is disallowed by the frozen four-point nonuniform marginal. The public response's invalid CDF multiplication is not an answer key. The verifier rejects copied labels, malformed mass tables, and product-law claims not supported by the submitted coupling.
- **Nearby rejections:** the same source's routine transform and elementary distribution rows were rejected as shallow calculation; its convergence-in-probability examples overlap existing convergence-mode coverage.

## Provenance

- Dataset: `Jiahao004/DeepTheorem`
- Revision: `f5935720f176cedff4ecd8ebf83d1696e31cfac8`
- Split/row/source id: `train`, `10044`, `87356`
- License: MIT
- Frozen row digest: recorded in `environment/input.json`

## Trust boundary

The verifier establishes only the exact finite probability-table countermodel and the resulting failure of the claimed implication under the frozen interpretation. It does not machine-check a general weak-convergence theorem, infer the intended meaning of ambiguous prose, or certify any claim above `COMPUTED`.
