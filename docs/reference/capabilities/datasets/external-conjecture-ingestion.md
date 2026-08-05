# External conjecture ingestion

[Documentation home](../../../index.md) · [Capability surface](../../tools.md)

`dataset.conjecture.ingest` applies a versioned publication policy to one
external conjecture record before it enters Jacobian research artifacts. The
operation is reusable across corpora; OpenConjecture is the first evaluated
source.

## Bound provenance

Every artifact binds:

- corpus ID and immutable revision;
- source URL and item ID;
- canonical record digest;
- normalized supplied-content digest, when text is present;
- source-license classification;
- license-evidence URL, evidence text digest, and policy version; and
- the exact text-publication or metadata-only decision.

An optional expected record or content digest makes replay fail closed when a
source row changes. License evidence text is checked against its declared
digest but is not copied into the output artifact.

## Publication policy

Policy `jacobian.external-conjecture-publication/v1` allows statement text only
for the registered `CC0-1.0`, `CC-BY-4.0`, `Apache-2.0`, and `MIT`
classifications and only when both a license-evidence URL and verified evidence
digest are supplied. Blank or incomplete evidence tuples are rejected.

`CC-BY-NC-4.0`, `CC-BY-ND-4.0`, proprietary, restricted, unknown, and missing
license classifications produce `METADATA_ONLY`. In that state:

- provenance and normalized metadata are retained;
- supplied text receives a digest so the decision remains source-bound;
- statement text and its indexable digest are withheld; and
- the output records an explicit withholding reason.

A metadata record for which no statement was supplied uses
`METADATA_INDEXED_NO_TEXT`; it is not mislabeled as policy-withheld content.
Because an input request can contain restricted text and raw license evidence,
this capability writes only safe metadata, the policy decision, and only text
approved for indexing into its durable artifact. Restricted statement and
evidence text never enter the artifact payload.

The policy artifact schema is producer-only: generic `artifact.put` calls
cannot issue an `ALLOW_TEXT` decision because they do not possess the
capability's validated license-evidence context.

This mirrors OpenConjecture's public-release pattern of publishing permitted
text while retaining more restrictive sources as metadata-only records. The
policy is repository code, not a claim that any caller-supplied license label
is legally correct; operators remain responsible for source review.

## Example

```json
{
  "capability_id": "dataset.conjecture.ingest",
  "mode": "EXPLORE",
  "input": {
    "corpus_id": "davisrbr/openconjecture",
    "corpus_revision": "f665b46c93a6a1d505ef9109417902d7b2973ab8",
    "source_url": "https://huggingface.co/datasets/davisrbr/openconjecture",
    "item_id": "fixture-1",
    "metadata": {
      "title": "Fixture conjecture",
      "domain": "number theory"
    },
    "statement": "Every fixture prime has the fixture property.",
    "source_license": "CC-BY-4.0",
    "license_evidence_url": "https://example.invalid/license/fixture-1",
    "license_evidence_text": "Creative Commons Attribution 4.0",
    "license_evidence_digest": "sha256:14cf6e4efc51a33be0438483f0bc0d53963cedad7406282e331b3f797779cc11"
  }
}
```

## Trust boundary

Every imported conjecture is `HEURISTIC` and `UNVERIFIED`. Ingestion never
establishes truth, proof, formal correctness, or informal-formal
correspondence. Indexing and retrieval must preserve these labels, and only an
independent verifier acting on an exact claim may later produce verified
evidence.
