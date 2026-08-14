# Bounded finite coverage verification

[Documentation home](../../../index.md) · [Tool surface](../../tools.md)

`finite.coverage.verify` verifies that a bounded paged archive contains every
member of an explicit finite scope exactly once. It binds typed archive pages
and independently replays an exactly-once archive claim.

## Bounded v1 contract

Version 1 accepts:

- one non-empty scope of at most 4096 items;
- between 1 and 64 archive pages;
- at most 1024 items per page and 4096 archive items in total;
- either `finite.string.nfc@1` or `finite.integer.decimal@1`.

The canonicalizer IDs are versioned registrations. The string registration
uses Unicode NFC semantics; the integer registration uses exact decimal
integer semantics. Canonical keys use a SHA-256 digest over a tagged,
canonically encoded value. Unknown canonicalizer IDs and mixed item types are
request errors. Canonically equivalent scope members are rejected because the
scope must identify a finite set unambiguously.

## Artifacts and bindings

An invocation materializes:

1. the selected canonicalizer registration;
2. a typed finite-scope artifact with unique canonical keys;
3. one typed artifact per archive page;
4. an archive manifest binding each page URI, object digest, payload digest,
   item digest, index, and item count;
5. an exactly-once claim and certificate;
6. a verification record only when the authorized checker accepts.

The scope-key digest, page bindings, archive digest, claim, candidate, scope,
semantics, and canonicalizer specification are all bound into the certificate
or its artifact lineage. Replacing or reordering a page therefore changes a
bound digest.

## Diagnostics and assurance

The output reports canonical keys for omissions, items outside the scope, and
duplicates. Duplicate occurrences include page and item indices. These
diagnostics are computed by the producer for inspection, but they do not
certify the result.

`VERIFIED` is returned only after the operator-authorized independent checker
recomputes canonical keys, validates every registration and digest binding,
checks contiguous page indices and bounds, and establishes that every scope
key has count exactly one. Invalid coverage returns `UNKNOWN`, leaves the
claim obligation open, and does not create a verification record.

The operation does not infer a finite scope, stream unbounded archives,
accept caller-defined canonicalization code, or prove that an external data
source was completely exported into the supplied pages.
