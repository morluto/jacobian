# Persistent state format

The current and minimum supported state format is revision 11. Older stores are
rejected before any migration code runs.

To recover data from an older store, keep that directory unchanged and open it
with a compatible older checkout. Create a fresh state directory for the
current version; Jacobian provides no cross-revision import bridge. Do not edit
`metadata.sqlite3` to change its revision—the migration ledger and state-format
record are integrity boundaries.

Earlier migration definitions remain in source because migration ledgers bind
their checksums. They do not define supported runtime services or an in-place
upgrade path. Revisions 9 and 10 replace the broad checker-package digest with
a versioned per-checker manifest that separates checker and worker source,
records exact Python distributions, and produces one implementation digest;
existing checker authorization rows are deliberately not reinterpreted. New
stores apply the complete ordered schema and record revision 11.

Verification records are immutable artifacts rather than state-table rows.
Record schema v4 payloads snapshot the accepting checker's full manifest.
Revision 11 is an explicit cutover: revision-10 stores and their v3 records
remain readable with a matching older checkout and are not reinterpreted by the
current runtime. There is no legacy checker-authorization or record import path.
