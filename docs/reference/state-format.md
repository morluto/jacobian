# Persistent state format

The current and minimum supported state format is revision 10. Older stores are
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
stores apply the complete ordered schema and record revision 10.
