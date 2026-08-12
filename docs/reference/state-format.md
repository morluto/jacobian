# Persistent state format

The current and minimum supported state format is revision 8. Older stores are
rejected before any migration code runs.

To recover data from an older store, keep that directory unchanged and open it
with a compatible older checkout. Create a fresh state directory for the
current version; Jacobian provides no cross-revision import bridge. Do not edit
`metadata.sqlite3` to change its revision—the migration ledger and state-format
record are integrity boundaries.

Earlier migration definitions remain in source because existing revision-8
ledgers bind their checksums. They do not define supported runtime services or
an in-place upgrade path. New stores apply the complete ordered schema and
record revision 8.
