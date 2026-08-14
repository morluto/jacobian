# Persistent state format

The current state format is revision 13. Revision 12 is the minimum supported
update source; older stores are rejected before any migration code runs.

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
existing checker authorization rows are deliberately not reinterpreted. The
operation-catalog boundary in revision 12 retires the generic experiment,
search, installed-plugin, and reasoning-log tables. Revision 13 makes SQLite
overlay-only: packaged built-in descriptors live in the wheel index, and
revision-12 built-in descriptor rows are overlay-stale until `jacobian update`.
New stores apply the complete ordered schema and record revision 13.

Verification records are immutable artifacts rather than state-table rows.
Record schema v4 payloads snapshot the accepting checker's full manifest.
Revision 12 is the minimum accepted migration source: revision-11 stores and
their earlier records remain readable with a matching older checkout, but are
not reinterpreted by the current runtime. Revision-12 stores must be updated to
revision 13 before serving. There is no legacy checker-authorization or record
import path. `jacobian init` and `jacobian update` authorize checkers and write
overlay bindings; they do not fail closed on packaged built-in title drift.
