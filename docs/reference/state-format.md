# Persistent state format

The current and minimum supported state format is revision 8. Older stores are
rejected before any migration code runs. This clean pre-stable boundary removes
the research-memory schema without retaining a runtime compatibility path.

To move data from an older store, export the records through a compatible
older checkout, create a fresh state directory with the current Jacobian
version, and import the exported records through the public persistence
workflow. Do not edit `metadata.sqlite3` to change its revision; the migration
ledger and state-format record are integrity boundaries.

Earlier migration definitions remain in the source because the
SQLite ledger is immutable historical evidence. They are not an indication
that the retired workspace schema or data-upgrade bridge is still supported.
New stores apply the complete ordered schema and record revision 8.

Revision 7 added the now-retired production reasoning-log tables
`reasoning_runs` and `reasoning_events`. The immutable migration remains so
existing revision-8 ledgers keep their historical checksums; current runtimes
do not construct or expose a reasoning-log service.

Revision 8 establishes the memoryless state boundary. Stores from earlier
pre-stable releases are not upgraded in place.
