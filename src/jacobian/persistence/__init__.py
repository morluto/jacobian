"""Internal ownership boundary for Jacobian's local persistent state."""

from jacobian.persistence.database import StateDatabase, StateDatabaseError
from jacobian.persistence.decoding import (
    PersistenceCorruptionCode,
    PersistenceCorruptionError,
    decode_persisted_model,
)
from jacobian.persistence.locking import PersistenceLock
from jacobian.persistence.state_health import (
    MigrationMismatch,
    StateHealth,
    inspect_state_health,
)

__all__ = [
    "MigrationMismatch",
    "PersistenceCorruptionCode",
    "PersistenceCorruptionError",
    "PersistenceLock",
    "StateDatabase",
    "StateDatabaseError",
    "StateHealth",
    "decode_persisted_model",
    "inspect_state_health",
]
