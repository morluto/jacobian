"""Opaque runtime-local carriers for typed mathematical values."""

from __future__ import annotations

import hashlib
import secrets
import threading
from collections import OrderedDict
from dataclasses import dataclass

from jacobian.canonical import canonicalize_json
from jacobian.contracts.common import Sha256Digest, ValueUri
from jacobian.contracts.results import ContractModel

_MAX_REFERENCES = 256
_MAX_TOTAL_BYTES = 16 * 1024 * 1024


class ValueReferenceError(ValueError):
    """A value reference is unknown, incompatible, or cannot be retained."""


@dataclass(frozen=True, slots=True)
class StoredValue:
    """Server-owned value identity and source provenance."""

    value: ContractModel
    value_type: type[ContractModel]
    digest: Sha256Digest
    canonical_bytes: int
    source_operation_id: str
    source_operation_version: str
    source_port: str


class ValueReferenceStore:
    """Bounded in-memory value carriers scoped to one runtime owner."""

    def __init__(self) -> None:
        self._values: OrderedDict[ValueUri, StoredValue] = OrderedDict()
        self._total_bytes = 0
        self._lock = threading.RLock()

    def put(
        self,
        value: ContractModel,
        *,
        operation_id: str,
        operation_version: str,
        output_port: str,
    ) -> ValueUri:
        encoded = canonicalize_json(value.model_dump(mode="json"))
        with self._lock:
            if len(encoded) > _MAX_TOTAL_BYTES:
                raise ValueReferenceError("runtime value-reference byte limit exceeded")
            while self._values and (
                len(self._values) >= _MAX_REFERENCES
                or self._total_bytes + len(encoded) > _MAX_TOTAL_BYTES
            ):
                _, evicted = self._values.popitem(last=False)
                self._total_bytes -= evicted.canonical_bytes
            token = self._new_token()
            self._values[token] = StoredValue(
                value=value.model_copy(deep=True),
                value_type=type(value),
                digest=f"sha256:{hashlib.sha256(encoded).hexdigest()}",
                canonical_bytes=len(encoded),
                source_operation_id=operation_id,
                source_operation_version=operation_version,
                source_port=output_port,
            )
            self._total_bytes += len(encoded)
            return token

    def resolve(
        self,
        value_ref: ValueUri,
        expected_type: type[ContractModel],
    ) -> ContractModel:
        with self._lock:
            try:
                stored = self._values[value_ref]
            except KeyError:
                raise ValueReferenceError(
                    "value reference is unknown or belongs to another runtime"
                ) from None
            if stored.value_type is not expected_type:
                raise ValueReferenceError(
                    f"value reference carries {stored.value_type.__name__}; "
                    f"expected {expected_type.__name__}"
                )
            self._values.move_to_end(value_ref)
            return stored.value.model_copy(deep=True)

    def inspect(self, value_ref: ValueUri) -> StoredValue:
        """Return immutable stored facts for runtime tests and diagnostics."""

        with self._lock:
            try:
                return self._values[value_ref]
            except KeyError:
                raise ValueReferenceError("unknown value reference") from None

    def close(self) -> None:
        with self._lock:
            self._values.clear()
            self._total_bytes = 0

    def _new_token(self) -> ValueUri:
        while True:
            token = f"value://{secrets.token_urlsafe(24)}"
            if token not in self._values:
                return token


__all__ = ["StoredValue", "ValueReferenceError", "ValueReferenceStore"]
