#!/usr/bin/env python3
"""Compare Jacobian's filesystem CAS with a disposable SQLite BLOB spike.

This is decision evidence for issue #1224, not a production storage backend.
It deliberately benchmarks the blob carrier below artifact semantics so the
experiment does not introduce a second repository abstraction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sqlite3
import statistics
import sys
import tempfile
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Lock, local
from typing import Protocol

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.tooling.command_runner import (  # noqa: E402
    ToolCommandRequest,
    ToolCommandStatus,
    run_tool_command,
)

from jacobian.canonical import sha256_digest  # noqa: E402
from jacobian.storage.repository import ArtifactRepository  # noqa: E402

DEFAULT_SIZES = (1_024, 100 * 1_024, 1_024 * 1_024, 10 * 1_024 * 1_024)
DEFAULT_CONCURRENCY = (1, 4, 16)


class _BlobCarrier(Protocol):
    def put(self, data: bytes) -> str: ...

    def get(self, digest: str, *, maximum_bytes: int | None = None) -> bytes: ...

    def close(self) -> None: ...


class _CurrentCas:
    def __init__(self, root: Path) -> None:
        self._repository = ArtifactRepository(root)

    def put(self, data: bytes) -> str:
        return self._repository._blobs.write(data)

    def get(self, digest: str, *, maximum_bytes: int | None = None) -> bytes:
        data = self._repository._blobs.read(digest)
        if maximum_bytes is not None and len(data) > maximum_bytes:
            raise ValueError("blob exceeds the requested read bound")
        return data

    def close(self) -> None:
        self._repository.close()


class _SQLiteBlobSpike:
    """Small experiment owner; intentionally not imported by production code."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._connections = local()
        self._owned: list[sqlite3.Connection] = []
        self._owned_lock = Lock()
        connection = self._connect()
        connection.executescript(
            """
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=FULL;
            CREATE TABLE IF NOT EXISTS blobs (
                digest TEXT PRIMARY KEY,
                size_bytes INTEGER NOT NULL CHECK (size_bytes >= 0),
                data BLOB NOT NULL
            ) WITHOUT ROWID;
            """
        )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.path,
            timeout=30,
            isolation_level=None,
            check_same_thread=False,
        )
        connection.execute("PRAGMA busy_timeout=30000")
        with self._owned_lock:
            self._owned.append(connection)
        return connection

    def _connection(self) -> sqlite3.Connection:
        connection = getattr(self._connections, "connection", None)
        if connection is None:
            connection = self._connect()
            self._connections.connection = connection
        return connection

    def put(self, data: bytes) -> str:
        digest = sha256_digest(data)
        connection = self._connection()
        connection.execute("BEGIN IMMEDIATE")
        try:
            connection.execute(
                "INSERT OR IGNORE INTO blobs (digest, size_bytes, data) VALUES (?, ?, ?)",
                (digest, len(data), data),
            )
            row = connection.execute(
                "SELECT size_bytes, data FROM blobs WHERE digest = ?", (digest,)
            ).fetchone()
            if row is None or int(row[0]) != len(data) or bytes(row[1]) != data:
                raise RuntimeError("SQLite BLOB digest collision or corruption")
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        return digest

    def get(self, digest: str, *, maximum_bytes: int | None = None) -> bytes:
        row = (
            self._connection()
            .execute("SELECT size_bytes, data FROM blobs WHERE digest = ?", (digest,))
            .fetchone()
        )
        if row is None:
            raise KeyError(digest)
        size = int(row[0])
        if maximum_bytes is not None and size > maximum_bytes:
            raise ValueError("blob exceeds the requested read bound")
        data = bytes(row[1])
        if len(data) != size or sha256_digest(data) != digest:
            raise RuntimeError("SQLite BLOB failed digest verification")
        return data

    def backup(self, destination: Path) -> None:
        with sqlite3.connect(destination) as target:
            self._connection().backup(target)

    def close(self) -> None:
        with self._owned_lock:
            connections = tuple(self._owned)
            self._owned.clear()
        for connection in connections:
            connection.close()


@dataclass(frozen=True, slots=True)
class _Sample:
    backend: str
    operation: str
    size_bytes: int
    concurrency: int
    samples: int
    p50_ms: float
    p95_ms: float
    throughput_mib_s: float


def _payload(size: int, ordinal: int) -> bytes:
    return hashlib.shake_256(f"{size}:{ordinal}".encode()).digest(size)


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))]


def _measure(
    backend_name: str,
    carrier: _BlobCarrier,
    *,
    size: int,
    concurrency: int,
    iterations: int,
) -> tuple[_Sample, _Sample, _Sample]:
    payloads = tuple(
        _payload(size, concurrency * 1_000_000 + index) for index in range(iterations)
    )

    def timed(function: Callable[[], object]) -> float:
        started = time.perf_counter_ns()
        function()
        return (time.perf_counter_ns() - started) / 1_000_000

    def parallel[ItemT](
        function: Callable[[ItemT], float], items: tuple[ItemT, ...]
    ) -> tuple[list[float], float]:
        started = time.perf_counter_ns()
        with ThreadPoolExecutor(max_workers=concurrency) as workers:
            values = list(workers.map(function, items))
        return values, (time.perf_counter_ns() - started) / 1_000_000_000

    writes, write_seconds = parallel(
        lambda item: timed(lambda: carrier.put(item)), payloads
    )
    digests = tuple(sha256_digest(payload) for payload in payloads)
    reads, read_seconds = parallel(
        lambda digest: timed(lambda: carrier.get(digest)), digests
    )
    deduplicated, deduplicated_seconds = parallel(
        lambda _index: timed(lambda: carrier.put(payloads[0])),
        tuple(range(iterations)),
    )

    def sample(operation: str, values: list[float], elapsed_seconds: float) -> _Sample:
        return _Sample(
            backend=backend_name,
            operation=operation,
            size_bytes=size,
            concurrency=concurrency,
            samples=len(values),
            p50_ms=round(statistics.median(values), 3),
            p95_ms=round(_percentile(values, 0.95), 3),
            throughput_mib_s=round(
                (size * len(values) / (1024 * 1024)) / elapsed_seconds, 3
            ),
        )

    return (
        sample("unique_write", writes, write_seconds),
        sample("verified_read", reads, read_seconds),
        sample("deduplicated_write", deduplicated, deduplicated_seconds),
    )


def _sqlite_correctness(root: Path) -> dict[str, bool]:
    path = root / "correctness.sqlite3"
    store = _SQLiteBlobSpike(path)
    payload = _payload(4096, 0)
    digest = store.put(payload)
    try:
        bounded_read = False
        try:
            store.get(digest, maximum_bytes=len(payload) - 1)
        except ValueError:
            bounded_read = True
        backup = root / "backup.sqlite3"
        store.backup(backup)
    finally:
        store.close()
    restarted = _SQLiteBlobSpike(path)
    backup_store = _SQLiteBlobSpike(backup)
    try:
        return {
            "bounded_read": bounded_read,
            "restart": restarted.get(digest) == payload,
            "backup_restore": backup_store.get(digest) == payload,
            "rollback": _rollback_probe(root / "rollback.sqlite3"),
        }
    finally:
        restarted.close()
        backup_store.close()


def _rollback_probe(path: Path) -> bool:
    completed = run_tool_command(
        ToolCommandRequest(
            executable=sys.executable,
            arguments=(__file__, "--rollback-child", str(path)),
            cwd=str(Path(__file__).resolve().parents[1]),
            timeout_seconds=10,
            stdout_limit_bytes=1024,
            stderr_limit_bytes=1024,
        )
    )
    if completed.status is not ToolCommandStatus.EXITED or completed.exit_code != 17:
        return False
    with sqlite3.connect(path) as connection:
        row = connection.execute(
            "SELECT COUNT(*) FROM blobs WHERE digest = 'uncommitted'"
        ).fetchone()
    return row is not None and int(row[0]) == 0


def _rollback_child(path: Path) -> None:
    store = _SQLiteBlobSpike(path)
    connection = store._connection()
    connection.execute("BEGIN IMMEDIATE")
    connection.execute(
        "INSERT INTO blobs (digest, size_bytes, data) VALUES ('uncommitted', 1, X'00')"
    )
    os._exit(17)


def _parse_csv(value: str) -> tuple[int, ...]:
    values = tuple(int(item) for item in value.split(","))
    if not values or any(item < 1 for item in values):
        raise argparse.ArgumentTypeError("values must be positive integers")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=_parse_csv, default=DEFAULT_SIZES)
    parser.add_argument("--concurrency", type=_parse_csv, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--rollback-child", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.rollback_child is not None:
        _rollback_child(args.rollback_child)
    if args.iterations < max(args.concurrency):
        parser.error("--iterations must be at least the maximum concurrency")

    with tempfile.TemporaryDirectory(prefix="jacobian-storage-spike-") as temporary:
        root = Path(temporary)
        rows: list[_Sample] = []
        factories: tuple[tuple[str, Callable[[Path], _BlobCarrier]], ...] = (
            ("filesystem_cas", _CurrentCas),
            ("sqlite_blob", _SQLiteBlobSpike),
        )
        for backend_name, factory in factories:
            carrier = factory(root / backend_name)
            try:
                for size in args.sizes:
                    for concurrency in args.concurrency:
                        rows.extend(
                            _measure(
                                backend_name,
                                carrier,
                                size=size,
                                concurrency=concurrency,
                                iterations=args.iterations,
                            )
                        )
            finally:
                carrier.close()
        report = {
            "schema_version": "1",
            "warning": "experiment only; not a production storage backend",
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "sqlite": sqlite3.sqlite_version,
                "sqlite_synchronous": "FULL",
                "sqlite_journal_mode": "WAL",
            },
            "sizes": args.sizes,
            "concurrency": args.concurrency,
            "iterations": args.iterations,
            "sqlite_correctness": _sqlite_correctness(root),
            "samples": [asdict(row) for row in rows],
        }
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(encoded, encoding="utf-8")
    sys.stdout.write(encoded)


if __name__ == "__main__":
    main()
