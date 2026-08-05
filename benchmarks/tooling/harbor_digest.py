"""Pinned Harbor task digest and provenance calculation."""

from __future__ import annotations

from pathlib import Path


class HarborDigestError(ValueError):
    """The pinned Harbor runtime cannot calculate a task digest."""


def task_digest(task_dir: Path) -> str:
    """Return Harbor's durable content hash for one task directory.

    The import is intentionally lazy: registry and topology checks remain
    usable without Harbor, while every digest caller uses Harbor's packager and
    therefore the same content-hash semantics as ``TrialLock.task.digest``.
    """

    try:
        from harbor.publisher.packager import Packager
    except (ImportError, ModuleNotFoundError) as exc:
        raise HarborDigestError(
            "Harbor is required to compute task digests; use the pinned Harbor runner"
        ) from exc
    digest, _ = Packager.compute_content_hash(task_dir)
    return digest


__all__ = ["HarborDigestError", "task_digest"]
