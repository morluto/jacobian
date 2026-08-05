"""Pinned Harbor task digest and provenance calculation."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any


class HarborDigestError(ValueError):
    """The pinned Harbor runtime cannot calculate a task digest."""


_COPY_RE = re.compile(
    r"(?i)^\s*COPY\s+(?P<src>[^\s]+(?:\s+[^\s]+)*)\s+(?P<dest>[^\s]+)",
)


def load_compose_doc(compose_path: Path) -> dict[str, Any] | None:
    """Parse a docker-compose.yaml file, returning ``None`` when absent or not a dict."""

    if not compose_path.is_file():
        return None
    try:
        import yaml
    except (ImportError, ModuleNotFoundError) as exc:
        raise HarborDigestError(
            "PyYAML is required to parse docker-compose.yaml build contexts"
        ) from exc
    try:
        doc = yaml.safe_load(compose_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise HarborDigestError(
            f"docker-compose.yaml is not valid YAML: {exc}"
        ) from exc
    return doc if isinstance(doc, dict) else None


def extract_build_context(doc: dict[str, Any]) -> str | None:
    """Extract the ``main`` service build context from a compose document."""

    services = doc.get("services")
    if not isinstance(services, dict):
        return None
    main = services.get("main")
    if not isinstance(main, dict):
        return None
    build = main.get("build")
    if isinstance(build, str):
        context = build
    elif isinstance(build, dict):
        context = build.get("context")
    else:
        return None
    return context if isinstance(context, str) and context else None


def _repository_root(task_dir: Path) -> Path:
    benchmarks_root = next(
        (
            parent
            for parent in task_dir.resolve().parents
            if parent.name == "benchmarks"
        ),
        None,
    )
    return benchmarks_root.parent if benchmarks_root is not None else task_dir


def _compose_build_context(task_dir: Path) -> Path | None:
    """Return the resolved build context for ``environment/docker-compose.yaml``.

    Returns ``None`` when the task has no compose file or the compose file
    does not override the ``main`` service build context.
    """

    compose_path = task_dir / "environment" / "docker-compose.yaml"
    doc = load_compose_doc(compose_path)
    if doc is None:
        return None
    context = extract_build_context(doc)
    if context is None:
        return None
    resolved = (compose_path.parent / context).resolve()
    try:
        resolved.relative_to(_repository_root(task_dir))
    except ValueError as exc:
        raise HarborDigestError(
            "docker-compose.yaml build context escapes the repository"
        ) from exc
    return resolved


def _classify_copy_source(
    source: str, context_root: Path, task_dir: Path
) -> Path | None:
    """Return an external file path if *source* is outside the task tree, else ``None``."""

    if source.startswith("--"):
        return None
    resolved = (context_root / source).resolve()
    try:
        resolved.relative_to(context_root)
    except ValueError as exc:
        raise HarborDigestError(
            "Dockerfile COPY source escapes the compose build context"
        ) from exc
    try:
        resolved.relative_to(task_dir)
        return None
    except ValueError:
        pass
    return resolved if resolved.is_file() else None


def _compose_context_external_files(
    task_dir: Path,
    context_root: Path,
) -> list[Path]:
    """Return files under the compose build context but outside the task tree.

    The Dockerfile's COPY directives reference paths relative to the build
    context.  Any COPY source that resolves outside ``task_dir`` is an
    external file that the task digest must bind but Harbor's native
    ``dirhash(task_dir)`` cannot see.
    """

    dockerfile_path = task_dir / "environment" / "Dockerfile"
    if not dockerfile_path.is_file():
        return []
    external: list[Path] = []
    dockerignore = context_root / ".dockerignore"
    if dockerignore.is_file():
        external.append(dockerignore)
    try:
        dockerfile_text = dockerfile_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise HarborDigestError(f"Dockerfile is not readable: {exc}") from exc
    for line in dockerfile_text.splitlines():
        match = _COPY_RE.match(line)
        if match is None:
            continue
        for source in match.group("src").split():
            external_file = _classify_copy_source(source, context_root, task_dir)
            if external_file is not None:
                external.append(external_file)
    return sorted(set(external))


def compose_context_supplement(task_dir: Path) -> str | None:
    """Return a content hash for external compose build-context files.

    When a task's ``environment/docker-compose.yaml`` widens the build context
    beyond the task directory, Harbor's native ``dirhash(task_dir)`` cannot
    see files outside the task tree.  This supplement hashes the
    ``.dockerignore`` at the context root plus every Dockerfile COPY source
    that resolves outside the task directory, so a changed central runner or
    ignore policy produces a different task digest and stale snapshot locks
    fail.

    Returns ``None`` when the task has no widened compose build context.
    """

    context_root = _compose_build_context(task_dir)
    if context_root is None:
        return None
    try:
        context_root.relative_to(task_dir)
        return None
    except ValueError:
        pass
    external_files = _compose_context_external_files(task_dir, context_root)
    if not external_files:
        return None
    hasher = hashlib.sha256()
    for path in external_files:
        try:
            relative = path.relative_to(context_root)
        except ValueError:
            relative = path
        hasher.update(str(relative).encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(path.read_bytes())
        hasher.update(b"\0")
    return DIGEST_PREFIX + hasher.hexdigest()


DIGEST_PREFIX = "sha256:"


def task_digest(task_dir: Path) -> str:
    """Return Harbor's native checksum augmented with compose-context files.

    The import is intentionally lazy: registry and topology checks remain
    usable without Harbor, while every digest caller gets the same Task model
    and therefore the same checksum semantics. When a task widens its compose
    build context beyond the task directory, the external files (central
    runner, ``.dockerignore``) are folded into the digest so stale snapshots
    fail.
    """

    try:
        from harbor.models.task.task import Task
    except (ImportError, ModuleNotFoundError) as exc:
        raise HarborDigestError(
            "Harbor is required to compute task digests; use the pinned Harbor runner"
        ) from exc
    native = str(Task(task_dir, disable_verification=True).checksum)
    supplement = compose_context_supplement(task_dir)
    if supplement is None:
        return native
    return (
        DIGEST_PREFIX
        + hashlib.sha256((native + "\n" + supplement).encode("utf-8")).hexdigest()
    )


def durable_task_digest(task_dir: Path) -> str:
    """Return Harbor's durable ``TrialLock.task.digest`` for a task directory."""

    try:
        from harbor.publisher.packager import Packager
    except (ImportError, ModuleNotFoundError) as exc:
        raise HarborDigestError(
            "Harbor is required to compute task digests; use the pinned Harbor runner"
        ) from exc
    digest, _ = Packager.compute_content_hash(task_dir)
    return digest


__all__ = [
    "HarborDigestError",
    "compose_context_supplement",
    "durable_task_digest",
    "extract_build_context",
    "load_compose_doc",
    "task_digest",
]
