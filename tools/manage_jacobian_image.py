#!/usr/bin/env python3
"""Build and resolve Jacobian evaluation container identities."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from tools.command_runner import (
    ToolCommandResult,
    ToolCommandStatus,
    operator_environment,
    run_operator_command,
)

ROOT = Path(__file__).resolve().parents[1]
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


class ImageError(RuntimeError):
    """Raised when an image cannot satisfy the evaluation identity contract."""


def _run(
    command: str,
    *arguments: str,
    timeout_seconds: float = 60.0,
    output_limit_bytes: int = 4 * 1024 * 1024,
) -> ToolCommandResult:
    environment = None
    if command == "docker":
        environment = operator_environment(
            include=("DOCKER_CONFIG", "DOCKER_CONTEXT", "DOCKER_HOST")
        )
    result = run_operator_command(
        command,
        arguments,
        cwd=ROOT,
        timeout_seconds=timeout_seconds,
        stdout_limit_bytes=output_limit_bytes,
        stderr_limit_bytes=output_limit_bytes,
        environment=environment,
    )
    if result.status is not ToolCommandStatus.EXITED or result.exit_code != 0:
        detail = result.diagnostic or result.stderr.decode("utf-8", "replace").strip()
        raise ImageError(f"{command} failed: {detail or result.status}")
    return result


def _text(value: bytes) -> str:
    return value.decode("utf-8", "strict")


def _git_sha() -> str:
    value = _text(_run("git", "rev-parse", "HEAD").stdout).strip()
    if not SHA_RE.fullmatch(value):
        raise ImageError("HEAD did not resolve to a full commit SHA")
    return value


def _source_dirty() -> bool:
    return bool(_text(_run("git", "status", "--porcelain").stdout).strip())


def _package_version() -> str:
    value = _text(_run("uv", "version", "--short").stdout).strip()
    if not value:
        raise ImageError("uv did not report the Jacobian package version")
    return value


def _docker_inspect(image: str) -> dict[str, Any]:
    result = _run("docker", "image", "inspect", image)
    try:
        payload = json.loads(_text(result.stdout))
    except json.JSONDecodeError as exc:
        raise ImageError("docker image inspect returned invalid JSON") from exc
    if (
        not isinstance(payload, list)
        or len(payload) != 1
        or not isinstance(payload[0], dict)
    ):
        raise ImageError(f"expected exactly one Docker image for {image}")
    return payload[0]


def _repo_digest(image: str, inspected: dict[str, Any]) -> str | None:
    repository = image.split("@", 1)[0].rsplit(":", 1)[0]
    values = inspected.get("RepoDigests", [])
    if not isinstance(values, list):
        return None
    matches = [
        value
        for value in values
        if isinstance(value, str) and value.startswith(repository + "@")
    ]
    if len(matches) == 1 and DIGEST_RE.fullmatch(matches[0].rsplit("@", 1)[1]):
        return matches[0]
    if "@" in image and DIGEST_RE.fullmatch(image.rsplit("@", 1)[1]):
        return image
    return None


def image_identity(image: str) -> dict[str, Any]:
    inspected = _docker_inspect(image)
    config = inspected.get("Config")
    labels = config.get("Labels") if isinstance(config, dict) else None
    if not isinstance(labels, dict):
        labels = {}
    source_sha = labels.get("org.opencontainers.image.revision")
    package_version = labels.get("org.opencontainers.image.version")
    source_dirty = labels.get("io.jacobian.source-dirty") == "true"
    os_name = inspected.get("Os")
    architecture = inspected.get("Architecture")
    image_id = inspected.get("Id")
    if not isinstance(source_sha, str) or not SHA_RE.fullmatch(source_sha):
        raise ImageError("image has no valid org.opencontainers.image.revision label")
    if not isinstance(package_version, str) or not package_version:
        raise ImageError("image has no org.opencontainers.image.version label")
    if not isinstance(os_name, str) or not isinstance(architecture, str):
        raise ImageError("image has no Docker OS/architecture identity")
    if not isinstance(image_id, str) or not DIGEST_RE.fullmatch(image_id):
        raise ImageError("image has no content-addressed Docker image ID")
    return {
        "source_sha": source_sha,
        "source_dirty": source_dirty,
        "reference": image,
        "digest_reference": _repo_digest(image, inspected),
        "image_id": image_id,
        "platform": f"{os_name}/{architecture}",
        "jacobian_package_version": package_version,
    }


def build(image: str) -> str:
    source_sha = _git_sha()
    dirty = _source_dirty()
    version = _package_version()
    command = [
        "docker",
        "build",
        "--build-arg",
        f"JACOBIAN_REVISION={source_sha}",
        "--build-arg",
        f"JACOBIAN_VERSION={version}",
        "--build-arg",
        f"JACOBIAN_SOURCE_DIRTY={str(dirty).lower()}",
        "--tag",
        image,
        ".",
    ]
    result = _run(
        command[0],
        *command[1:],
        timeout_seconds=30 * 60,
        output_limit_bytes=64 * 1024 * 1024,
    )
    print(_text(result.stdout), end="")
    print(_text(result.stderr), file=sys.stderr, end="")
    return image


def pull(registry_image: str, revision: str | None = None) -> str:
    if _source_dirty():
        raise ImageError("published evaluation images require a clean worktree")
    source_sha = revision or _git_sha()
    if not SHA_RE.fullmatch(source_sha):
        raise ImageError("revision must be a full 40-character commit SHA")
    tagged = f"{registry_image}:sha-{source_sha}"
    result = _run(
        "docker",
        "pull",
        tagged,
        timeout_seconds=15 * 60,
        output_limit_bytes=64 * 1024 * 1024,
    )
    print(_text(result.stdout), file=sys.stderr, end="")
    print(_text(result.stderr), file=sys.stderr, end="")
    identity = image_identity(tagged)
    if identity["source_sha"] != source_sha:
        raise ImageError(
            "published image revision label does not match the requested SHA"
        )
    digest_reference = identity["digest_reference"]
    if not isinstance(digest_reference, str):
        raise ImageError("Docker did not resolve the published tag to an OCI digest")
    return digest_reference


def select(registry_image: str) -> str:
    if _source_dirty():
        return build("jacobian:local")
    return pull(registry_image)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--image", default="jacobian:local")
    pull_parser = subparsers.add_parser("pull")
    pull_parser.add_argument("--registry-image", required=True)
    pull_parser.add_argument("--revision")
    select_parser = subparsers.add_parser("select")
    select_parser.add_argument("--registry-image", required=True)
    args = parser.parse_args()
    try:
        if args.command == "build":
            print(build(args.image))
        elif args.command == "pull":
            print(pull(args.registry_image, args.revision))
        elif args.command == "select":
            print(select(args.registry_image))
    except ImageError as exc:
        print(f"image error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
