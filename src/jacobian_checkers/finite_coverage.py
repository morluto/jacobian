"""Independent replay for bounded exactly-once finite archive coverage."""

from __future__ import annotations

import hashlib
from typing import Any

from jacobian.canonical import canonicalize_json

_SPECS: dict[str, dict[str, str]] = {
    "finite.integer.decimal@1": {
        "item_type": "INTEGER",
        "algorithm": "DECIMAL_INTEGER",
        "key_format": "SHA256_RFC8785_TAGGED_VALUE",
    },
    "finite.string.nfc@1": {
        "item_type": "STRING",
        "algorithm": "NFC_STRING",
        "key_format": "SHA256_RFC8785_TAGGED_VALUE",
    },
}


def _reject(detail: str) -> dict[str, Any]:
    return {
        "accepted": False,
        "conclusion": "UNKNOWN",
        "arithmetic": "EXACT_INTEGER",
        "method": "EXHAUSTIVE_FINITE",
        "coverage": "EXHAUSTIVE",
        "detail": detail,
    }


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonicalize_json(value)).hexdigest()


def _canonical_key(canonicalizer_id: str, item: Any) -> str:
    return _digest({"canonicalizer_id": canonicalizer_id, "value": item})


def _require(condition: bool, detail: str) -> dict[str, Any] | None:
    return None if condition else _reject(detail)


def _load_core_artifacts(
    request: dict[str, Any],
) -> Any:
    if request.get("request_version") != "1":
        return _reject("unsupported checker request version")
    supporting = request.get("supporting_artifacts")
    if not isinstance(supporting, list):
        return _reject("canonicalizer and page artifacts are required")
    claim_artifact = request["claim"]
    archive_artifact = request["candidate"]
    scope_artifact = request["scope"]
    certificate_artifact = request["certificate"]
    claim = claim_artifact["payload"]
    archive = archive_artifact["payload"]
    scope = scope_artifact["payload"]
    certificate = certificate_artifact["payload"]
    if claim.get("predicate") != "FINITE_EXACTLY_ONCE_COVERAGE":
        return _reject("unsupported finite coverage claim")
    if certificate.get("certificate_type") != "finite.coverage":
        return _reject("unexpected certificate type")
    if certificate.get("format_version") != "1":
        return _reject("unsupported certificate format")
    if certificate.get("bindings") != request["expected_bindings"]:
        return _reject("certificate bindings do not match the request")
    return (
        claim_artifact,
        archive_artifact,
        scope_artifact,
        certificate_artifact,
        supporting,
        claim,
        archive,
        scope,
        certificate,
    )


def _resolve_registration(
    *,
    scope: dict[str, Any],
    supporting: list[Any],
) -> Any:
    canonicalizer_id = scope.get("canonicalizer_id")
    spec = _SPECS.get(canonicalizer_id) if isinstance(canonicalizer_id, str) else None
    if spec is None:
        return _reject("unregistered finite canonicalizer")
    registration_artifacts = [
        artifact
        for artifact in supporting
        if isinstance(artifact.get("payload"), dict)
        and artifact["payload"].get("registration_version") == "1"
    ]
    if len(registration_artifacts) != 1:
        return _reject("exactly one canonicalizer registration is required")
    registration_artifact = registration_artifacts[0]
    registration = registration_artifact["payload"]
    expected_specification_digest = _digest(
        {"canonicalizer_id": canonicalizer_id, **spec}
    )
    if (
        registration.get("canonicalizer_id") != canonicalizer_id
        or registration.get("item_type") != spec["item_type"]
        or registration.get("algorithm") != spec["algorithm"]
        or registration.get("key_format") != spec["key_format"]
        or registration.get("specification_digest") != expected_specification_digest
        or registration_artifact.get("artifact_uri") != scope.get("canonicalizer_uri")
        or registration_artifact.get("object_digest")
        != scope.get("canonicalizer_object_digest")
    ):
        return _reject("canonicalizer registration or digest is not bound")
    return (
        canonicalizer_id,
        spec,
        registration_artifact,
        expected_specification_digest,
    )


def _validate_scope(
    *,
    scope: dict[str, Any],
    canonicalizer_id: str,
    expected_type: type,
    expected_specification_digest: str,
) -> Any:
    scope_items = scope.get("items")
    scope_keys = scope.get("canonical_keys")
    if (
        not isinstance(scope_items, list)
        or not 1 <= len(scope_items) <= 4096
        or not all(type(item) is expected_type for item in scope_items)
        or not isinstance(scope_keys, list)
        or len(scope_items) != len(scope_keys)
    ):
        return _reject("finite scope is malformed or uses the wrong item type")
    recomputed_scope_keys = [
        _canonical_key(canonicalizer_id, item) for item in scope_items
    ]
    if (
        recomputed_scope_keys != scope_keys
        or len(set(scope_keys)) != len(scope_keys)
        or scope.get("scope_keys_digest") != _digest(scope_keys)
        or scope.get("canonicalizer_specification_digest")
        != expected_specification_digest
    ):
        return _reject("scope canonical keys or digest are invalid")
    return scope_items, scope_keys


def _validate_archive_pages(
    *,
    archive: dict[str, Any],
    scope_artifact: dict[str, Any],
    registration_artifact: dict[str, Any],
    supporting: list[Any],
    canonicalizer_id: str,
    expected_type: type,
    expected_specification_digest: str,
) -> Any:
    if (
        archive.get("archive_version") != "1"
        or archive.get("scope_uri") != scope_artifact.get("artifact_uri")
        or archive.get("scope_object_digest") != scope_artifact.get("object_digest")
        or archive.get("canonicalizer_uri") != registration_artifact.get("artifact_uri")
        or archive.get("canonicalizer_object_digest")
        != registration_artifact.get("object_digest")
        or archive.get("canonicalizer_specification_digest")
        != expected_specification_digest
        or archive.get("canonicalizer_id") != canonicalizer_id
    ):
        return _reject("archive scope or canonicalizer binding is invalid")
    page_bindings = archive.get("page_bindings")
    if not isinstance(page_bindings, list) or not 1 <= len(page_bindings) <= 64:
        return _reject("archive page bindings are malformed")
    page_artifacts = {
        artifact.get("artifact_uri"): artifact
        for artifact in supporting
        if isinstance(artifact.get("payload"), dict)
        and artifact["payload"].get("page_version") == "1"
    }
    if len(page_artifacts) != len(page_bindings):
        return _reject("archive page artifacts do not match the manifest")
    archive_keys: list[str] = []
    total_items = 0
    expected_page_uris: list[str] = []
    for expected_index, binding in enumerate(page_bindings):
        page_error = _validate_one_page(
            binding=binding,
            expected_index=expected_index,
            page_artifacts=page_artifacts,
            registration_artifact=registration_artifact,
            canonicalizer_id=canonicalizer_id,
            expected_type=expected_type,
        )
        if isinstance(page_error, dict):
            return page_error
        keys, item_count, page_uri = page_error
        archive_keys.extend(keys)
        total_items += item_count
        expected_page_uris.append(page_uri)
    if total_items > 4096 or archive.get("total_item_count") != total_items:
        return _reject("archive total item count is invalid")
    digest_payload = {
        "scope_uri": archive["scope_uri"],
        "scope_object_digest": archive["scope_object_digest"],
        "canonicalizer_uri": archive["canonicalizer_uri"],
        "canonicalizer_object_digest": archive["canonicalizer_object_digest"],
        "canonicalizer_specification_digest": archive[
            "canonicalizer_specification_digest"
        ],
        "canonicalizer_id": archive["canonicalizer_id"],
        "page_bindings": page_bindings,
        "total_item_count": total_items,
    }
    if archive.get("archive_digest") != _digest(digest_payload):
        return _reject("archive digest is invalid")
    return archive_keys, expected_page_uris, page_bindings, total_items


def _validate_one_page(
    *,
    binding: Any,
    expected_index: int,
    page_artifacts: dict[Any, Any],
    registration_artifact: dict[str, Any],
    canonicalizer_id: str,
    expected_type: type,
) -> Any:
    if not isinstance(binding, dict) or binding.get("page_index") != expected_index:
        return _reject("archive page indices are not contiguous")
    page_uri = binding.get("page_uri")
    if not isinstance(page_uri, str):
        return _reject("a bound archive page URI is malformed")
    page_artifact = page_artifacts.get(page_uri)
    if page_artifact is None:
        return _reject("a bound archive page is unavailable")
    page = page_artifact["payload"]
    items = page.get("items")
    keys = page.get("canonical_keys")
    if (
        page_artifact.get("object_digest") != binding.get("page_object_digest")
        or page.get("page_index") != expected_index
        or page.get("canonicalizer_uri") != registration_artifact.get("artifact_uri")
        or page.get("canonicalizer_object_digest")
        != registration_artifact.get("object_digest")
        or page.get("canonicalizer_id") != canonicalizer_id
        or not isinstance(items, list)
        or len(items) > 1024
        or not all(type(item) is expected_type for item in items)
        or not isinstance(keys, list)
        or len(items) != len(keys)
    ):
        return _reject("a bound archive page is malformed")
    recomputed_keys = [_canonical_key(canonicalizer_id, item) for item in items]
    items_digest = _digest({"items": items, "canonical_keys": recomputed_keys})
    payload_digest = _digest(page)
    if (
        keys != recomputed_keys
        or page.get("items_digest") != items_digest
        or binding.get("items_digest") != items_digest
        or binding.get("page_payload_digest") != payload_digest
        or binding.get("item_count") != len(items)
    ):
        return _reject("page keys, counts, or digest bindings are invalid")
    return recomputed_keys, len(items), page_uri


def _validate_certificate_and_claim(
    *,
    certificate: dict[str, Any],
    claim: dict[str, Any],
    claim_artifact: dict[str, Any],
    archive_artifact: dict[str, Any],
    scope_artifact: dict[str, Any],
    registration_artifact: dict[str, Any],
    archive: dict[str, Any],
    scope: dict[str, Any],
    page_bindings: list[Any],
    canonicalizer_id: str,
    expected_specification_digest: str,
) -> dict[str, Any] | None:
    certificate_payload = certificate.get("payload")
    claim_uri = claim_artifact.get("artifact_uri")
    if (
        not isinstance(certificate_payload, dict)
        or certificate_payload.get("relation_id")
        != "finite.relation.covers-exactly-once"
        or certificate_payload.get("obligation_uri") != claim_uri
        or certificate_payload.get("canonicalizer_uri")
        != registration_artifact.get("artifact_uri")
        or certificate_payload.get("canonicalizer_object_digest")
        != registration_artifact.get("object_digest")
        or certificate_payload.get("canonicalizer_specification_digest")
        != expected_specification_digest
        or certificate_payload.get("scope_keys_digest")
        != scope.get("scope_keys_digest")
        or certificate_payload.get("archive_digest") != archive.get("archive_digest")
        or certificate_payload.get("page_binding_digest") != _digest(page_bindings)
    ):
        return _reject("certificate coverage metadata is not bound")
    if (
        claim.get("scope_uri") != scope_artifact.get("artifact_uri")
        or claim.get("archive_uri") != archive_artifact.get("artifact_uri")
        or claim.get("canonicalizer_uri") != registration_artifact.get("artifact_uri")
        or claim.get("canonicalizer_id") != canonicalizer_id
        or claim.get("scope_keys_digest") != scope.get("scope_keys_digest")
        or claim.get("archive_digest") != archive.get("archive_digest")
    ):
        return _reject("claim does not bind the exact scope and archive")
    return None


def _exact_once_and_lineage(
    *,
    scope_keys: list[str],
    archive_keys: list[str],
    archive_artifact: dict[str, Any],
    scope_artifact: dict[str, Any],
    registration_artifact: dict[str, Any],
    expected_page_uris: list[str],
    page_bindings: list[Any],
    claim_artifact: dict[str, Any],
) -> dict[str, Any]:
    scope_set = set(scope_keys)
    counts: dict[str, int] = {}
    for key in archive_keys:
        counts[key] = counts.get(key, 0) + 1
    if scope_set - set(counts):
        return _reject("paged archive omits one or more finite scope items")
    if set(counts) - scope_set:
        return _reject("paged archive contains items outside the finite scope")
    if any(count != 1 for count in counts.values()):
        return _reject("paged archive repeats one or more finite scope items")
    candidate_parents = archive_artifact.get("parents")
    if not isinstance(candidate_parents, list) or not {
        scope_artifact.get("artifact_uri"),
        registration_artifact.get("artifact_uri"),
        *expected_page_uris,
    }.issubset(set(candidate_parents)):
        return _reject("archive lineage does not bind scope and pages")
    return {
        "accepted": True,
        "conclusion": "TRUE",
        "arithmetic": "EXACT_INTEGER",
        "method": "EXHAUSTIVE_FINITE",
        "coverage": "EXHAUSTIVE",
        "detail": (
            f"replayed {len(scope_keys)} scope keys across "
            f"{len(page_bindings)} bound pages exactly once"
        ),
        "relation_id": "finite.relation.covers-exactly-once",
        "relationship_source_artifact_uris": [scope_artifact["artifact_uri"]],
        "relationship_target_artifact_uris": [archive_artifact["artifact_uri"]],
        "obligation_uri": claim_artifact.get("artifact_uri"),
    }


def check_finite_coverage(request: dict[str, Any]) -> dict[str, Any]:
    """Recompute every canonical key without importing producer code."""

    try:
        loaded = _load_core_artifacts(request)
        if isinstance(loaded, dict):
            return loaded
        (
            claim_artifact,
            archive_artifact,
            scope_artifact,
            _certificate_artifact,
            supporting,
            claim,
            archive,
            scope,
            certificate,
        ) = loaded
        registered = _resolve_registration(scope=scope, supporting=supporting)
        if isinstance(registered, dict):
            return registered
        (
            canonicalizer_id,
            spec,
            registration_artifact,
            expected_specification_digest,
        ) = registered
        expected_type = str if spec["item_type"] == "STRING" else int
        scoped = _validate_scope(
            scope=scope,
            canonicalizer_id=canonicalizer_id,
            expected_type=expected_type,
            expected_specification_digest=expected_specification_digest,
        )
        if isinstance(scoped, dict):
            return scoped
        _scope_items, scope_keys = scoped
        archived = _validate_archive_pages(
            archive=archive,
            scope_artifact=scope_artifact,
            registration_artifact=registration_artifact,
            supporting=supporting,
            canonicalizer_id=canonicalizer_id,
            expected_type=expected_type,
            expected_specification_digest=expected_specification_digest,
        )
        if isinstance(archived, dict):
            return archived
        archive_keys, expected_page_uris, page_bindings, _total_items = archived
        bound = _validate_certificate_and_claim(
            certificate=certificate,
            claim=claim,
            claim_artifact=claim_artifact,
            archive_artifact=archive_artifact,
            scope_artifact=scope_artifact,
            registration_artifact=registration_artifact,
            archive=archive,
            scope=scope,
            page_bindings=page_bindings,
            canonicalizer_id=canonicalizer_id,
            expected_specification_digest=expected_specification_digest,
        )
        if bound is not None:
            return bound

        return _exact_once_and_lineage(
            scope_keys=scope_keys,
            archive_keys=archive_keys,
            archive_artifact=archive_artifact,
            scope_artifact=scope_artifact,
            registration_artifact=registration_artifact,
            expected_page_uris=expected_page_uris,
            page_bindings=page_bindings,
            claim_artifact=claim_artifact,
        )
    except (KeyError, TypeError, ValueError):
        return _reject("malformed finite coverage checker request")
