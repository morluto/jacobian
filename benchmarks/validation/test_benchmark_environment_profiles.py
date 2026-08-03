from __future__ import annotations

import re

from benchmarks.tooling.harbor_suite import (
    NETWORK_MODES,
    load_environment_profiles,
    load_registry,
)


def test_every_task_resolves_a_digest_pinned_environment_profile() -> None:
    profiles = load_environment_profiles()
    for suite in load_registry():
        for task in suite.tasks:
            profile = profiles[task.environment_profile]
            assert "@sha256:" in profile.agent_image
            assert "@sha256:" in profile.verifier_image
            assert re.fullmatch(r"sha256:[0-9a-f]{64}", profile.verifier_runtime_digest)
            dockerfile = (task.path / "environment" / "Dockerfile").read_text()
            assert dockerfile.splitlines()[0] == f"FROM {profile.agent_image}"
            verifier_dockerfile = (task.path / "tests" / "Dockerfile").read_text()
            assert (
                verifier_dockerfile.splitlines()[0] == f"FROM {profile.verifier_image}"
            )
            if not profile.allow_apt:
                assert "apt-get" not in dockerfile


def test_network_policy_is_independent_of_image_profile() -> None:
    # Network policy lives in each task's task.toml, never in the shared
    # environment-profile definitions, so a profile may coexist with any
    # documented network mode (public, no-network, or allowlist).  Asserting
    # the present all-offline inventory as an exact-equality restriction would
    # forbid a future task from using the documented public/allowlist policy
    # with the standard core-python profile.
    profiles = load_environment_profiles()
    for profile in profiles.values():
        assert not hasattr(profile, "network_mode")
        assert not hasattr(profile, "network_policy")

    for suite in load_registry():
        for task in suite.tasks:
            task_toml = (task.path / "task.toml").read_text()
            modes = set(re.findall(r'network_mode = "([^"]+)"', task_toml))
            assert modes  # every task pins both agent and verifier network modes
            assert modes <= NETWORK_MODES


def test_dataset_roots_never_commit_mutable_publication_manifests() -> None:
    for suite in load_registry():
        assert not (suite.path / "dataset.toml").exists()
