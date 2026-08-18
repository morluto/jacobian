"""Harbor Docker/Compose/.dockerignore build-context security policy.

This module owns Dockerfile FROM parsing, apt policy, hidden-material COPY
rules, verifier image identity, Compose build contexts, .dockerignore
deny-all policy, and missing build-context source checks.
"""

__all__: list[str] = []
