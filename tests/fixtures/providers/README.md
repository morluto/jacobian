# Pinned provider integration fixtures

These fixtures exercise optional mathematical backends in isolated, pinned
environments. They are integration evidence for Jacobian development, not
Harbor tasks and not agent-evaluation data.

Each provider directory keeps only the executable spike, frozen input and pin,
and its reproducible container definition. The corresponding boundary tests
exercise the spike's parser, pin binding, and typed unavailable outcomes:

```sh
uv run pytest tests/boundary/process/providers
```

To build a single pinned environment for manual investigation, use its Compose
fixture with the repository as the build context:

```sh
docker compose -f tests/fixtures/providers/cgal/docker-compose.yaml build
```
