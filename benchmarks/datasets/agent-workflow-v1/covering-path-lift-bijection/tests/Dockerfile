FROM ghcr.io/astral-sh/uv:0.8.4-python3.12-bookworm-slim@sha256:dc7e1d08f8ca979826ec0b68b31c783e0b35b568be6a078d4cbedf38c4cc085e
# Finite graph-cover path-lifting verifier.
LABEL jacobian.task="jacobian/covering-path-lift-bijection" \
      jacobian.checksum="03a3b144aa9dc6652fcd259d48af7ae7733ad0edb5460e58aa0bf6484a38bd5a"
COPY expected.json input.json test.sh verifier.py verifier_support.py /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh && \
    python -c 'import json; assert json.load(open("/tests/expected.json"))["task_id"] == "jacobian/covering-path-lift-bijection"'
