FROM ghcr.io/astral-sh/uv:0.8.4-python3.12-bookworm-slim@sha256:dc7e1d08f8ca979826ec0b68b31c783e0b35b568be6a078d4cbedf38c4cc085e
# Exact graph-semantic verifier for two incompatible C4 invariants.
LABEL jacobian.task="jacobian/c4-characteristic-invariant-audit" \
      jacobian.checksum="64585c4440eef723382441cc9b4ac6e6b5c4d8efeae3a9363cae9ec9732e2d0d"
COPY expected.json input.json test.sh verifier.py verifier_support.py public_contract.json /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh
