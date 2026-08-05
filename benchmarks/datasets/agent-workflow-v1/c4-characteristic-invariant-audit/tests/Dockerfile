FROM ghcr.io/astral-sh/uv:0.8.4-python3.12-bookworm-slim@sha256:dc7e1d08f8ca979826ec0b68b31c783e0b35b568be6a078d4cbedf38c4cc085e
# Exact graph-semantic verifier for two incompatible C4 invariants.
LABEL jacobian.task="jacobian/c4-characteristic-invariant-audit" \
      jacobian.checksum="d1d085a87ae925f8985196088b89a596c11efb55455715265e9ca2ca93ddd352"
COPY expected.json input.json test.sh verifier.py verifier_support.py /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh
