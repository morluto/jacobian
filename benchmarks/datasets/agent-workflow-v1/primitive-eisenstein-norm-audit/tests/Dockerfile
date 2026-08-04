FROM ghcr.io/astral-sh/uv:0.8.4-python3.12-bookworm-slim@sha256:dc7e1d08f8ca979826ec0b68b31c783e0b35b568be6a078d4cbedf38c4cc085e
# Exact local-arithmetic verifier for a primitive Eisenstein norm audit.
LABEL jacobian.checksum="3fb5402a5fa670ef4b043c4b008f1968ca3ad3c16175113b4b5b107512be077f"
LABEL jacobian.task="jacobian/primitive-eisenstein-norm-audit"
COPY expected.json input.json test.sh verifier.py verifier_support.py /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh
