FROM ghcr.io/astral-sh/uv:0.8.4-python3.12-bookworm-slim@sha256:dc7e1d08f8ca979826ec0b68b31c783e0b35b568be6a078d4cbedf38c4cc085e
LABEL jacobian.checksum="8e89c3eb275df7f6d2f0411a3f4fd66b786b0b5741ad40d7564f7a7e2d7edd07"
# Exact row-position and balanced matching-layer verifier.
LABEL jacobian.task="jacobian/balanced-row-permutation"
COPY expected.json input.json test.sh verifier.py verifier_support.py /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh
