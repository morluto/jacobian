FROM ghcr.io/astral-sh/uv:0.8.4-python3.12-bookworm-slim@sha256:dc7e1d08f8ca979826ec0b68b31c783e0b35b568be6a078d4cbedf38c4cc085e
LABEL jacobian.checksum="6b7dc14680d24192f42eb6b7dd74da5f4f0c58aed0c6532abfefd5073aa729cd"
# Exact row-position and balanced matching-layer verifier.
LABEL jacobian.task="jacobian/balanced-row-permutation"
COPY expected.json input.json test.sh verifier.py verifier_support.py /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh
