FROM ghcr.io/astral-sh/uv:0.8.4-python3.12-bookworm-slim@sha256:dc7e1d08f8ca979826ec0b68b31c783e0b35b568be6a078d4cbedf38c4cc085e
LABEL jacobian.checksum="c1722c65261e83f5df6f83e29b012d2c2692afc8a2fc1e960d67148418227dd6"
# Exact symbolic verifier for a cyclic sharp inequality.
LABEL jacobian.task="jacobian/cyclic-vector-inequality"
COPY expected.json input.json test.sh verifier.py verifier_support.py /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh
