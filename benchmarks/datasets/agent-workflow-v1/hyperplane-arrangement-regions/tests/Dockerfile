FROM ghcr.io/astral-sh/uv:0.8.4-python3.12-bookworm-slim@sha256:dc7e1d08f8ca979826ec0b68b31c783e0b35b568be6a078d4cbedf38c4cc085e
LABEL jacobian.checksum="0d361315defdc6a0b54f713f2f96b4667f7b49ecdadc043fdb94b1be2c2098ef"
# Exact affine-arrangement verifier for cube and tetrahedron planes.
LABEL jacobian.task="jacobian/hyperplane-arrangement-regions"
COPY expected.json input.json test.sh verifier.py verifier_support.py /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh
