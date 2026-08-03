FROM ghcr.io/astral-sh/uv:0.8.4-python3.12-bookworm-slim@sha256:dc7e1d08f8ca979826ec0b68b31c783e0b35b568be6a078d4cbedf38c4cc085e
LABEL jacobian.checksum="2b2ac36b94aea110ee993e8bba08c03f0218560200c13df22aff44316443c848"
# Exact affine-arrangement verifier for cube and tetrahedron planes.
LABEL jacobian.task="jacobian/hyperplane-arrangement-regions"
COPY expected.json input.json test.sh verifier.py verifier_support.py /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh
