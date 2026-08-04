FROM ghcr.io/astral-sh/uv:0.8.4-python3.12-bookworm-slim@sha256:dc7e1d08f8ca979826ec0b68b31c783e0b35b568be6a078d4cbedf38c4cc085e
LABEL jacobian.checksum="a92251622c9a72b885d8b874592ba0ce963f093e37f68159b0a011c82345b0db"
# Exact verifier for multivariable path-dependent limit certificates.
LABEL jacobian.task="jacobian/path-dependent-limit"
COPY expected.json input.json test.sh verifier.py verifier_support.py /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh
