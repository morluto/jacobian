FROM ghcr.io/astral-sh/uv:0.8.4-python3.12-bookworm-slim@sha256:dc7e1d08f8ca979826ec0b68b31c783e0b35b568be6a078d4cbedf38c4cc085e
LABEL jacobian.checksum="4b341eaead1494c3e7110b8d184d44a75119687e110e19a42934f5748fa2da30"
# Exact rational verifier for a rank-one spectral limit certificate.
LABEL jacobian.task="jacobian/rank-one-spectral-limit-certificate"
COPY expected.json input.json test.sh verifier.py verifier_support.py /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh
