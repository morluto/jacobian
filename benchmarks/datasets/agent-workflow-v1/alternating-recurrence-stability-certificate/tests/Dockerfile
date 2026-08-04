FROM ghcr.io/astral-sh/uv:0.8.4-python3.12-bookworm-slim@sha256:dc7e1d08f8ca979826ec0b68b31c783e0b35b568be6a078d4cbedf38c4cc085e
LABEL jacobian.checksum="34e339b9432ebf17b9614c7cb2e6fec3053c03d947fcdcf223d72ffb109c6b38"
# Exact verifier for an alternating recurrence stability certificate.
LABEL jacobian.task="jacobian/alternating-recurrence-stability-certificate"
COPY expected.json input.json test.sh verifier.py verifier_support.py /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh
