FROM ghcr.io/astral-sh/uv:0.8.4-python3.12-bookworm-slim@sha256:dc7e1d08f8ca979826ec0b68b31c783e0b35b568be6a078d4cbedf38c4cc085e
LABEL jacobian.checksum="6e060334be66078b685505567bbd808b06d521f9a9a0504d4427d91472db0637"
# Exact verifier for an alternating recurrence stability certificate.
LABEL jacobian.task="jacobian/alternating-recurrence-stability-certificate"
COPY expected.json input.json test.sh verifier.py verifier_support.py /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh
