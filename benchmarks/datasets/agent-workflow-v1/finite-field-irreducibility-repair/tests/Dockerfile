FROM ghcr.io/astral-sh/uv:0.8.4-python3.12-bookworm-slim@sha256:dc7e1d08f8ca979826ec0b68b31c783e0b35b568be6a078d4cbedf38c4cc085e
LABEL jacobian.checksum="5291ae3bd33fe3d5ffb67e7db01295f73d2515ace737b19867b47aceb18f3bd1"
# Exact finite-field polynomial verifier for an irreducibility proof repair.
LABEL jacobian.task="jacobian/finite-field-irreducibility-repair"
COPY expected.json input.json test.sh verifier.py verifier_support.py /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh
