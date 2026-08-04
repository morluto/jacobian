FROM ghcr.io/astral-sh/uv:0.8.4-python3.12-bookworm-slim@sha256:dc7e1d08f8ca979826ec0b68b31c783e0b35b568be6a078d4cbedf38c4cc085e
LABEL jacobian.checksum="58a8f35ca79df7781f458bbc51417e361b0412fb49fe6acb659557e70e61fc73"
# Exact finite-field polynomial verifier for an irreducibility proof repair.
LABEL jacobian.task="jacobian/finite-field-irreducibility-repair"
COPY expected.json input.json test.sh verifier.py verifier_support.py /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh
