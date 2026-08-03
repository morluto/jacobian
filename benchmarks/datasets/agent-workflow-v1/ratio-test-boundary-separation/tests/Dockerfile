FROM ghcr.io/astral-sh/uv:0.8.4-python3.12-bookworm-slim@sha256:dc7e1d08f8ca979826ec0b68b31c783e0b35b568be6a078d4cbedf38c4cc085e
LABEL jacobian.checksum="31caf10de4da3a57d45b4ec30fcc54c5dcf7f24ad994524fe34afb2976ae154f"
# Exact verifier for paired ratio-test boundary certificates.
LABEL jacobian.task="jacobian/ratio-test-boundary-separation"
COPY expected.json input.json test.sh verifier.py verifier_support.py /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh
