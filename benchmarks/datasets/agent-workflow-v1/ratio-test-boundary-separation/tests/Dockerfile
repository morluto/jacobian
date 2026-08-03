FROM ghcr.io/astral-sh/uv:0.8.4-python3.12-bookworm-slim@sha256:dc7e1d08f8ca979826ec0b68b31c783e0b35b568be6a078d4cbedf38c4cc085e
LABEL jacobian.checksum="a0f128ca462f38b9bd8477bfe0edce4dba4a46fd3df9e596e33044e00a948e3f"
# Exact verifier for paired ratio-test boundary certificates.
LABEL jacobian.task="jacobian/ratio-test-boundary-separation"
COPY expected.json input.json test.sh verifier.py verifier_support.py /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh
