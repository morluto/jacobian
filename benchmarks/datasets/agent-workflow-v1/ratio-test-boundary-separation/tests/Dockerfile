FROM ghcr.io/astral-sh/uv:0.8.4-python3.12-bookworm-slim@sha256:dc7e1d08f8ca979826ec0b68b31c783e0b35b568be6a078d4cbedf38c4cc085e
LABEL jacobian.checksum="f3f539df8007857cb0ffcb37e2b485b23a391426074d6afa1e845858ab0128d3"
# Exact verifier for paired ratio-test boundary certificates.
LABEL jacobian.task="jacobian/ratio-test-boundary-separation"
COPY expected.json input.json test.sh verifier.py verifier_support.py /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh
