FROM ghcr.io/astral-sh/uv:0.8.4-python3.12-bookworm-slim@sha256:dc7e1d08f8ca979826ec0b68b31c783e0b35b568be6a078d4cbedf38c4cc085e
# Exact dependence-aware expectation verifier.
LABEL jacobian.task="jacobian/random-function-expectation-audit" \
      jacobian.checksum="c9d31fdaf4e1b450ab95c6f02b84badbc292f4ddf68acf36bc4816036598f55d"
COPY expected.json input.json test.sh verifier.py verifier_support.py /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh && \
    python -c 'import json; assert json.load(open("/tests/expected.json"))["task_id"] == "jacobian/random-function-expectation-audit"'
