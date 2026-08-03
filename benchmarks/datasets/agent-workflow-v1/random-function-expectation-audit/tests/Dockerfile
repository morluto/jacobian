FROM ghcr.io/astral-sh/uv:0.8.4-python3.12-bookworm-slim@sha256:dc7e1d08f8ca979826ec0b68b31c783e0b35b568be6a078d4cbedf38c4cc085e
# Exact dependence-aware expectation verifier.
LABEL jacobian.task="jacobian/random-function-expectation-audit" \
      jacobian.checksum="359ce4f195b5d0367da4c4615cb1c7f563ed8f94d406276ccee5ba22b4f68846"
COPY expected.json input.json test.sh verifier.py verifier_support.py /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh && \
    python -c 'import json; assert json.load(open("/tests/expected.json"))["task_id"] == "jacobian/random-function-expectation-audit"'
