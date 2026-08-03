FROM ghcr.io/astral-sh/uv:0.8.4-python3.12-bookworm-slim@sha256:dc7e1d08f8ca979826ec0b68b31c783e0b35b568be6a078d4cbedf38c4cc085e
# Layered mathematical and rubric-audit verifier.
LABEL jacobian.task="jacobian/log-inequality-meta-audit" \
      jacobian.checksum="e29e8f9288615c53843628f04e00b062f521d86ac40de8f8ec1ef34b1322893c"
COPY expected.json input.json test.sh verifier.py verifier_support.py /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh && \
    python -c 'import json; assert json.load(open("/tests/expected.json"))["task_id"] == "jacobian/log-inequality-meta-audit"'
