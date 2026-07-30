FROM ghcr.io/astral-sh/uv:0.8.4-python3.12-bookworm-slim@sha256:dc7e1d08f8ca979826ec0b68b31c783e0b35b568be6a078d4cbedf38c4cc085e
LABEL jacobian.task="rational-linear-solution" \
      jacobian.checksum="993071b1e86dce854416bc0a5eaf0413d4dd1a0a85b50d5f1be0da6d320c8b62"
COPY expected.json input.json test.sh verifier.py /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh && \
    python -c 'import json; assert json.load(open("/tests/expected.json"))["task_id"] == "rational-linear-solution"'
