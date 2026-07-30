FROM ghcr.io/astral-sh/uv:0.8.4-python3.12-bookworm-slim@sha256:dc7e1d08f8ca979826ec0b68b31c783e0b35b568be6a078d4cbedf38c4cc085e
LABEL jacobian.task="hermite-normal-form" \
      jacobian.checksum="8b2890a74b0726d667bac7e512ce86d3bc76100e65f3d6b0e43e4c4f10a7e2f5"
COPY authorized_record.json expected.json input.json test.sh verifier.py /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh && \
    python -c 'import json; assert json.load(open("/tests/expected.json"))["task_id"] == "hermite-normal-form"'
