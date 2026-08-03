FROM ghcr.io/astral-sh/uv:0.8.4-python3.12-bookworm-slim@sha256:dc7e1d08f8ca979826ec0b68b31c783e0b35b568be6a078d4cbedf38c4cc085e
LABEL jacobian.checksum="d814a75a2df084d9a7b72b551c4a9c948993c35dce92b5e3d4fb501a1bf9073b"
COPY verifier.py verifier_support.py input.json expected.json test.sh /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh && \
    python -c 'import json; assert json.load(open("/tests/expected.json"))["task_id"] == "jacobian/rank-one-determinant-limit"'
