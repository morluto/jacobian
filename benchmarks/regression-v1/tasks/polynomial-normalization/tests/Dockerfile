FROM ghcr.io/astral-sh/uv:0.8.4-python3.12-bookworm-slim@sha256:dc7e1d08f8ca979826ec0b68b31c783e0b35b568be6a078d4cbedf38c4cc085e
# Build context key: polynomial normalization.
LABEL jacobian.task="polynomial-normalization" \
      jacobian.checksum="b9f91182393d89340af392d7c47a567958ec68892ee694a16e2d58194d759f1f"
COPY authorized_record.json polynomial-normalization-expected.json input.json test.sh verifier.py /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh && \
    python -c 'import json; assert json.load(open("/tests/polynomial-normalization-expected.json"))["task_id"] == "polynomial-normalization"'
