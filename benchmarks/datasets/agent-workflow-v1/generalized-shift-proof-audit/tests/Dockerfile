FROM ghcr.io/astral-sh/uv:0.8.4-python3.12-bookworm-slim@sha256:dc7e1d08f8ca979826ec0b68b31c783e0b35b568be6a078d4cbedf38c4cc085e
LABEL jacobian.checksum="1cf01884cdae0ccef4c7fbfe1e0fb790feca871dee60c9c2af0de32550e6e888"
COPY verifier.py verifier_support.py input.json expected.json test.sh /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh && \
    python -c 'import json; assert json.load(open("/tests/expected.json"))["task_id"] == "jacobian/generalized-shift-proof-audit"'
