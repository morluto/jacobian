FROM ghcr.io/astral-sh/uv:0.8.4-python3.12-bookworm-slim@sha256:dc7e1d08f8ca979826ec0b68b31c783e0b35b568be6a078d4cbedf38c4cc085e
LABEL jacobian.task="graph-artifact-composition" \
      jacobian.checksum="f90144cb388b13c1c480b1fc769fb4366c5f9c68edc1caae6eaaa692844481c3"
COPY expected.json input.json test.sh verifier.py /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh && \
    python -c 'import json; assert json.load(open("/tests/expected.json"))["task_id"] == "graph-artifact-composition"'
