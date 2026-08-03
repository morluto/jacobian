FROM ghcr.io/astral-sh/uv:0.8.4-python3.12-bookworm-slim@sha256:dc7e1d08f8ca979826ec0b68b31c783e0b35b568be6a078d4cbedf38c4cc085e
# Independent first-order resolution replay verifier.
LABEL jacobian.task="jacobian/superposition-proof-replay" \
      jacobian.checksum="c766a0ff2e71f7f40d22d06dd0e08b254d3042b0de64da768a474e440ac47dbf"
COPY expected.json input.json test.sh verifier.py verifier_support.py /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh && \
    python -c 'import json; assert json.load(open("/tests/expected.json"))["task_id"] == "jacobian/superposition-proof-replay"'
