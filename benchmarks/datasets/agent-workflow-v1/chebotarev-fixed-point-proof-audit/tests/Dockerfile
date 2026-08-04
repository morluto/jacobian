FROM ghcr.io/astral-sh/uv:0.8.4-python3.12-bookworm-slim@sha256:dc7e1d08f8ca979826ec0b68b31c783e0b35b568be6a078d4cbedf38c4cc085e
# Exact verifier for a conditional Chebotarev proof audit.
LABEL jacobian.task="jacobian/chebotarev-fixed-point-proof-audit" \
      jacobian.checksum="a06eee131f42c459d07b4f236d61896ef7a6e251c3cf36ca34ba8e82b61996eb"
COPY expected.json input.json test.sh verifier.py verifier_support.py /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh && \
    python -c 'import json; assert json.load(open("/tests/expected.json"))["task_id"] == "jacobian/chebotarev-fixed-point-proof-audit"'
