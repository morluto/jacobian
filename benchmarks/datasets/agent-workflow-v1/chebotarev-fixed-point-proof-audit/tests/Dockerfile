FROM ghcr.io/astral-sh/uv:0.8.4-python3.12-bookworm-slim@sha256:dc7e1d08f8ca979826ec0b68b31c783e0b35b568be6a078d4cbedf38c4cc085e
# Exact verifier for a conditional Chebotarev proof audit.
LABEL jacobian.task="jacobian/chebotarev-fixed-point-proof-audit" \
      jacobian.checksum="fa36fa496b84f95cd387ed337e43302d50abc4ed9298d269bdc7b396681bea4d"
COPY expected.json input.json test.sh verifier.py verifier_support.py /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh && \
    python -c 'import json; assert json.load(open("/tests/expected.json"))["task_id"] == "jacobian/chebotarev-fixed-point-proof-audit"'
