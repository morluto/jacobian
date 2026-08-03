FROM ghcr.io/astral-sh/uv:0.8.4-python3.12-bookworm-slim@sha256:dc7e1d08f8ca979826ec0b68b31c783e0b35b568be6a078d4cbedf38c4cc085e
# Exact finite optimization and proof-repair verifier.
LABEL jacobian.task="jacobian/metric-tsp-proof-repair" \
      jacobian.checksum="c4d4a198f553497536e9434a0079dbdf01eda36d5158ef2cdbab9157cf2262f6"
COPY expected.json input.json test.sh verifier.py verifier_support.py /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh && \
    python -c 'import json; assert json.load(open("/tests/expected.json"))["task_id"] == "jacobian/metric-tsp-proof-repair"'
