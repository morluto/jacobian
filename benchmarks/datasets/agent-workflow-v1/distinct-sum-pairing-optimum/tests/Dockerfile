FROM ghcr.io/astral-sh/uv:0.8.4-python3.12-bookworm-slim@sha256:dc7e1d08f8ca979826ec0b68b31c783e0b35b568be6a078d4cbedf38c4cc085e
# Exhaustive finite optimizer for canonical distinct-sum pairings.
LABEL jacobian.task="jacobian/distinct-sum-pairing-optimum" \
      jacobian.checksum="5a3e6a7f042d93d7f1aaf238e70d01acdc1e301b16986dcd23135c2864acb601"
COPY expected.json input.json test.sh verifier.py verifier_support.py /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh && \
    python -c 'import json; assert json.load(open("/tests/expected.json"))["task_id"] == "jacobian/distinct-sum-pairing-optimum"'
