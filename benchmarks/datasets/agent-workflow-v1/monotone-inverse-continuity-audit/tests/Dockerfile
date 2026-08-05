FROM ghcr.io/astral-sh/uv:0.8.4-python3.12-bookworm-slim@sha256:dc7e1d08f8ca979826ec0b68b31c783e0b35b568be6a078d4cbedf38c4cc085e
# Exact verifier for a strict-monotonicity missing-continuity audit.
LABEL jacobian.task="jacobian/monotone-inverse-continuity-audit" \
      jacobian.checksum="4479089e33f2b2cdff3ab148be064ce3808f29d337530fb2478065afc4aadad3"
COPY expected.json input.json test.sh verifier.py verifier_support.py /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh && \
    python -c 'import json; assert json.load(open("/tests/expected.json"))["task_id"] == "jacobian/monotone-inverse-continuity-audit"'
