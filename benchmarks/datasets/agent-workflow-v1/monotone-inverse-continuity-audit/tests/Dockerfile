FROM ghcr.io/astral-sh/uv:0.8.4-python3.12-bookworm-slim@sha256:dc7e1d08f8ca979826ec0b68b31c783e0b35b568be6a078d4cbedf38c4cc085e
# Exact verifier for a strict-monotonicity missing-continuity audit.
LABEL jacobian.task="jacobian/monotone-inverse-continuity-audit" \
      jacobian.checksum="0c2be788402add6e3c966234523160ba6064db86d1dee16e0555245d1be0346c"
COPY expected.json input.json test.sh verifier.py verifier_support.py /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh && \
    python -c 'import json; assert json.load(open("/tests/expected.json"))["task_id"] == "jacobian/monotone-inverse-continuity-audit"'
