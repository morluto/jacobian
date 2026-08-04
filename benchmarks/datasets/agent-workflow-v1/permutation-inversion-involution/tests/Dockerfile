FROM ghcr.io/astral-sh/uv:0.8.4-python3.12-bookworm-slim@sha256:dc7e1d08f8ca979826ec0b68b31c783e0b35b568be6a078d4cbedf38c4cc085e
# Exact verifier for an inversion-complementing permutation involution.
LABEL jacobian.task="jacobian/permutation-inversion-involution" \
      jacobian.checksum="3e5f738c756f6a2aa0c8f91ccf0a8704d6d55691672cc22287e8be80170ed72f"
COPY expected.json input.json test.sh verifier.py verifier_support.py /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh && \
    python -c 'import json; assert json.load(open("/tests/expected.json"))["task_id"] == "jacobian/permutation-inversion-involution"'
