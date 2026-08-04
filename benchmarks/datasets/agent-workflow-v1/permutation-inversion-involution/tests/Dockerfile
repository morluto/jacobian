FROM ghcr.io/astral-sh/uv:0.8.4-python3.12-bookworm-slim@sha256:dc7e1d08f8ca979826ec0b68b31c783e0b35b568be6a078d4cbedf38c4cc085e
# Exact verifier for an inversion-complementing permutation involution.
LABEL jacobian.task="jacobian/permutation-inversion-involution" \
      jacobian.checksum="9119a81b1e5726ef41f948a31355e192d0697037dc766268cf1e6d216a8a78a3"
COPY expected.json input.json test.sh verifier.py verifier_support.py /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh && \
    python -c 'import json; assert json.load(open("/tests/expected.json"))["task_id"] == "jacobian/permutation-inversion-involution"'
