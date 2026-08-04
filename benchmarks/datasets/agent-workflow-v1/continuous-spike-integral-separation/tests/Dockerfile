FROM ghcr.io/astral-sh/uv:0.8.4-python3.12-bookworm-slim@sha256:dc7e1d08f8ca979826ec0b68b31c783e0b35b568be6a078d4cbedf38c4cc085e
# Exact verifier for a continuous triangular-spike construction.
LABEL jacobian.task="jacobian/continuous-spike-integral-separation" \
      jacobian.checksum="58108c9ad5112bcbe8f1d13a54f0de8ee0cd28bb419471f60640d3ed9e5f9a15"
COPY expected.json input.json test.sh verifier.py verifier_support.py /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh && python -c 'import json; assert json.load(open("/tests/expected.json"))["task_id"] == "jacobian/continuous-spike-integral-separation"'
