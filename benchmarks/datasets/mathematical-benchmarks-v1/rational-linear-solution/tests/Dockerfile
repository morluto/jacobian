FROM ghcr.io/astral-sh/uv:0.8.4-python3.12-bookworm-slim@sha256:dc7e1d08f8ca979826ec0b68b31c783e0b35b568be6a078d4cbedf38c4cc085e
RUN python -m pip install --no-cache-dir attrs==26.1.0 jsonschema==4.26.0 jsonschema-specifications==2025.9.1 referencing==0.37.0 rpds-py==2026.6.3 typing-extensions==4.16.0
LABEL jacobian.task="jacobian/rational-linear-solution" \
      jacobian.checksum="2c90323c413926d2b73f8e8f031ee95c5517870c8271dac32faf8d898c06a736"
COPY expected.json mathematical-benchmarks-v1-rational-linear-solution-input.json test.sh verifier.py verifier_support.py public_contract.json /tests/
COPY mathematical-benchmarks-v1-rational-linear-solution-input.json /app/input.json
RUN chmod +x /tests/test.sh && \
    python -c 'import json; assert json.load(open("/tests/expected.json"))["task_id"] == "jacobian/rational-linear-solution"'
