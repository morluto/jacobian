FROM ghcr.io/astral-sh/uv:0.8.4-python3.12-bookworm-slim@sha256:dc7e1d08f8ca979826ec0b68b31c783e0b35b568be6a078d4cbedf38c4cc085e
RUN python -m pip install --no-cache-dir attrs==26.1.0 jsonschema==4.26.0 jsonschema-specifications==2025.9.1 referencing==0.37.0 rpds-py==2026.6.3 typing-extensions==4.16.0
LABEL jacobian.task="jacobian/graph-counterexample" \
      jacobian.checksum="1d3eb0774b1265159d94aefd28dbb2939db34cdf92297313a0968fbfeb71487d"
COPY expected.json mathematical-benchmarks-v1-graph-counterexample-input.json test.sh verifier.py verifier_support.py public_contract.json /tests/
COPY mathematical-benchmarks-v1-graph-counterexample-input.json /app/input.json
RUN chmod +x /tests/test.sh && \
    python -c 'import json; assert json.load(open("/tests/expected.json"))["task_id"] == "jacobian/graph-counterexample"'
