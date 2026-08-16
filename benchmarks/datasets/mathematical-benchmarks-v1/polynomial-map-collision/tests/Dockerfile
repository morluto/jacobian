FROM ghcr.io/astral-sh/uv:0.8.4-python3.12-bookworm-slim@sha256:dc7e1d08f8ca979826ec0b68b31c783e0b35b568be6a078d4cbedf38c4cc085e
RUN python -m pip install --no-cache-dir attrs==26.1.0 jsonschema==4.26.0 jsonschema-specifications==2025.9.1 referencing==0.37.0 rpds-py==2026.6.3 typing-extensions==4.16.0
LABEL jacobian.task="jacobian/polynomial-map-collision" \
      jacobian.checksum="29295bb0cb09e16405d0e623ae329464d122a7fa51634d0563a598abfa886c00"
COPY authorized_record.json mathematical-benchmarks-v1-polynomial-map-collision-expected.json mathematical-benchmarks-v1-polynomial-map-collision-input.json test.sh verifier.py verifier_support.py public_contract.json /tests/
COPY mathematical-benchmarks-v1-polynomial-map-collision-input.json /app/input.json
RUN chmod +x /tests/test.sh && \
    python -c 'import json; assert json.load(open("/tests/mathematical-benchmarks-v1-polynomial-map-collision-expected.json"))["task_id"] == "jacobian/polynomial-map-collision"'
