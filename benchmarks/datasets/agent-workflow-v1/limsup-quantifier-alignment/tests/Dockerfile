FROM ghcr.io/astral-sh/uv:0.8.4-python3.12-bookworm-slim@sha256:dc7e1d08f8ca979826ec0b68b31c783e0b35b568be6a078d4cbedf38c4cc085e
# Exact semantic-alignment verifier for a limsup quantifier-and-polarity claim.
LABEL jacobian.task="jacobian/limsup-quantifier-alignment" \
      jacobian.checksum="49c691d7208751e8e27b37d4c976d742f262f2aeaefc8840c4817838032f46a8"
COPY expected.json input.json test.sh verifier.py verifier_support.py /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh && \
    python -c 'import json; assert json.load(open("/tests/expected.json"))["task_id"] == "jacobian/limsup-quantifier-alignment"'
