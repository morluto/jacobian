FROM ghcr.io/astral-sh/uv:0.8.4-python3.12-bookworm-slim@sha256:dc7e1d08f8ca979826ec0b68b31c783e0b35b568be6a078d4cbedf38c4cc085e
# Exact semantic-alignment verifier for a limsup quantifier-and-polarity claim.
LABEL jacobian.task="jacobian/limsup-quantifier-alignment" \
      jacobian.checksum="fdd6ca020a5fc166c6053f9957609aefce5581ceacd2319527ebe0c697b62227"
COPY expected.json input.json test.sh verifier.py verifier_support.py /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh && \
    python -c 'import json; assert json.load(open("/tests/expected.json"))["task_id"] == "jacobian/limsup-quantifier-alignment"'
