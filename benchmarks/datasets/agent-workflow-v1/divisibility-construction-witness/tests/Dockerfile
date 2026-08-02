FROM ghcr.io/astral-sh/uv:0.8.4-python3.12-bookworm-slim@sha256:dc7e1d08f8ca979826ec0b68b31c783e0b35b568be6a078d4cbedf38c4cc085e
# Exact construction-witness verifier image.
LABEL jacobian.task="jacobian/divisibility-construction-witness" \
      jacobian.checksum="6d886411a29c163f0517ce924ed67488e12e8145cd72f323686acc1015bb3d28"
COPY expected.json input.json test.sh verifier.py verifier_support.py /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh && \
    python -c 'import json; assert json.load(open("/tests/expected.json"))["task_id"] == "jacobian/divisibility-construction-witness"'
