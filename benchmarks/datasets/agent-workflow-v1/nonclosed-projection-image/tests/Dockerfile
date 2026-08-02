FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de
LABEL jacobian.checksum="e7f146788eaba2cd95627cbb4ffd5f4d407971c3dbeb76038306c44b60cab44f"
LABEL jacobian.task="jacobian/nonclosed-projection-image"
COPY expected.json input.json test.sh verifier.py verifier_support.py /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh && \
    python -c 'import json; assert json.load(open("/tests/expected.json"))["task_id"] == "jacobian/nonclosed-projection-image"'
