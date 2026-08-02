FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de
LABEL jacobian.checksum="b15b280e00e0f3412177c6418cc7afd66a45b69aa48470399d2a2a7ab01b56e3"
LABEL jacobian.task="jacobian/generated-lemma-vacuity-audit"
COPY expected.json input.json test.sh verifier.py verifier_support.py /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh && \
    python -c 'import json; assert json.load(open("/tests/expected.json"))["task_id"] == "jacobian/generated-lemma-vacuity-audit"'
