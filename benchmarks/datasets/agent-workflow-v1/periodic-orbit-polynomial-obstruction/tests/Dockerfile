FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de
LABEL jacobian.checksum="b0c23755d49960d26ade4714a65450f5f9062fc9bad6e68eec613d0f982711e5"
LABEL jacobian.task="jacobian/periodic-orbit-polynomial-obstruction"
COPY expected.json input.json test.sh verifier.py verifier_support.py /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh && \
    python -c 'import json; assert json.load(open("/tests/expected.json"))["task_id"] == "jacobian/periodic-orbit-polynomial-obstruction"'
