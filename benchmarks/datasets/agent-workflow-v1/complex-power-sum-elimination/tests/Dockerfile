FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de
LABEL jacobian.checksum="044d176e65e76b1b5b11668d567cafa6b67ffcd19405a43c94044703ae052526"
LABEL jacobian.task="jacobian/complex-power-sum-elimination"
COPY expected.json input.json test.sh verifier.py verifier_support.py /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh && \
    python -c 'import json; assert json.load(open("/tests/expected.json"))["task_id"] == "jacobian/complex-power-sum-elimination"'
