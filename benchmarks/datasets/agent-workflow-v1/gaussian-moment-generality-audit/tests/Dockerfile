FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de
LABEL jacobian.checksum="3c30f584d31f900bfb214ef8a474b7e29f76f2c39812f7b82f6a8d0cafd4f563"
LABEL jacobian.task="jacobian/gaussian-moment-generality-audit"
COPY expected.json input.json test.sh verifier.py verifier_support.py /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh && python -c 'import json; assert json.load(open("/tests/expected.json"))["task_id"] == "jacobian/gaussian-moment-generality-audit"'
