FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de
LABEL jacobian.checksum="00742572c31fc013db7958d073c05b8650959fce7c15a33dc99bfbfd15831e31"
LABEL jacobian.task="jacobian/apollonius-gap-repair"
COPY expected.json input.json test.sh verifier.py verifier_support.py /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh && python -c 'import json; assert json.load(open("/tests/expected.json"))["task_id"] == "jacobian/apollonius-gap-repair"'
