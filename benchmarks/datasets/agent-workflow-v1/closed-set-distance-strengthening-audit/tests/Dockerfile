FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de
LABEL jacobian.checksum="8fe3551aea59e0f927f4c84edf4a055962f212b45d8c8e4014e29b7fd8c76ea6"
LABEL jacobian.task="jacobian/closed-set-distance-strengthening-audit"
COPY expected.json input.json test.sh verifier.py verifier_support.py /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh
