FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de
LABEL jacobian.checksum="775cac2b8f20c4d71441a353da447a14e60217874b5aafd7fa02470538f3dc1c"
LABEL jacobian.task="jacobian/exact-farkas-ldl-slice"
COPY expected.json input.json test.sh verifier.py verifier_support.py /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh
