FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de
LABEL jacobian.checksum="e070fe8cf975ed60abbfd9e5d5c4a52c208eba4bc08ef11060f5b79ae85a79b4"
LABEL jacobian.checksum="placeholder"
LABEL jacobian.task="jacobian/cyclotomic-reciprocity-certificate"
COPY expected.json input.json test.sh verifier.py verifier_support.py /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh
