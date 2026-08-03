FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de
LABEL jacobian.checksum="149db2e5c66d45ff1beae14f473a6b5160f23381fc178d1b19e092cf457b32fc"
LABEL jacobian.task="jacobian/exact-farkas-ldl-slice"
COPY expected.json input.json test.sh verifier.py verifier_support.py /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh
