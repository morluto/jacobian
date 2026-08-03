FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de
LABEL jacobian.checksum="03c874fc734d884d0e80168fd04c7c544605a7d1bf80267cd90bd83e9ab1230c"
LABEL jacobian.task="jacobian/polynomial-root-localization-certificate"
COPY expected.json input.json test.sh verifier.py verifier_support.py /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh
