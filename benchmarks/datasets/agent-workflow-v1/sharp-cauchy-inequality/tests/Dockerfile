FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de
LABEL jacobian.checksum="eaed98eea67cc4ff5c2d455edfc9ed4cd345478110ed02609dd80e0c4656856b"
LABEL jacobian.task="jacobian/sharp-cauchy-inequality"
COPY expected.json input.json test.sh verifier.py verifier_support.py /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh
