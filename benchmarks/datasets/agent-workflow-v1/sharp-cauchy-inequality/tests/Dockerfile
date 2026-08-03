FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de
LABEL jacobian.checksum="b6cd8e2c2169ea312d4f4960d4f0fb4d2d7e6c0bc4ac54541c222e0fa30e6816"
LABEL jacobian.task="jacobian/sharp-cauchy-inequality"
COPY expected.json input.json test.sh verifier.py verifier_support.py /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh
