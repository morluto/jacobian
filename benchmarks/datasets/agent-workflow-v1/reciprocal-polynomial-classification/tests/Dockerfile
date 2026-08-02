FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de
LABEL jacobian.checksum="fae22f20818131ee8bc1e53d7b39c0aae5b362f2b3fc0a020ff476ede5d42f5d"
LABEL jacobian.task="jacobian/reciprocal-polynomial-classification"
COPY expected.json input.json test.sh verifier.py verifier_support.py /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh
