FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de
LABEL jacobian.checksum="3109430c60edcb417e5c81a7ff33ef5fb1600d6128b6df286acd33a92ba30b56"
LABEL jacobian.task="jacobian/reciprocal-polynomial-classification"
COPY expected.json input.json test.sh verifier.py verifier_support.py /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh
