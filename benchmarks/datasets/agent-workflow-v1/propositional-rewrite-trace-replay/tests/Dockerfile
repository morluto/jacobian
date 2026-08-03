FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de
LABEL jacobian.checksum="1763c28b957ec5f5b44ef1c112044ae32bafbe1570642978b0870c9872ad491c"
LABEL jacobian.task="jacobian/propositional-rewrite-trace-replay"
COPY expected.json input.json test.sh verifier.py verifier_support.py /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh
