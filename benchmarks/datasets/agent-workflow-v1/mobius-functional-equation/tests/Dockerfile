FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de
LABEL jacobian.checksum="eab2e80329c234327a22f63ce36c3c12c9bbdf5cb027ff35c556e21812ea49be"
LABEL jacobian.task="jacobian/mobius-functional-equation"
COPY expected.json input.json test.sh verifier.py verifier_support.py /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh
