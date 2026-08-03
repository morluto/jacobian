FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de
LABEL jacobian.checksum="64112d773958db384dcf1d75c25132934a15475dfa285b77ee2c7505432cfb63"
LABEL jacobian.task="jacobian/mobius-functional-equation"
COPY expected.json input.json test.sh verifier.py verifier_support.py /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh
