FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de
LABEL jacobian.checksum="11e66b6afa9bbea19382fd8326c2dd783472df95d4e579a6c439675b661a90d6"
LABEL jacobian.task="jacobian/cyclotomic-reciprocity-certificate"
COPY expected.json input.json test.sh verifier.py verifier_support.py /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh
