FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de
LABEL jacobian.checksum="012daefcbace17b4046d17886afe9d0610caa9e5728f5a9a6ba1542f4e92400d"
LABEL jacobian.task="jacobian/cyclotomic-reciprocity-certificate"
COPY expected.json input.json test.sh verifier.py verifier_support.py /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh
