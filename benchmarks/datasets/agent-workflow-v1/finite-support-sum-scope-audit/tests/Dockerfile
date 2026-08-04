FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de
LABEL jacobian.checksum="560f4d84eb697e8749913534ddbffd0f24430d2e66c2256266bb55c5fd235975"
LABEL jacobian.task="jacobian/finite-support-sum-scope-audit"
COPY expected.json input.json test.sh verifier.py verifier_support.py /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh
