FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de
LABEL jacobian.checksum="b347f1cdd811676a81110e295d98538ad0ccddf4229d752db3b34c8ef6426a3a"
LABEL jacobian.task="jacobian/real-rooted-sign-polynomials"
COPY expected.json input.json test.sh verifier.py verifier_support.py /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh
