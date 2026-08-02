FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de
LABEL jacobian.checksum="a45cedd3d772e53edab21be94088dfd063b143a9258d01dde7e2627afa1c7854"
LABEL jacobian.task="jacobian/exact-farkas-ldl-slice"
COPY expected.json input.json test.sh verifier.py verifier_support.py /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh
