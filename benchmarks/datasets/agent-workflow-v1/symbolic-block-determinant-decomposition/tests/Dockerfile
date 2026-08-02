FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de
LABEL jacobian.checksum="09e6995b5cf5c5a882bd81eb00da0aea9f9a74287c7052890bfe89ec8bce27d7"
LABEL jacobian.task="jacobian/symbolic-block-determinant-decomposition"
COPY expected.json input.json test.sh verifier.py verifier_support.py /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh
