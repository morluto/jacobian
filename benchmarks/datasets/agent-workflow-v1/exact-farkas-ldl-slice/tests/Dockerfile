FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de
LABEL jacobian.checksum="e11de2e9903064967181c471ce564e92efa75840de531141c458532d178010ed"
LABEL jacobian.task="jacobian/exact-farkas-ldl-slice"
COPY expected.json input.json test.sh verifier.py verifier_support.py /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh
