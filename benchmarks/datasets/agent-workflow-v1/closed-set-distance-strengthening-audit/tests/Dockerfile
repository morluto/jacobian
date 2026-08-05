FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de
LABEL jacobian.checksum="c97f0aa72eea15b3505c282c2ff63912902d414f6efcfb697a2f117c95957140"
LABEL jacobian.task="jacobian/closed-set-distance-strengthening-audit"
COPY expected.json input.json test.sh verifier.py verifier_support.py /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh
