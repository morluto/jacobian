FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de
LABEL jacobian.task="jacobian/vizing-bounded-cartesian-products" \
      jacobian.checksum="0eba7b4cb29a47aae2d613ea04941d599c32998dda49aa089653441a50db6945"
COPY expected.json input.json public_contract.json test.sh verifier.py verifier_support.py /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh && python -c 'import json; assert json.load(open("/tests/input.json"))["task_id"] == "jacobian/vizing-bounded-cartesian-products"'
