FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de

ARG LEAN_VERSION=4.31.0
ARG LEAN_ARCHIVE_SHA256=07a633cc8d9151cbc08825ea4cdda50d4b02a2c9cb852c0131b13046f49cad7f

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl libgmp10 zstd \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /opt/lean \
    && curl -fsSL "https://github.com/leanprover/lean4/releases/download/v${LEAN_VERSION}/lean-${LEAN_VERSION}-linux.tar.zst" \
      -o /opt/lean/lean.tar.zst \
    && printf '%s  %s\n' "$LEAN_ARCHIVE_SHA256" /opt/lean/lean.tar.zst | sha256sum -c - \
    && tar --use-compress-program=unzstd -xf /opt/lean/lean.tar.zst -C /opt/lean \
    && rm /opt/lean/lean.tar.zst

ENV PATH=/opt/lean/lean-4.31.0-linux/bin:$PATH
RUN python -m pip install --no-cache-dir attrs==26.1.0 jsonschema==4.26.0 jsonschema-specifications==2025.9.1 referencing==0.37.0 rpds-py==2026.6.3 typing-extensions==4.16.0

LABEL jacobian.task="jacobian/mathlib-gcd-premise-retrieval" \
      jacobian.checksum="d3912670cd4b795123ce3103d3718bbc1eac0e6da962569a6fa3c0981ae6845c"
COPY expected.json input.json test.sh verifier.py verifier_support.py public_contract.json /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh \
    && lean --version \
    && python -c 'import json; assert json.load(open("/tests/expected.json"))["task_id"] == "jacobian/mathlib-gcd-premise-retrieval"'
