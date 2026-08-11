FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de
RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates curl git zstd && \
    rm -rf /var/lib/apt/lists/*
RUN python -m pip install --no-cache-dir attrs==26.1.0 jsonschema==4.26.0 jsonschema-specifications==2025.9.1 referencing==0.37.0 rpds-py==2026.6.3 typing-extensions==4.16.0
RUN mkdir -p /opt/provider && \
    curl -fsSL https://github.com/leanprover/lean4/releases/download/v4.31.0/lean-4.31.0-linux.tar.zst -o /opt/provider/lean.tar.zst && \
    printf '%s  %s\n' 07a633cc8d9151cbc08825ea4cdda50d4b02a2c9cb852c0131b13046f49cad7f /opt/provider/lean.tar.zst | sha256sum -c - && \
    tar --use-compress-program=unzstd -xf /opt/provider/lean.tar.zst -C /opt/provider && \
    git init /opt/provider/repl && \
    git -C /opt/provider/repl remote add origin https://github.com/leanprover-community/repl.git && \
    git -C /opt/provider/repl fetch --depth 1 origin 0cc60263319308000bbaa5354427f775fe3dc7d0 && \
    git -C /opt/provider/repl checkout --detach FETCH_HEAD && \
    PATH=/opt/provider/lean-4.31.0-linux/bin:$PATH lake -d /opt/provider/repl build
LABEL jacobian.checksum="39d10b572218598959187170084b43fda4c6f4b0ae44ba3818f2b98b1015b491"
ENV PATH=/opt/provider/lean-4.31.0-linux/bin:$PATH
COPY expected.json input.json test.sh verifier.py verifier_support.py replay.py public_contract.json /tests/
RUN chmod +x /tests/test.sh
