#
# A deliberately small trusted runtime for Harbor verifiers that replay Lean
# source. It contains Lean's kernel/compiler, but neither task material nor an
# agent-facing REPL. Tasks must consume a published digest, never a tag.
FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de

ARG LEAN_VERSION=4.31.0
ARG LEAN_ARCHIVE_SHA256=07a633cc8d9151cbc08825ea4cdda50d4b02a2c9cb852c0131b13046f49cad7f

LABEL org.opencontainers.image.source=https://github.com/morluto/jacobian
LABEL org.opencontainers.image.description="Pinned Lean compiler for independent Jacobian verifier replay"
LABEL io.jacobian.lean.version=${LEAN_VERSION}
LABEL io.jacobian.lean.archive-sha256=${LEAN_ARCHIVE_SHA256}

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl libgmp10 zstd \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /opt/lean \
    && curl -fsSL "https://github.com/leanprover/lean4/releases/download/v${LEAN_VERSION}/lean-${LEAN_VERSION}-linux.tar.zst" \
      -o /opt/lean/lean.tar.zst \
    && printf '%s  %s\n' "$LEAN_ARCHIVE_SHA256" /opt/lean/lean.tar.zst | sha256sum -c - \
    && tar --use-compress-program=unzstd -xf /opt/lean/lean.tar.zst -C /opt/lean \
    && rm /opt/lean/lean.tar.zst \
    && /opt/lean/lean-${LEAN_VERSION}-linux/bin/lean --version \
    && python -m pip install --no-cache-dir \
      attrs==26.1.0 \
      jsonschema==4.26.0 \
      jsonschema-specifications==2025.9.1 \
      referencing==0.37.0 \
      rpds-py==2026.6.3 \
      typing-extensions==4.16.0

ENV PATH=/opt/lean/lean-4.31.0-linux/bin:$PATH
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app
