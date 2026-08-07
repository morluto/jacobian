#
# This is an agent/provider runtime. It intentionally remains distinct from
# lean-checker: REPL telemetry is not independent verification evidence.
FROM ghcr.io/astral-sh/uv:0.8.4-python3.12-bookworm-slim@sha256:dc7e1d08f8ca979826ec0b68b31c783e0b35b568be6a078d4cbedf38c4cc085e

ARG LEAN_VERSION=4.31.0
ARG LEAN_ARCHIVE_SHA256=07a633cc8d9151cbc08825ea4cdda50d4b02a2c9cb852c0131b13046f49cad7f
ARG LEAN_REPL_REVISION=0cc60263319308000bbaa5354427f775fe3dc7d0

LABEL org.opencontainers.image.source=https://github.com/morluto/jacobian
LABEL org.opencontainers.image.description="Pinned Lean REPL runtime for Jacobian provider-feasibility tasks"
LABEL io.jacobian.lean.version=${LEAN_VERSION}
LABEL io.jacobian.lean.archive-sha256=${LEAN_ARCHIVE_SHA256}
LABEL io.jacobian.lean-repl.revision=${LEAN_REPL_REVISION}

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl git libgmp10 zstd \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /opt/provider \
    && curl -fsSL "https://github.com/leanprover/lean4/releases/download/v${LEAN_VERSION}/lean-${LEAN_VERSION}-linux.tar.zst" \
      -o /opt/provider/lean.tar.zst \
    && printf '%s  %s\n' "$LEAN_ARCHIVE_SHA256" /opt/provider/lean.tar.zst | sha256sum -c - \
    && tar --use-compress-program=unzstd -xf /opt/provider/lean.tar.zst -C /opt/provider \
    && rm /opt/provider/lean.tar.zst \
    && git init /opt/provider/repl \
    && git -C /opt/provider/repl remote add origin https://github.com/leanprover-community/repl.git \
    && git -C /opt/provider/repl fetch --depth 1 origin "$LEAN_REPL_REVISION" \
    && git -C /opt/provider/repl checkout --detach FETCH_HEAD \
    && PATH=/opt/provider/lean-${LEAN_VERSION}-linux/bin:$PATH lake -d /opt/provider/repl build \
    && /opt/provider/lean-${LEAN_VERSION}-linux/bin/lean --version

ENV PATH=/opt/provider/lean-4.31.0-linux/bin:$PATH
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/opt

WORKDIR /app
