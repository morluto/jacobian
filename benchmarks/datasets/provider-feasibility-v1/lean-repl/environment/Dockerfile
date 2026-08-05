FROM ghcr.io/astral-sh/uv:0.8.4-python3.12-bookworm-slim@sha256:dc7e1d08f8ca979826ec0b68b31c783e0b35b568be6a078d4cbedf38c4cc085e
RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates curl git zstd && \
    rm -rf /var/lib/apt/lists/*
RUN mkdir -p /opt/provider && \
    curl -fsSL https://github.com/leanprover/lean4/releases/download/v4.31.0/lean-4.31.0-linux.tar.zst -o /opt/provider/lean.tar.zst && \
    tar --use-compress-program=unzstd -xf /opt/provider/lean.tar.zst -C /opt/provider && \
    git init /opt/provider/repl && \
    git -C /opt/provider/repl remote add origin https://github.com/leanprover-community/repl.git && \
    git -C /opt/provider/repl fetch --depth 1 origin 0cc60263319308000bbaa5354427f775fe3dc7d0 && \
    git -C /opt/provider/repl checkout --detach FETCH_HEAD && \
    PATH=/opt/provider/lean-4.31.0-linux/bin:$PATH lake -d /opt/provider/repl build
WORKDIR /app
COPY __init__.py /opt/benchmarks/__init__.py
COPY tooling/__init__.py /opt/benchmarks/tooling/__init__.py
COPY tooling/command_runner.py /opt/benchmarks/tooling/command_runner.py
COPY datasets/provider-feasibility-v1/lean-repl/environment/input.json datasets/provider-feasibility-v1/lean-repl/environment/lean_repl_pin.json datasets/provider-feasibility-v1/lean-repl/environment/spike.py datasets/provider-feasibility-v1/lean-repl/environment/submission_schema.json /app/
ENV PATH=/opt/provider/lean-4.31.0-linux/bin:$PATH
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/opt
