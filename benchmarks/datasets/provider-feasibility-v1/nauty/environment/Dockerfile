FROM ghcr.io/astral-sh/uv:0.8.4-python3.12-bookworm-slim@sha256:dc7e1d08f8ca979826ec0b68b31c783e0b35b568be6a078d4cbedf38c4cc085e
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*
RUN mkdir -p /opt/provider && \
    curl -fsSL https://users.cecs.anu.edu.au/~bdm/nauty/nauty2_9_3.tar.gz -o /opt/provider/nauty.tar.gz && \
    printf '%s  %s\n' 9fc4edae04f88a0f5883985be3b39cf7f898fd6cc96e96b9ee25452743cc1b5b /opt/provider/nauty.tar.gz | sha256sum -c - && \
    tar -xzf /opt/provider/nauty.tar.gz -C /opt/provider && \
    cd /opt/provider/nauty2_9_3 && ./configure && make -j2
WORKDIR /app
COPY __init__.py /opt/benchmarks/__init__.py
COPY tooling/__init__.py /opt/benchmarks/tooling/__init__.py
COPY tooling/command_runner.py /opt/benchmarks/tooling/command_runner.py
COPY tooling/spike_utils.py /opt/benchmarks/tooling/spike_utils.py
COPY datasets/provider-feasibility-v1/nauty/environment/input.json datasets/provider-feasibility-v1/nauty/environment/nauty_provider_pin.json datasets/provider-feasibility-v1/nauty/environment/spike.py datasets/provider-feasibility-v1/nauty/environment/submission_schema.json /app/
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/opt
