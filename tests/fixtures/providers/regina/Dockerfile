FROM ghcr.io/astral-sh/uv:0.8.4-python3.12-bookworm-slim@sha256:dc7e1d08f8ca979826ec0b68b31c783e0b35b568be6a078d4cbedf38c4cc085e
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /opt/provider && \
    curl -fsSL https://files.pythonhosted.org/packages/68/42/b4a994e393298c08c361342ee19bf240b7b16f155c492ce0631082c14e66/regina-7.4.1-cp312-cp312-manylinux_2_28_x86_64.whl -o /opt/provider/regina-7.4.1-cp312-cp312-manylinux_2_28_x86_64.whl && \
    curl -fsSL https://github.com/regina-normal/regina/releases/download/regina-7.4.1/regina-7.4.1.tar.gz -o /opt/provider/regina-source.tar.gz && \
    printf '%s  %s\n' d33ccc69697ec680a8dcf8f0c663428853da44f3578898ad9b908e4669e83c71 /opt/provider/regina-7.4.1-cp312-cp312-manylinux_2_28_x86_64.whl | sha256sum -c - && \
    printf '%s  %s\n' 74131014099b77083dd1d09004fd6bc972421c792d188cdf43f0fd11ca6cb2ed /opt/provider/regina-source.tar.gz | sha256sum -c - && \
    uv pip install --system /opt/provider/regina-7.4.1-cp312-cp312-manylinux_2_28_x86_64.whl
WORKDIR /app
COPY benchmarks/__init__.py /opt/benchmarks/__init__.py
COPY benchmarks/tooling/__init__.py /opt/benchmarks/tooling/__init__.py
COPY tools/command_runner.py /opt/tools/command_runner.py
COPY benchmarks/tooling/spike_utils.py /opt/benchmarks/tooling/spike_utils.py
COPY tests/fixtures/providers/regina/input.json tests/fixtures/providers/regina/pin.json /app/
COPY benchmarks/tooling/providers/regina.py /app/spike.py
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/opt
