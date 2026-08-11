FROM ghcr.io/astral-sh/uv:0.8.4-python3.12-bookworm-slim@sha256:dc7e1d08f8ca979826ec0b68b31c783e0b35b568be6a078d4cbedf38c4cc085e
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /opt/provider && \
    curl -fsSL https://files.pythonhosted.org/packages/02/fb/fad8db68c325617708b8033f196c09633fea0962c0878ccc73571ce889b2/gudhi-3.13.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl -o /opt/provider/gudhi-3.13.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl && \
    curl -fsSL https://github.com/GUDHI/gudhi-devel/archive/refs/tags/tags/gudhi-release-3.13.0.tar.gz -o /opt/provider/gudhi-source.tar.gz && \
    printf '%s  %s\n' d7ce52905d147a7e599607758843c7f37525c98e7aeaca87cbbbcbd3adb71598 /opt/provider/gudhi-3.13.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl | sha256sum -c - && \
    printf '%s  %s\n' 7640d716d6a118b7787602826e4144bfa1ad5903a31f4a760a7f3a1e68cfdc02 /opt/provider/gudhi-source.tar.gz | sha256sum -c - && \
    uv pip install --system /opt/provider/gudhi-3.13.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl numpy==2.5.1
WORKDIR /app
COPY __init__.py /opt/benchmarks/__init__.py
COPY tooling/__init__.py /opt/benchmarks/tooling/__init__.py
COPY tooling/command_runner.py /opt/benchmarks/tooling/command_runner.py
COPY tooling/spike_utils.py /opt/benchmarks/tooling/spike_utils.py
COPY datasets/provider-feasibility-v1/gudhi/environment/input.json datasets/provider-feasibility-v1/gudhi/environment/pin.json datasets/provider-feasibility-v1/gudhi/environment/spike.py datasets/provider-feasibility-v1/gudhi/environment/submission_schema.json /app/
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/opt
