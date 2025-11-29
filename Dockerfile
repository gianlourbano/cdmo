# Multi-stage Docker build for STS solver using Astral UV images
FROM minizinc/minizinc:edge-jammy AS base

RUN apt-get update && apt-get install -y \
    libc6-dev wget curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ADD https://astral.sh/uv/install.sh /uv-installer.sh

# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml ./
COPY sts_solver/ ./sts_solver/
COPY source/ ./source/
COPY README.md ./
COPY run_all.sh ./

# Install Python dependencies with uv
RUN uv sync

# Install additional OR-Tools dependencies


# Install MiniZinc bundle (amd64/x86_64) at a modern version
# Force amd64 userspace to avoid architecture mismatches
# ARG MINIZINC_VERSION=2.9.3
# RUN set -eux; \
#     dpkg --print-architecture | grep -q amd64 || (echo "This image must be built for amd64" && exit 1); \
#     url="https://github.com/MiniZinc/MiniZincIDE/releases/download/${MINIZINC_VERSION}/MiniZincIDE-${MINIZINC_VERSION}-bundle-linux-x86_64.tgz"; \
#     curl -fL "$url" -o /tmp/minizinc.tgz; \
#     tar -xzf /tmp/minizinc.tgz -C /opt; \
#     dir=$(tar -tzf /tmp/minizinc.tgz | head -1 | cut -d/ -f1); \
#     mv "/opt/$dir" /opt/minizinc; \
#     ln -sf /opt/minizinc/bin/minizinc /usr/local/bin/minizinc; \
#     ln -sf /opt/minizinc/bin/mzn2fzn /usr/local/bin/mzn2fzn; \
#     rm -f /tmp/minizinc.tgz

# Create results directory
RUN mkdir -p res/{CP,SAT,SMT,MIP}

# Set the default command
CMD ["/bin/bash"]