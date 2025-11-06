# Multi-stage Docker build for STS solver using Astral UV images
FROM astral/uv:python3.12-bookworm-slim AS base

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

# Install Python dependencies with uv
RUN uv sync

# Install additional OR-Tools dependencies
RUN apt-get update && apt-get install -y \
    libc6-dev wget \
    && rm -rf /var/lib/apt/lists/*

# Install MiniZinc
RUN wget https://github.com/MiniZinc/MiniZincIDE/releases/download/2.7.5/MiniZincIDE-2.7.5-bundle-linux-x86_64.tgz \
    && tar -xzf MiniZincIDE-2.7.5-bundle-linux-x86_64.tgz \
    && mv MiniZincIDE-2.7.5-bundle-linux-x86_64 /opt/minizinc \
    && ln -s /opt/minizinc/bin/minizinc /usr/local/bin/minizinc \
    && ln -s /opt/minizinc/bin/mzn2fzn /usr/local/bin/mzn2fzn \
    && rm MiniZincIDE-2.7.5-bundle-linux-x86_64.tgz



# Create results directory
RUN mkdir -p res/{CP,SAT,SMT,MIP}

# Set the default command
CMD ["/bin/bash"]