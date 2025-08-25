# Multi-stage Docker build for STS solver
FROM ubuntu:22.04 as base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PATH="/root/.local/bin:$PATH"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    curl \
    wget \
    unzip \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml ./
COPY sts_solver/ ./sts_solver/
COPY source/ ./source/

# Install Python dependencies with uv
RUN /root/.local/bin/uv sync

# Install additional OR-Tools dependencies
RUN apt-get update && apt-get install -y \
    libc6-dev \
    && rm -rf /var/lib/apt/lists/*

# Install MiniZinc
RUN wget https://github.com/MiniZinc/MiniZincIDE/releases/download/2.7.5/MiniZincIDE-2.7.5-bundle-linux-x86_64.tgz \
    && tar -xzf MiniZincIDE-2.7.5-bundle-linux-x86_64.tgz \
    && mv MiniZincIDE-2.7.5-bundle-linux-x86_64 /opt/minizinc \
    && ln -s /opt/minizinc/bin/minizinc /usr/local/bin/minizinc \
    && ln -s /opt/minizinc/bin/mzn2fzn /usr/local/bin/mzn2fzn \
    && rm MiniZincIDE-2.7.5-bundle-linux-x86_64.tgz

# Install Gecode solver
RUN apt-get update && apt-get install -y \
    gecode \
    && rm -rf /var/lib/apt/lists/*

# Create results directory
RUN mkdir -p res/{CP,SAT,SMT,MIP}

# Set the default command
CMD ["/bin/bash"]