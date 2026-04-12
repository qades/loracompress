# Dockerfile for LoRA Reproduction Experiment
# For ROCm (AMD Strix Halo): Use docker.io/kyuz0/amd-strix-halo-llm-finetuning:latest
# For CUDA (NVIDIA): Use nvidia/cuda:12.1-devel-ubuntu22.04 with PyTorch CUDA

ARG BASE_IMAGE=docker.io/kyuz0/amd-strix-halo-llm-finetuning:latest
FROM ${BASE_IMAGE}

# Set working directory
WORKDIR /workspace/lora-reproduction

# Install Python dependencies
# Note: ROCm PyTorch is pre-installed in the base image
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY configs/ ./configs/

# Make scripts executable
RUN chmod +x scripts/*.py

# Create results directory
RUN mkdir -p results autoresearch_results

# Set environment variables for reproducibility
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/workspace/.cache/huggingface
ENV TRANSFORMERS_CACHE=/workspace/.cache/transformers

# ROCm specific (harmless for CUDA)
ENV HSA_OVERRIDE_GFX_VERSION=11.0.0
ENV PYTORCH_HIP_ALLOC_CONF=expandable_segments:True

# Default command
CMD ["/bin/bash"]
