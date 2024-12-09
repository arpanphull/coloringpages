# Use the NVIDIA CUDA runtime image with cuDNN and Ubuntu 22.04
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Set the working directory
WORKDIR /workspace

# Remove the faulty APT configuration file and install system dependencies
RUN rm -f /etc/apt/apt.conf.d/docker-clean && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 python3-pip git ca-certificates curl \
    python3-dev gcc build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
# Disable progress bar for pip to reduce resource usage
ENV RICH_NO_PROGRESS true
# Copy requirements.txt into the container and install dependencies without caching
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir --progress-bar off -r requirements.txt


COPY . .
CMD ["python3", "inference_new_img2img.py"]