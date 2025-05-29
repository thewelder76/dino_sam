FROM pytorch/pytorch:2.2.2-cuda11.8-cudnn8-devel

# Environment
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/workspace/models/bert-base-uncased

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl wget ca-certificates ffmpeg \
    libgl1 libglib2.0-0 build-essential ninja-build python3-dev python3-pip \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# --- Clone GroundingDINO ---
RUN git clone https://github.com/IDEA-Research/GroundingDINO.git
WORKDIR /workspace/GroundingDINO

# Install GroundingDINO Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -e .

# ✅ Build CUDA extension from correct location
# This file is in the root of the repo, NOT under models/
RUN TORCH_CUDA_ARCH_LIST="6.1" python setup.py build_ext --inplace

# --- Clone Segment Anything ---
WORKDIR /workspace
RUN git clone https://github.com/facebookresearch/segment-anything.git
ENV PYTHONPATH="${PYTHONPATH}:/workspace/segment-anything"

# --- Copy and install app code ---
WORKDIR /workspace
COPY requirements.txt main.py inference.py ./
RUN pip install --no-cache-dir -r requirements.txt

# --- Optional: Copy model weights here ---
# COPY models/ ./models/

# --- Start server ---
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

