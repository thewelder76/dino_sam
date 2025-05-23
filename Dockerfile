FROM pytorch/pytorch:2.2.2-cuda11.8-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl libgl1 libglib2.0-0 ffmpeg build-essential \
    && rm -rf /var/lib/apt/lists/*

# Only install Python deps that don't include 'torch'
COPY requirements.txt .
RUN grep -v '^torch' requirements.txt > trimmed.txt \
 && pip install --upgrade pip && pip install -r trimmed.txt

# Clone and install Grounding DINO (with native extension)
RUN git clone https://github.com/IDEA-Research/GroundingDINO.git /app/GroundingDINO \
 && pip install -e /app/GroundingDINO \
 && cd /app/GroundingDINO \
 && python setup.py build_ext --inplace

# Clone and install Segment Anything
RUN git clone https://github.com/facebookresearch/segment-anything.git /app/segment-anything \
 && pip install -e /app/segment-anything

# Copy your app code
WORKDIR /workspace
COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

