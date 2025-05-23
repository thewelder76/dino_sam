FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git curl libgl1 libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

RUN git clone https://github.com/IDEA-Research/GroundingDINO.git /app/GroundingDINO && \
    pip install -e /app/GroundingDINO

RUN git clone https://github.com/facebookresearch/segment-anything.git /app/segment-anything && \
    pip install -e /app/segment-anything

COPY . /workspace
WORKDIR /workspace
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
