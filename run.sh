#!/bin/bash
docker build -t dino-sam-api .
docker run --gpus all -p 8000:8000 -v $(pwd)/models:/workspace/models dino-sam-api
