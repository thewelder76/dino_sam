from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import List
from PIL import Image
import numpy as np
import io
import os

from inference import load_models, run_detection

app = FastAPI()

gdino_model, sam_predictor = load_models()

@app.post("/tag-image")
async def tag_image(
    image: UploadFile = File(...),
    prompt: str = Form(...)
):
    image_bytes = await image.read()
    image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_np = np.array(image_pil)

    detections = run_detection(image_np, prompt, gdino_model, sam_predictor)

    return JSONResponse(content={"detections": detections})
