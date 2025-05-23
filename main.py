from fastapi import FastAPI, UploadFile, File, Form
from PIL import Image
import numpy as np
from io import BytesIO

from inference import run_detection, load_models

app = FastAPI()

# Load once at startup
gdino_model, sam_predictor = load_models()

@app.post("/tag-image")
async def tag_image(
    uploaded_file: UploadFile = File(...),
    prompt: str = Form(...)
):
    image_bytes = await uploaded_file.read()
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image_np = np.array(image)

    detections = run_detection(image_np, prompt, gdino_model, sam_predictor)
    return {
        "boxes": detections[0].tolist(),
        "logits": detections[1].tolist(),
        "phrases": detections[2],
    }

