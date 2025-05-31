from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
from io import BytesIO

from inference import run_detection, load_models

app = FastAPI()

# Optional CORS for local testing or frontend use
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models once at startup
gdino_model, sam_predictor = load_models()

@app.post("/tag-image")
async def tag_image(
    uploaded_file: UploadFile = File(...),
    prompt: str = Form(...)
):
    image_bytes = await uploaded_file.read()
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image_np = np.array(image)

    boxes, logits, phrases = run_detection(image_np, prompt, gdino_model, sam_predictor)
    res={
        "boxes": boxes.tolist() if boxes is not None else [],
        "logits": logits.tolist() if logits is not None else [],
        "phrases": phrases if phrases is not None else [],
    }
    print(res)
    return res
