import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from groundingdino.util.inference import load_model, predict
from segment_anything import SamPredictor, sam_model_registry

def load_models():
    print("🔧 Loading Grounding DINO and SAM models...")

    dino_config = "./models/GroundingDINO_SwinT_OGC.py"
    dino_checkpoint = "./models/groundingdino_swint_ogc.pth"
    sam_checkpoint = "./models/sam_vit_h.pth"

    dino_model = load_model(
        model_config_path=dino_config,
        model_checkpoint_path=dino_checkpoint
    )

    sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
    sam_predictor = SamPredictor(sam)

    return dino_model, sam_predictor

def run_detection(image_np, prompt_str, dino_model, sam_predictor):
    image = Image.fromarray(image_np).convert("RGB")

    # Preprocess image for GroundingDINO
    transform = transforms.Compose([
        transforms.Resize((800, 800)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])  

    image_tensor = transform(image).unsqueeze(0)  # Add batch dim
    image_tensor = image_tensor.to(next(dino_model.parameters()).device)

    # Remove batch dim before calling predict()
    image_tensor = image_tensor.squeeze(0)  # Shape: (C, H, W)

    boxes, logits, phrases = predict(
        model=dino_model,
        image=image_tensor,
        caption=prompt_str,
        box_threshold=0.3,
        text_threshold=0.25
    )

    return boxes, logits, phrases

