import numpy as np
import torch
import cv2

from groundingdino.util.inference import load_model, predict
from segment_anything import SamPredictor, sam_model_registry

def load_models():
    print("🔧 Loading Grounding DINO and SAM models...")
    dino_model = load_model()
    sam = sam_model_registry["vit_h"](checkpoint="models/sam_vit_h.pth")
    sam.to("cuda" if torch.cuda.is_available() else "cpu")
    predictor = SamPredictor(sam)
    return dino_model, predictor

def run_detection(image_np, prompt, dino_model, sam_predictor):
    from torchvision import transforms
    from PIL import Image

    h, w = image_np.shape[:2]
    image_pil = Image.fromarray(image_np)

    transform = transforms.Compose([
        transforms.Resize((800, 800)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image_tensor = transform(image_pil).unsqueeze(0)

    labels = [s.strip() for s in prompt.split(",")]
    #detections = predict_with_grounding(dino_model, image_tensor, labels)

    prompt_str = ". ".join(labels) + "."  # e.g., "hammer. screwdriver. wrench."
    boxes, logits, phrases = predict(dino_model, image_tensor, prompt_str, box_threshold=0.3, text_threshold=0.25)

    # Reformat detections
    detections = []
    for i in range(len(boxes)):
        box = boxes[i].tolist()  # x0, y0, x1, y1
        label = phrases[i]
        score = float(logits[i].max().item())
        detections.append({"label": label, "score": score, "box": box})


    sam_predictor.set_image(image_np)

    results = []
    for det in detections:
        label, score, box = det['label'], det['score'], det['box']
        x0, y0, x1, y1 = box
        masks, scores, _ = sam_predictor.predict(
            box=np.array([x0, y0, x1, y1]),
            multimask_output=False
        )
        results.append({
            "label": label,
            "confidence": float(score),
            "bbox": [int(x0), int(y0), int(x1), int(y1)],
            "mask": masks[0].astype(bool).tolist()
        })

    return results
