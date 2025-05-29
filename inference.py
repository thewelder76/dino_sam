import os
import sys
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from segment_anything import SamPredictor, sam_model_registry
from groundingdino.util.inference import load_model, predict
from torch.utils.cpp_extension import load as load_cpp_extension
import builtins


def _load_custom_cuda_ops():
    """
    Build and inject the GroundingDINO custom C++/CUDA ops (_C).
    This makes `_C.ms_deform_attn_forward` available globally.
    """
    print("🔧 Compiling GroundingDINO custom CUDA ops as _C...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    vision_path = os.path.join(
        script_dir,
        "GroundingDINO",
        "groundingdino",
        "models",
        "GroundingDINO",
        "csrc"
    )

    _C = load_cpp_extension(
        name="_C",
        sources=[os.path.join(vision_path, "vision.cpp")],
        extra_cflags=["-O3"],
        extra_include_paths=[vision_path],
        verbose=True,
    )

    builtins._C = _C
    print("✅ Loaded _C (CUDA extension) into global namespace.")


def load_models():
    print("🚀 Loading GroundingDINO and SAM models...")

    _load_custom_cuda_ops()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    dino_config = os.path.join(script_dir, "GroundingDINO", "groundingdino", "config", "GroundingDINO_SwinT_OGC.py")
    dino_checkpoint = os.path.join(script_dir, "models", "groundingdino_swint_ogc.pth")
    sam_checkpoint = os.path.join(script_dir, "models", "sam_vit_h.pth")

    dino_model = load_model(
        model_config_path=dino_config,
        model_checkpoint_path=dino_checkpoint
    )

    sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
    sam_predictor = SamPredictor(sam)

    return dino_model, sam_predictor


def run_detection(image_np, prompt_str, dino_model, sam_predictor):
    image = Image.fromarray(image_np).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((800, 800)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    image_tensor = transform(image).unsqueeze(0).to(
        next(dino_model.parameters()).device
    )
    image_tensor = image_tensor.squeeze(0)

    boxes, logits, phrases = predict(
        model=dino_model,
        image=image_tensor,
        caption=prompt_str,
        box_threshold=0.3,
        text_threshold=0.25
    )

    return boxes, logits, phrases

