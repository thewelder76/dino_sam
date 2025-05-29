from torch.utils.cpp_extension import load
import os

vision_path = "groundingdino/models/GroundingDINO/csrc"

vision_ext = load(
    name="vision_ext",
    sources=[
        os.path.join(vision_path, "vision.cpp"),
    ],
    extra_cflags=["-O3"],
    extra_include_paths=[vision_path],
    verbose=True,
)

print("✅ Extension built successfully.")

