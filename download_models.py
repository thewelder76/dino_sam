import os
import requests
from huggingface_hub import HfFolder

os.makedirs("models", exist_ok=True)

# Get your token from ~/.huggingface/token
token = HfFolder.get_token()
if not token:
    raise RuntimeError("❌ No Hugging Face token found. Run `huggingface-cli login` first.")

headers = {"Authorization": f"Bearer {token}"}

def download_file(url, dest_path, headers=None):
    print(f"⬇️  Downloading {url} → {dest_path}")
    with requests.get(url, headers=headers, stream=True) as r:
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

# Replace original broken link with this:
download_file(
    url="https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth",
    dest_path="models/groundingdino_swint_ogc.pth",
    headers=headers  # optional, public mirror
)

# # Grounding DINO checkpoint
# download_file(
#     url="https://huggingface.co/IDEA-Research/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth",
#     dest_path="models/groundingdino_swint_ogc.pth",
#     headers=headers
# )

# Grounding DINO config
download_file(
    url="https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py",
    dest_path="models/GroundingDINO_SwinT_OGC.py"
)

# SAM ViT-H weights
download_file(
    url="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    dest_path="models/sam_vit_h.pth"
)

print("✅ All files downloaded into ./models/")

