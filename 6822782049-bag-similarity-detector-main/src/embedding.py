import torch
import open_clip
from PIL import Image

# Load CLIP model
model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32', pretrained='openai'
)
model.eval()

def get_embedding(image_path):
    img = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0)

    with torch.no_grad():
        embedding = model.encode_image(img)

    return embedding.squeeze().numpy()