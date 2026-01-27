import torch
import torchvision
from torchvision.transforms import v2
from PIL import Image

# REPO_DIR = <PATH/TO/A/LOCAL/DIRECTORY/WHERE/THE/DINOV3/REPO/WAS/CLONED>
REPO_DIR = "."

import torch

# DINOv3
dinov3_vit7b16_lc = torch.hub.load(
    REPO_DIR,
    "dinov3_vit7b16_lc",
    source="local",
    weights="dinov3_vit7b16_imagenet1k_linear_head-90d8ed92.pth",
    # backbone_weights="dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth",
    backbone_weights="dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth",
)


# print("done")
def get_img():
    import requests

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    return image


def make_transform(resize_size: int = 256):
    to_tensor = v2.ToImage()
    resize = v2.Resize((resize_size, resize_size), antialias=True)
    to_float = v2.ToDtype(torch.float32, scale=True)
    normalize = v2.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return v2.Compose([to_tensor, resize, to_float, normalize])


img_size = 1024
img = get_img()
transform = make_transform(img_size)
# with torch.inference_mode():
#     with torch.autocast("cuda", dtype=torch.bfloat16):
#         batch_img = transform(img)[None]
#         batch_img = batch_img
#         # depths = depther(batch_img)
#         classifier = dinov3_vit7b16_lc(batch_img)
x = transform(img).unsqueeze(0)  # batch 1
with torch.no_grad():
    logits = dinov3_vit7b16_lc(x)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    pred = torch.argmax(probs, dim=-1)

print("Predicted class:", pred.item())

# plt.figure(figsize=(12, 6))
# plt.subplot(121)
# plt.imshow(img)
# plt.axis("off")
# plt.subplot(122)
# plt.imshow(depths[0, 0].cpu(), cmap=colormaps["Spectral"])
# plt.axis("off")
