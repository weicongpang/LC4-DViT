import os
import json
import warnings
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from models.build import build_model  

warnings.filterwarnings("ignore", category=FutureWarning)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

IM_SIZE = 512
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

TEST_ROOT = "/root/autodl-tmp/Processed_RS/test"
WEIGHTS_PATH = "/root/lvc2dvit/results/dvit_all/run_dvit_20251121_173152(best)/final_trained_dvit.pth"
CLASS_MAP_PATH = "/root/lvc2dvit/classification_to_name.json"
RESULTS_DIR = "/root/lvc2dvit/attention_heatmap_result/dvit_all"

TARGET_IMAGES = [
    ("Beach",   "beach_178.jpg"),
    ("Bridge",  "bridge_220.jpg"),
    ("Mountain","mountain_119.jpg"),
    ("Forest",  "forest_14.jpg"),
    ("Desert",  "desert_77.jpg"),
    ("Pond",    "pond_20.jpg"),
    ("River",   "river_84.jpg"),
    ("Port",    "port_16.jpg"),
]

# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

def load_class_map():
    if not os.path.exists(CLASS_MAP_PATH):
        print(f"[WARN] class map not found: {CLASS_MAP_PATH}")
        return {}
    with open(CLASS_MAP_PATH, "r") as f:
        return json.load(f)

transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.Resize((IM_SIZE, IM_SIZE), InterpolationMode.BICUBIC, antialias=True),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

def load_model():
    model = build_model().to(DEVICE)
    model.eval()
    ckpt = torch.load(WEIGHTS_PATH, map_location=DEVICE)
    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict, strict=True)
    return model


@torch.no_grad()
def dvit_generate_cam_and_cosine(model, img_tensor):
    """
    DO NOT MODIFY COSINE OR CAM IMPLEMENTATION
    """
    model.eval()

    feat = model.backbone(img_tensor)  # (1, C, Hf, Wf)
    B, C, Hf, Wf = feat.shape

    x = feat.permute(0, 2, 3, 1).reshape(B, Hf * Wf, C)
    x = model.patch_norm(x)
    x = model.patch_proj(x)            # (1, N, D)
    N = Hf * Wf
    assert x.size(1) == N

    cls_token = model.cls_token.expand(B, -1, -1)
    x = torch.cat([cls_token, x], dim=1)
    x = x + model.pos_embedding[:, : N + 1]
    x = model.pos_dropout(x)

    x = model.transformer(x)
    cls_out = x[:, 0]
    patch_tokens = x[:, 1:]

    patch_mean = patch_tokens.mean(dim=1)
    hybrid_vec = torch.cat([cls_out, patch_mean], dim=-1)
    latent_vec = model.to_latent(hybrid_vec)

    logits = model.mlp_head(latent_vec)
    pred = int(logits.argmax(dim=1).item())

    weight = model.mlp_head[-1].weight
    cam_w = weight[pred]

    cam = torch.matmul(patch_tokens.squeeze(0), cam_w)
    cam = F.relu(cam)
    cam = cam.view(Hf, Wf)
    cam = cam / (cam.max() + 1e-8)

    cls_norm = F.normalize(cls_out, dim=-1)
    patch_norm = F.normalize(patch_tokens, dim=-1)
    cos_map = (patch_norm * cls_norm.unsqueeze(1)).sum(-1).squeeze(0)

    cos_map = (cos_map - cos_map.min()) / (cos_map.max() - cos_map.min() + 1e-8)
    cos_map = cos_map.view(Hf, Wf)

    return {
        "pred": pred,
        "cosine_map": cos_map.cpu().numpy(),
        "mean_cosine": float(cos_map.mean().item())
    }


@torch.no_grad()
def mobilenetv2_cam_and_cos(model, img_tensor):
    """
    Extract feature map → CAM → cosine similarity map.
    Works for MobileNetV2 (CNN).
    """

    # 1. Extract the CNN feature map (last conv layer)
    feat = model.features(img_tensor)    # (B, C, H, W)
    feat = model.conv(feat)              # last 1x1 conv
    B, C, H, W = feat.shape

    # 2. Global vector = GAP + classifier input
    gap = F.adaptive_avg_pool2d(feat, (1, 1)).view(B, C)  # (B, C)

    # 3. Flatten feature map into patch tokens: (H*W, C)
    patches = feat.view(B, C, H*W).permute(0, 2, 1).squeeze(0)  # (N, C)

    # 4. L2 normalize
    global_vec = F.normalize(gap, dim=-1)              # (C)
    patches_norm = F.normalize(patches, dim=-1)        # (N, C)

    # 5. Cosine similarity per patch
    cos = torch.matmul(patches_norm, global_vec.squeeze(0))     # (N,)
    cos = cos.view(H, W)
    cos = (cos - cos.min()) / (cos.max() - cos.min() + 1e-8)
    cos_np = cos.cpu().numpy()

    # 6. CAM using fc layer
    fc_weight = model.classifier.weight                 # (num_classes, C)
    preds = model.classifier(gap)
    pred_idx = preds.argmax(-1).item()

    cam_w = fc_weight[pred_idx]                         # (C,)
    cam_map = torch.matmul(patches, cam_w)              # (N,)
    cam_map = cam_map.view(H, W)
    cam_map = F.relu(cam_map)
    cam_map = cam_map / (cam_map.max() + 1e-8)
    cam_np = cam_map.cpu().numpy()

    return {
        "pred": pred_idx,
        "cosine_map": cos_np,
        "cam_map": cam_np,
        "mean_cosine": float(cos_np.mean())
    }

# For FlashInternImage
@torch.no_grad()
def flashintern_generate_cam_and_cosine(model, img_tensor):
    model.eval()

    # --------------------------
    # 1. Forward to last feature map
    # --------------------------
    #   model.forward_features() returns pooled global features
    #   we need feature map BEFORE avgpool → so we reconstruct it
    # --------------------------
    #   forward_features:
    #       x = patch_embed
    #       x -> levels[] (DCNv4 layers)
    #       x -> conv_head (1x1 conv)
    #       x -> avgpool (global)
    # --------------------------

    # step 1: patch embed
    x = model.patch_embed(img_tensor)           # (B, H/4, W/4, C)
    B, H, W, C = x.shape
    x = x.view(B, H * W, C)
    shape = (H, W)

    # step 2: all DCNv4 stages
    for level in model.levels:
        x, shape = level(x, shape=shape)

    Hf, Wf = shape
    C_final = x.shape[-1]

    # reshape to feature map
    feat_map = x.view(B, Hf, Wf, C_final).permute(0, 3, 1, 2).contiguous()
    # feat_map: (1, C, Hf, Wf)

    # --------------------------
    # 2. conv_head (1x1 conv + BN + act)
    # --------------------------
    feat_map = model.conv_head(feat_map)        # (1, C2, Hf, Wf)
    B, C2, Hf, Wf = feat_map.shape

    # --------------------------
    # 3. Global pooled feature → classification head
    # --------------------------
    pooled = model.avgpool(feat_map)            # (1, C2, 1, 1)
    pooled = pooled.view(B, C2)                 # (1, C2)

    logits = model.head(pooled)                 # (1, num_classes)
    pred = int(logits.argmax(dim=1).item())

    # --------------------------
    # 4. CAM: w_pred ⋅ feature_map
    # --------------------------
    cls_w = model.head.weight[pred].view(C2, 1, 1)    # (C2,1,1)
    cam = (feat_map * cls_w).sum(dim=1)               # (1, Hf, Wf)
    cam = F.relu(cam)
    cam = cam.squeeze(0)
    cam = cam / (cam.max() + 1e-8)
    cam_np = cam.cpu().numpy()

    # --------------------------
    # 5. Cosine similarity map
    # --------------------------
    pooled_norm = F.normalize(pooled, dim=-1)                   # (1, C2)
    fmap_flat = feat_map.view(1, C2, Hf * Wf).permute(0, 2, 1)  # (1, Hf*Wf, C2)
    fmap_norm = F.normalize(fmap_flat, dim=-1)                  # (1,Hf*Wf,C2)

    cos = (fmap_norm * pooled_norm.unsqueeze(1)).sum(-1)        # (1,Hf*Wf)
    cos = cos.view(Hf, Wf)
    cos = (cos - cos.min()) / (cos.max() - cos.min() + 1e-8)
    cos_np = cos.cpu().numpy()

    return {
        "pred": pred,
        "logits": logits.squeeze(0).cpu(),
        "cam_map": cam_np,
        "cosine_map": cos_np,
        "mean_cosine": float(cos_np.mean())
    }


@torch.no_grad()
def resnet50_generate_cam_and_cosine(model, img_tensor):
    """
    Extract feature map from ResNet50 (layer4) → CAM → cosine similarity map.
    适用于你给出的 ResNet 实现（包含 conv1/bn1/relu/maxpool、layer1~4、avgpool、fc）。
    """
    model.eval()

    # 1. 前向到最后一层卷积特征图 (layer4)
    x = model.conv1(img_tensor)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)

    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    feat = model.layer4(x)          # (B, C, H, W)

    B, C, H, W = feat.shape

    # 2. 全局特征向量（GAP 后接 fc）
    gap = model.avgpool(feat)       # (B, C, 1, 1)
    gap = torch.flatten(gap, 1)     # (B, C)

    logits = model.fc(gap)          # (B, num_classes)
    pred_idx = int(logits.argmax(dim=1).item())

    # 3. 将特征图摊平成 patch tokens: (N, C)
    patches = feat.view(B, C, H * W).permute(0, 2, 1).squeeze(0)   # (N, C)

    # 4. 计算每个 patch 与全局向量的余弦相似度
    global_vec = F.normalize(gap, dim=-1).squeeze(0)      # (C,)
    patches_norm = F.normalize(patches, dim=-1)           # (N, C)

    cos = torch.matmul(patches_norm, global_vec)          # (N,)
    cos = cos.view(H, W)
    cos = (cos - cos.min()) / (cos.max() - cos.min() + 1e-8)
    cos_np = cos.cpu().numpy()

    # 5. 用 fc 权重做 CAM
    fc_weight = model.fc.weight                           # (num_classes, C)
    cam_w = fc_weight[pred_idx]                           # (C,)
    cam_map = torch.matmul(patches, cam_w)                # (N,)
    cam_map = cam_map.view(H, W)
    cam_map = F.relu(cam_map)
    cam_map = cam_map / (cam_map.max() + 1e-8)
    cam_np = cam_map.cpu().numpy()

    # 6. 返回与原框架兼容的结果
    return {
        "pred": pred_idx,
        "cosine_map": cos_np,          # 用于你现在的 overlay
        "cam_map": cam_np,             # 如果后面想画 CAM 也可以用
        "mean_cosine": float(cos_np.mean())
    }


@torch.no_grad()
def vit_generate_cam_and_cosine(model, img_tensor):
    """
    适用于你给出的 VisionTransformer 实现：
      - 使用 CLS token 做分类
      - head 为 Linear(embed_dim -> num_classes)

    返回：
      - pred: 预测类别下标（int）
      - cosine_map: (Hf, Wf) 的余弦相似度热力图 (numpy)
      - cam_map: (Hf, Wf) 的 CAM 热力图 (numpy)
      - mean_cosine: 平均余弦相似度 (float)
    """
    model.eval()
    B = img_tensor.shape[0]
    assert B == 1, "当前实现假定 batch_size=1，方便生成单张图的 heatmap"

    # -----------------------------
    # 1. 手动跑一遍 ViT 的前向（拿到 CLS 与 patch tokens）
    # -----------------------------
    # patch embedding: (B, N, C)
    x = model.patch_embed(img_tensor)
    B, N, C = x.shape

    # CLS token 拼接
    cls_tokens = model.cls_token.expand(B, 1, C)        # (B, 1, C)
    x = torch.cat((cls_tokens, x), dim=1)               # (B, N+1, C)

    # 位置编码
    # pos_embed 形状是 (1, num_patches+1, C)，这里做一次切片更稳妥
    x = x + model.pos_embed[:, : N + 1, :]
    x = model.pos_drop(x)

    # Transformer blocks
    for blk in model.blocks:
        x = blk(x)

    # LayerNorm
    x = model.norm(x)

    # CLS 向量 & patch 向量
    cls_out = x[:, 0]        # (B, C)
    patch_tokens = x[:, 1:]  # (B, N, C)

    # -----------------------------
    # 2. 分类预测（基于 CLS）
    # -----------------------------
    logits = model.head(cls_out)              # (B, num_classes)
    pred_idx = int(logits.argmax(dim=1).item())

    # -----------------------------
    # 3. 计算 CAM：patch_tokens @ fc_weight[pred]
    # -----------------------------
    # head: Linear(C -> num_classes)
    fc_weight = model.head.weight             # (num_classes, C)
    cam_w = fc_weight[pred_idx]               # (C,)

    # (N, C) @ (C,) -> (N,)
    patches_flat = patch_tokens.squeeze(0)    # (N, C)
    cam = torch.matmul(patches_flat, cam_w)   # (N,)
    cam = F.relu(cam)

    # reshape 到特征图大小 (Hf, Wf)
    img_size = model.patch_embed.img_size
    patch_size = model.patch_embed.patch_size
    Hf = img_size // patch_size
    Wf = img_size // patch_size

    cam = cam.view(Hf, Wf)
    cam = cam / (cam.max() + 1e-8)
    cam_np = cam.cpu().numpy()

    # -----------------------------
    # 4. 计算 CLS 与每个 patch 的余弦相似度
    # -----------------------------
    cls_norm = F.normalize(cls_out, dim=-1)           # (B, C)
    patch_norm = F.normalize(patch_tokens, dim=-1)    # (B, N, C)

    # (B, N, C) · (B, 1, C) -> (B, N)
    cos_map = (patch_norm * cls_norm.unsqueeze(1)).sum(-1).squeeze(0)  # (N,)

    # 归一化到 [0, 1]
    cos_map = (cos_map - cos_map.min()) / (cos_map.max() - cos_map.min() + 1e-8)
    cos_map = cos_map.view(Hf, Wf)
    cos_np = cos_map.cpu().numpy()

    mean_cos = float(cos_map.mean().item())

    return {
        "pred": pred_idx,
        "cosine_map": cos_np,
        "cam_map": cam_np,
        "mean_cosine": mean_cos
    }



def overlay_heatmap_on_image(img_np, heatmap_2d, alpha=0.5):
    H, W, _ = img_np.shape
    heat = cv2.resize(heatmap_2d, (W, H))
    heat_uint8 = np.uint8(255 * heat)
    heat_uint8 = np.ascontiguousarray(heat_uint8)

    color_map = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
    color_map = cv2.cvtColor(color_map, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img_np, 1 - alpha, color_map, alpha, 0)
    return overlay


def save_cosine_overlay(img_path, cosine_map, save_path, alpha=0.5):
    """
    Save cosine overlay image without any border, padding, or text.
    """
    img = Image.open(img_path).convert("RGB")
    img = img.resize((IM_SIZE, IM_SIZE))
    img_np = np.array(img)

    cos_overlay = overlay_heatmap_on_image(img_np, cosine_map, alpha=alpha)

    plt.figure(figsize=(5, 5), dpi=300)
    plt.imshow(cos_overlay)
    plt.axis("off")

    # Remove all margins and white borders:
    plt.gca().set_position([0, 0, 1, 1])
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.margins(0)
    plt.gcf().patch.set_facecolor('black')
    plt.gcf().patch.set_alpha(0)

    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()



# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    class_map = load_class_map()
    idx_to_name = {int(k): v for k, v in class_map.items()}

    model = load_model()

    cosine_records = []

    print("\n================ Cosine Similarity Results ================\n")

    for cls_name, fname in TARGET_IMAGES:
        img_path = os.path.join(TEST_ROOT, cls_name, fname)
        if not os.path.exists(img_path):
            print(f"[WARN] Image missing: {img_path}")
            continue

        img_raw = Image.open(img_path).convert("RGB")
        img_tensor = transform(img_raw).unsqueeze(0).to(DEVICE)

        # result = generate_cam_and_cosine(model, img_tensor)
        result = dvit_generate_cam_and_cosine(model, img_tensor)
        cos_value = result["mean_cosine"]
        cosine_map = result["cosine_map"]

        cosine_records.append((cls_name, fname, cos_value))

        print(f"{cls_name}/{fname}  -->  cosine = {cos_value:.6f}")

        save_name = f"{cls_name}_{os.path.splitext(fname)[0]}.png"
        save_path = os.path.join(RESULTS_DIR, save_name)

        save_cosine_overlay(img_path, cosine_map, save_path)

    # --------------------------------------------------------
    # Save mean cosine to txt
    # --------------------------------------------------------
    txt_path = os.path.join(RESULTS_DIR, "cosine_similarity_results.txt")

    with open(txt_path, "w") as f:
        f.write("Per-image cosine similarity:\n")
        for cls_name, fname, cosv in cosine_records:
            f.write(f"{cls_name}/{fname}: {cosv:.6f}\n")

        if cosine_records:
            all_cos = [c[2] for c in cosine_records]
            mean_cos = float(np.mean(all_cos))
            f.write("\nMean cosine similarity (all images): {:.6f}\n".format(mean_cos))
        else:
            f.write("\nNo cosine values computed.\n")

    print("\nSaved cosine results to:", txt_path)
    print("Saved 8 cosine heatmaps to:", RESULTS_DIR)


if __name__ == "__main__":
    main()
