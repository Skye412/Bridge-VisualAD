import os
import sys
import gc
import json
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

DINOMALY_DIR = "/home/skye/data/Skye/Dinomaly"
VISUALAD_DIR = "/home/skye/data/Skye/visualad"
if DINOMALY_DIR not in sys.path:
    sys.path.insert(0, DINOMALY_DIR)
if VISUALAD_DIR not in sys.path:
    sys.path.append(VISUALAD_DIR)

from dinomaly_bridge import build_model
from utils import cal_anomaly_maps
from eval_bridge_metrics_common import (
    BridgeAnomalyDataset,
    DEFECT_CLASSES,
    eval_metrics_from_amap_and_sp,
    norm01_f32_list as norm01_f32,
)
from PIL import Image

def get_dinomaly_preprocess(image_size=392):
    return transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


@torch.no_grad()
def evaluate_dinomaly_on_loader(model, loader, device, eval_size=518, top_fraction=0.01):
    model.eval()
    gt_sp_list, pr_sp_list, masks_list, amaps_list = [], [], [], []
    class_masks_dict = {c: [] for c in DEFECT_CLASSES}

    for items in tqdm(loader, desc="Eval Dinomaly", leave=False):
        img = items["img"].to(device)
        gt_mask = (items["img_mask"] > 0.5)
        labels = items["anomaly"]

        en, de = model(img)
        amap, _ = cal_anomaly_maps(en, de, eval_size)  # [B, 1, 518, 518]
        amap_np = amap[:, 0].float().cpu().numpy()

        for b in range(img.shape[0]):
            smooth_m = gaussian_filter(amap_np[b], sigma=4.0)
            flat_m = smooth_m.reshape(-1)
            k_top = max(1, int(np.ceil(flat_m.size * top_fraction)))
            sp_val = float(np.sort(flat_m)[-k_top:].mean())

            amaps_list.append(smooth_m.astype(np.float16))
            pr_sp_list.append(sp_val)
            gt_sp_list.append(int(labels[b].item()))
            masks_list.append(gt_mask[b, 0].cpu().numpy().astype(np.uint8))
            if "class_masks" in items:
                for cname in DEFECT_CLASSES:
                    class_masks_dict[cname].append((items["class_masks"][cname][b, 0].cpu().numpy() > 0.5).astype(np.uint8))

    gt_sp = np.array(gt_sp_list)
    final_sp = np.array(pr_sp_list, dtype=np.float32)
    masks_np = np.stack(masks_list, axis=0)
    masks_list.clear()
    amaps_norm = norm01_f32(amaps_list)
    gc.collect()

    metrics = eval_metrics_from_amap_and_sp(gt_sp, final_sp, masks_np, amaps_norm, class_masks_dict, compute_pro=True)
    del masks_np, amaps_norm
    gc.collect()
    return metrics


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    out_root = os.path.join(VISUALAD_DIR, "outputs")
    preprocess_392 = get_dinomaly_preprocess(392)

    s2ds_test = BridgeAnomalyDataset(
        "/home/skye/data/Skye/databases/Bridge512_balance_S2DS", "test", 518,
        preprocess_392, True, "/home/skye/data/Skye/visualad/datasets/Bridge512_balance_S2DS_meta.json"
    )
    cb_test = BridgeAnomalyDataset(
        "/home/skye/data/Skye/databases/Bridge512_balance_CODEBRIM", "test", 518,
        preprocess_392, True, "/home/skye/data/Skye/visualad/datasets/Bridge512_balance_CODEBRIM_meta.json"
    )

    s2ds_loader = torch.utils.data.DataLoader(s2ds_test, batch_size=8, shuffle=False, num_workers=2)
    cb_loader = torch.utils.data.DataLoader(cb_test, batch_size=8, shuffle=False, num_workers=2)

    model, trainable = build_model(device, variant="v0")
    trainable_params = sum(p.numel() for p in trainable.parameters())

    configs = [
        (
            "S2DS",
            "Bridge512_balance_S2DS",
            "/home/skye/data/Skye/Dinomaly/runs/s2ds512-v0-b16-s1-review20261018/best_val.pth",
            s2ds_loader,
            "Bridge512_balance_CODEBRIM",
            cb_loader,
        ),
        (
            "CODEBRIM",
            "Bridge512_balance_CODEBRIM",
            "/home/skye/data/Skye/Dinomaly/runs/codebrim512-v0-b16-s1-review20261018/best_val.pth",
            cb_loader,
            "Bridge512_balance_S2DS",
            s2ds_loader,
        ),
    ]

    for short_ds, ds_name, ckpt_path, in_loader, cross_ds_name, cross_loader in configs:
        save_dir = os.path.join(out_root, f"comp_dinomaly_{short_ds}")
        os.makedirs(save_dir, exist_ok=True)
        print(f"\n=== Evaluating Dinomaly ({short_ds}) from {ckpt_path} ===")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        del ckpt
        torch.cuda.empty_cache()

        in_m = evaluate_dinomaly_on_loader(model, in_loader, device)
        print(f"[Dinomaly In-Domain {ds_name}]:", json.dumps(in_m))

        cross_m = evaluate_dinomaly_on_loader(model, cross_loader, device)
        print(f"[Dinomaly Cross-Domain {ds_name} -> {cross_ds_name}]:", json.dumps(cross_m))

        summary = {
            "method": "Dinomaly (CVPR 2025)",
            "variant": "comp_dinomaly",
            "train_dataset": ds_name,
            "trainable_params": trainable_params,
            "checkpoint_path": ckpt_path,
            "in_domain_test": in_m,
            "cross_domain_test_dataset": cross_ds_name,
            "cross_domain_test": cross_m,
        }
        with open(os.path.join(save_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
