import os
import sys
import gc
import json
import time
import shutil
import argparse
import numpy as np
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

WPFORMER_DIR = "/home/skye/data/Skye/WPFormer"
VISUALAD_DIR = "/home/skye/data/Skye/visualad"
if WPFORMER_DIR not in sys.path:
    sys.path.insert(0, WPFORMER_DIR)
if VISUALAD_DIR not in sys.path:
    sys.path.append(VISUALAD_DIR)

from eval_bridge_metrics_common import (
    BridgeAnomalyDataset,
    DEFECT_CLASSES,
    eval_metrics_from_amap_and_sp,
    norm01_f32_list,
    setup_seed_common,
    topk_pool_np,
)


def get_wpformer_transform(image_size=512):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def compute_wpformer_loss(p_logit, gts):
    bce = F.binary_cross_entropy_with_logits(p_logit, gts, reduction="mean")
    pred = torch.sigmoid(p_logit)
    inter = (pred * gts).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + gts.sum(dim=(1, 2, 3))
    dice = (1.0 - (2.0 * inter + 1e-6) / (union + 1e-6)).mean()
    return bce + dice


@torch.no_grad()
def evaluate_wpformer(model, loader, device, eval_size=518, sigma=1.5, compute_pro=True):
    model.eval()
    gt_sp_list, masks_list, amaps_list = [], [], []
    class_masks_dict = {c: [] for c in DEFECT_CLASSES}

    for items in tqdm(loader, desc="Eval WPFormer", leave=False):
        imgs = items["img"].to(device)
        gt_mask = (items["img_mask"] > 0.5)
        labels = items["anomaly"]

        with torch.cuda.amp.autocast():
            preds = model(imgs)[-1]  # [B, 1, 512, 512]
        probs = torch.sigmoid(preds.float())
        probs_up = F.interpolate(probs, size=(eval_size, eval_size), mode="bilinear", align_corners=False)[:, 0]
        amap_np = probs_up.float().cpu().numpy()

        for b in range(imgs.shape[0]):
            sm = gaussian_filter(amap_np[b], sigma=sigma) if sigma > 0 else amap_np[b]
            amaps_list.append(sm.astype(np.float16))
            gt_sp_list.append(int(labels[b].item()))
            masks_list.append(gt_mask[b, 0].cpu().numpy().astype(np.uint8))
            if "class_masks" in items:
                for cname in DEFECT_CLASSES:
                    class_masks_dict[cname].append((items["class_masks"][cname][b, 0].cpu().numpy() > 0.5).astype(np.uint8))

    gt_sp = np.array(gt_sp_list)
    masks_np = np.stack(masks_list, axis=0)
    masks_list.clear()
    amaps_norm = norm01_f32_list(amaps_list)
    final_sp = topk_pool_np(amaps_norm, ratio=0.01)
    gc.collect()

    metrics = eval_metrics_from_amap_and_sp(gt_sp, final_sp, masks_np, amaps_norm, class_masks_dict, compute_pro=compute_pro)
    del masks_np, amaps_norm
    gc.collect()
    return metrics


def train_and_eval_wpformer(dataset_name, short_ds, epochs=60, batch_size=6, lr=8e-5, seed=111, smoke_test=False):
    setup_seed_common(seed)
    os.chdir(WPFORMER_DIR)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    out_root = os.path.join(VISUALAD_DIR, "outputs_smoke" if smoke_test else "outputs")
    save_dir = os.path.join(out_root, f"comp_wpformer_{short_ds}")
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, "train_eval.log")

    def log_print(msg):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as lf:
            lf.write(line + "\n")

    from model.WPFormer import WPFormer
    model = WPFormer(method="pvt_v2_b2", channel=64).to(device)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_print(f"=== WPFormer ({dataset_name}, Official {epochs} Ep) | Trainable params: {trainable_params} ===")

    train_root = f"/home/skye/data/Skye/databases/{dataset_name}"
    train_meta = f"/home/skye/data/Skye/visualad/datasets/{dataset_name}_meta.json"
    preprocess = get_wpformer_transform(512)

    train_set = BridgeAnomalyDataset(train_root, "train", 518, preprocess, False, train_meta)
    val_set = BridgeAnomalyDataset(train_root, "val", 518, preprocess, False, train_meta)
    test_set = BridgeAnomalyDataset(train_root, "test", 518, preprocess, True, train_meta)

    if smoke_test:
        train_set.data_all = train_set.data_all[:8]
        val_set.data_all = val_set.data_all[:8]
        test_set.data_all = test_set.data_all[:8]

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)
    scaler = torch.cuda.amp.GradScaler()

    best_val_score = -1.0
    best_ckpt = os.path.join(save_dir, "best_model.pth")
    best_ckpt_8ep = os.path.join(save_dir, "best_model_8ep.pth")
    history = []
    weights = [0.5, 0.7, 1.0, 1.0]

    for epoch in range(epochs):
        model.train()
        loss_list = []

        for items in tqdm(train_loader, desc=f"WPFormer-{short_ds} Ep {epoch+1}/{epochs}", leave=False):
            imgs = items["img"].to(device, non_blocking=True)
            gts518 = (items["img_mask"].to(device, non_blocking=True) > 0.5).float()
            gts512 = F.interpolate(gts518, size=(512, 512), mode="nearest")

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                preds = model(imgs)
                loss = 0.0
                for i, pred in enumerate(preds[:4]):
                    loss = loss + weights[i] * compute_wpformer_loss(pred, gts512)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_list.append(float(loss.detach().item()))

        scheduler.step()

        do_val = (epoch + 1 in [4, 6, 8, 12, 16, 20, 25, 30, 35, 40, 45, 50, 54, 57, 60]) or (epoch + 1 == epochs)
        if do_val:
            val_m = evaluate_wpformer(model, val_loader, device, compute_pro=False)
            val_key = (val_m["Image-AUROC"] + val_m["Pixel-AUROC"] + val_m["Pixel-F1"] + val_m["Pixel-AP"]) / 4.0
            log_print(f"Epoch [{epoch+1}/{epochs}] loss={np.mean(loss_list):.4f} | Val I-AUROC={val_m['Image-AUROC']:.2f} P-AUROC={val_m['Pixel-AUROC']:.2f} P-F1={val_m['Pixel-F1']:.2f} P-AP={val_m['Pixel-AP']:.2f}")
            history.append({"epoch": epoch + 1, "loss": round(float(np.mean(loss_list)), 4), "val": val_m})

            if val_key >= best_val_score:
                best_val_score = val_key
                torch.save(model.state_dict(), best_ckpt)

        if (epoch + 1) == 8 or (smoke_test and (epoch + 1) == epochs):
            shutil.copyfile(best_ckpt, best_ckpt_8ep)

    other_ds_name = "Bridge512_balance_CODEBRIM" if dataset_name == "Bridge512_balance_S2DS" else "Bridge512_balance_S2DS"
    other_root = f"/home/skye/data/Skye/databases/{other_ds_name}"
    other_meta = f"/home/skye/data/Skye/visualad/datasets/{other_ds_name}_meta.json"
    other_test_set = BridgeAnomalyDataset(other_root, "test", 518, preprocess, True, other_meta)
    if smoke_test:
        other_test_set.data_all = other_test_set.data_all[:8]
    other_test_loader = DataLoader(other_test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # Evaluate 8-epoch checkpoint
    in_m_8ep, cross_m_8ep = None, None
    if os.path.exists(best_ckpt_8ep):
        model.load_state_dict(torch.load(best_ckpt_8ep, map_location=device))
        in_m_8ep = evaluate_wpformer(model, test_loader, device, compute_pro=True)
        cross_m_8ep = evaluate_wpformer(model, other_test_loader, device, compute_pro=True)
        log_print(f"[WPFormer (8Ep) In-Domain {dataset_name}] {json.dumps(in_m_8ep)}")
        log_print(f"[WPFormer (8Ep) Cross-Domain {dataset_name} -> {other_ds_name}] {json.dumps(cross_m_8ep)}")

    # Evaluate full official checkpoint
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    in_m = evaluate_wpformer(model, test_loader, device, compute_pro=True)
    log_print(f"[WPFormer (Official {epochs}Ep) In-Domain {dataset_name}] {json.dumps(in_m)}")
    cross_m = evaluate_wpformer(model, other_test_loader, device, compute_pro=True)
    log_print(f"[WPFormer (Official {epochs}Ep) Cross-Domain {dataset_name} -> {other_ds_name}] {json.dumps(cross_m)}")

    summary = {
        "method": "WPFormer (CVPR 2025)",
        "variant": "comp_wpformer",
        "train_dataset": dataset_name,
        "epochs": epochs,
        "trainable_params": trainable_params,
        "in_domain_test": in_m,
        "cross_domain_test_dataset": other_ds_name,
        "cross_domain_test": cross_m,
        "in_domain_test_8ep": in_m_8ep,
        "cross_domain_test_8ep": cross_m_8ep,
        "history": history,
    }
    with open(os.path.join(save_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()
    ep = 1 if args.smoke_test else args.epochs
    for s_ds, full_ds in [("S2DS", "Bridge512_balance_S2DS"), ("CODEBRIM", "Bridge512_balance_CODEBRIM")]:
        train_and_eval_wpformer(full_ds, s_ds, epochs=ep, smoke_test=args.smoke_test)
