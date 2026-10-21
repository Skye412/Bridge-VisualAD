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

ANOMALYCLIP_DIR = "/home/skye/data/Skye/third_party/AnomalyCLIP"
VISUALAD_DIR = "/home/skye/data/Skye/visualad"
if ANOMALYCLIP_DIR not in sys.path:
    sys.path.insert(0, ANOMALYCLIP_DIR)
if VISUALAD_DIR not in sys.path:
    sys.path.append(VISUALAD_DIR)

try:
    import thop
except ImportError:
    import types
    _thop = types.ModuleType("thop")
    _thop.profile = lambda *a, **kw: (0, 0)
    sys.modules["thop"] = _thop

import AnomalyCLIP_lib
from prompt_ensemble import AnomalyCLIP_PromptLearner
from loss import FocalLoss, BinaryDiceLoss
from utils import get_transform
from eval_bridge_metrics_common import (
    BridgeAnomalyDataset,
    DEFECT_CLASSES,
    eval_metrics_from_amap_and_sp,
    norm01_f32_list,
    setup_seed_common,
)


@torch.no_grad()
def extract_anomalyclip_cache(model, dataset, device, features_list, cache_dir, prefix, batch_size=16):
    os.makedirs(cache_dir, exist_ok=True)
    n_samples = len(dataset)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    img_mmap_path = os.path.join(cache_dir, f"{prefix}_img_feats.mmap")
    patch_mmap_path = os.path.join(cache_dir, f"{prefix}_patch_feats.mmap")

    img_mmap = None
    patch_mmap = None
    gts_list, labels_list = [], []
    cursor = 0

    model.eval()
    for items in tqdm(loader, desc=f"Pre-extract CLIP ({prefix})", leave=False):
        image = items["img"].to(device, non_blocking=True)
        bsz = image.shape[0]
        with torch.cuda.amp.autocast():
            image_features, patch_features = model.encode_image(image, features_list, DPAM_layer=20)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            pf_stack = torch.stack([pf / pf.norm(dim=-1, keepdim=True) for pf in patch_features], dim=1)  # [B, 4, L, C]

        if img_mmap is None:
            d_img = image_features.shape[-1]
            n_layers, seq_l, d_p = pf_stack.shape[1], pf_stack.shape[2], pf_stack.shape[3]
            img_mmap = np.memmap(img_mmap_path, dtype=np.float16, mode="w+", shape=(n_samples, d_img))
            patch_mmap = np.memmap(patch_mmap_path, dtype=np.float16, mode="w+", shape=(n_samples, n_layers, seq_l, d_p))

        img_mmap[cursor:cursor + bsz] = image_features.half().cpu().numpy()
        patch_mmap[cursor:cursor + bsz] = pf_stack.half().cpu().numpy()
        gts_list.append((items["img_mask"].squeeze(1).numpy() > 0.5).astype(np.uint8))
        labels_list.append(items["anomaly"].numpy().astype(np.int64))
        cursor += bsz

    img_mmap.flush()
    patch_mmap.flush()
    gts_arr = np.concatenate(gts_list, axis=0)
    labels_arr = np.concatenate(labels_list, axis=0)
    return img_mmap, patch_mmap, gts_arr, labels_arr, img_mmap_path, patch_mmap_path


@torch.no_grad()
def evaluate_anomalyclip_cached(model, prompt_learner, img_mmap, patch_mmap, gts_arr, labels_arr, device, image_size=518, sigma=4.0, batch_size=16):
    model.eval()
    prompt_learner.eval()

    prompts, tokenized_prompts, compound_prompts_text = prompt_learner(cls_id=None)
    text_features = model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()
    text_features = torch.stack(torch.chunk(text_features, dim=0, chunks=2), dim=1)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    n_samples = len(labels_arr)
    pr_sp_list, amaps_list = [], []

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        image_features = torch.from_numpy(np.array(img_mmap[start:end], dtype=np.float32)).to(device)
        pf_batch = torch.from_numpy(np.array(patch_mmap[start:end], dtype=np.float32)).to(device)

        text_probs = image_features.unsqueeze(1) @ text_features.permute(0, 2, 1)
        text_probs = (text_probs / 0.07).softmax(-1)[:, 0, 1]

        anomaly_map_list = []
        for idx in range(pf_batch.shape[1]):
            patch_feature = pf_batch[:, idx]
            similarity, _ = AnomalyCLIP_lib.compute_similarity(patch_feature, text_features[0])
            similarity_map = AnomalyCLIP_lib.get_similarity_map(similarity[:, 1:, :], image_size)
            anomaly_map = (similarity_map[..., 1] + 1.0 - similarity_map[..., 0]) / 2.0
            anomaly_map_list.append(anomaly_map)

        anomaly_map = torch.stack(anomaly_map_list, dim=0).sum(dim=0)
        amap_np = anomaly_map.float().cpu().numpy()
        sp_np = text_probs.float().cpu().numpy()
        if sp_np.ndim == 0:
            sp_np = np.array([float(sp_np)])

        for b in range(end - start):
            smooth_m = gaussian_filter(amap_np[b], sigma=sigma)
            amaps_list.append(smooth_m.astype(np.float16))
            pr_sp_list.append(float(sp_np[b]))

    final_sp = np.array(pr_sp_list, dtype=np.float32)
    amaps_norm = norm01_f32_list(amaps_list)
    metrics = eval_metrics_from_amap_and_sp(labels_arr, final_sp, gts_arr, amaps_norm, None, compute_pro=False)
    del amaps_norm
    gc.collect()
    return metrics


@torch.no_grad()
def evaluate_anomalyclip(model, prompt_learner, loader, device, features_list, image_size=518, sigma=4.0, compute_pro=True):
    model.eval()
    prompt_learner.eval()

    prompts, tokenized_prompts, compound_prompts_text = prompt_learner(cls_id=None)
    text_features = model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()
    text_features = torch.stack(torch.chunk(text_features, dim=0, chunks=2), dim=1)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    gt_sp_list, pr_sp_list, masks_list, amaps_list = [], [], [], []
    class_masks_dict = {c: [] for c in DEFECT_CLASSES}

    for items in tqdm(loader, desc="Eval AnomalyCLIP", leave=False):
        image = items["img"].to(device)
        gt_mask = (items["img_mask"] > 0.5)
        labels = items["anomaly"]

        with torch.cuda.amp.autocast():
            image_features, patch_features = model.encode_image(image, features_list, DPAM_layer=20)
        image_features = image_features.float()
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        text_probs = image_features.unsqueeze(1) @ text_features.permute(0, 2, 1)
        text_probs = (text_probs / 0.07).softmax(-1)
        text_probs = text_probs[:, 0, 1]  # [B]

        anomaly_map_list = []
        for idx, patch_feature in enumerate(patch_features):
            patch_feature = patch_feature.float()
            patch_feature = patch_feature / patch_feature.norm(dim=-1, keepdim=True)
            similarity, _ = AnomalyCLIP_lib.compute_similarity(patch_feature, text_features[0])
            similarity_map = AnomalyCLIP_lib.get_similarity_map(similarity[:, 1:, :], image_size)
            anomaly_map = (similarity_map[..., 1] + 1.0 - similarity_map[..., 0]) / 2.0
            anomaly_map_list.append(anomaly_map)

        anomaly_map = torch.stack(anomaly_map_list, dim=0).sum(dim=0)  # [B, H, W]
        amap_np = anomaly_map.float().cpu().numpy()
        sp_np = text_probs.float().cpu().numpy()
        if sp_np.ndim == 0:
            sp_np = np.array([float(sp_np)])

        for b in range(image.shape[0]):
            smooth_m = gaussian_filter(amap_np[b], sigma=sigma)
            amaps_list.append(smooth_m.astype(np.float16))
            pr_sp_list.append(float(sp_np[b]))
            gt_sp_list.append(int(labels[b].item()))
            masks_list.append(gt_mask[b, 0].cpu().numpy().astype(np.uint8))
            if "class_masks" in items:
                for cname in DEFECT_CLASSES:
                    class_masks_dict[cname].append((items["class_masks"][cname][b, 0].cpu().numpy() > 0.5).astype(np.uint8))

    gt_sp = np.array(gt_sp_list)
    final_sp = np.array(pr_sp_list, dtype=np.float32)
    masks_np = np.stack(masks_list, axis=0)
    masks_list.clear()
    amaps_norm = norm01_f32_list(amaps_list)
    gc.collect()

    metrics = eval_metrics_from_amap_and_sp(gt_sp, final_sp, masks_np, amaps_norm, class_masks_dict, compute_pro=compute_pro)
    del masks_np, amaps_norm
    gc.collect()
    return metrics


def train_and_eval_anomalyclip(dataset_name, short_ds, epochs=15, batch_size=8, lr=0.001, seed=111, smoke_test=False):
    setup_seed_common(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    out_root = os.path.join(VISUALAD_DIR, "outputs_smoke" if smoke_test else "outputs")
    save_dir = os.path.join(out_root, f"comp_anomalyclip_{short_ds}")
    cache_dir = os.path.join(out_root, f"_cache_anomalyclip_{short_ds}")
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, "train_eval.log")

    def log_print(msg):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as lf:
            lf.write(line + "\n")

    image_size = 518
    features_list = [6, 12, 18, 24]
    dummy_opt = argparse.Namespace(image_size=image_size)
    preprocess, _ = get_transform(dummy_opt)

    train_root = f"/home/skye/data/Skye/databases/{dataset_name}"
    train_meta = f"/home/skye/data/Skye/visualad/datasets/{dataset_name}_meta.json"
    train_ds = BridgeAnomalyDataset(train_root, "train", image_size, preprocess, False, train_meta)
    val_ds = BridgeAnomalyDataset(train_root, "val", image_size, preprocess, False, train_meta)
    test_ds = BridgeAnomalyDataset(train_root, "test", image_size, preprocess, True, train_meta)

    if smoke_test:
        train_ds.data_all = train_ds.data_all[:8]
        val_ds.data_all = val_ds.data_all[:8]
        test_ds.data_all = test_ds.data_all[:8]

    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    AnomalyCLIP_parameters = {"Prompt_length": 12, "learnabel_text_embedding_depth": 9, "learnabel_text_embedding_length": 4}
    model, _ = AnomalyCLIP_lib.load(
        "ViT-L/14@336px",
        device=device,
        design_details=AnomalyCLIP_parameters,
        download_root="/home/skye/data/Skye/Bayes-PFL-main/pretrained_weight",
    )
    model.eval()

    prompt_learner = AnomalyCLIP_PromptLearner(model.to("cpu"), AnomalyCLIP_parameters)
    prompt_learner.to(device)
    model.to(device)
    model.visual.DAPM_replace(DPAM_layer=20)

    trainable_params = sum(p.numel() for p in prompt_learner.parameters() if p.requires_grad)
    log_print(f"=== AnomalyCLIP ({dataset_name}, Official {epochs} Ep) | Trainable params: {trainable_params} ===")

    optimizer = torch.optim.Adam(list(prompt_learner.parameters()), lr=lr, betas=(0.5, 0.999))
    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()
    lam = 4.0

    best_val_score = -1.0
    best_ckpt = os.path.join(save_dir, "best_model.pth")
    best_ckpt_8ep = os.path.join(save_dir, "best_model_8ep.pth")
    history = []

    try:
        tr_img_m, tr_pf_m, tr_gts, tr_labels, _, _ = extract_anomalyclip_cache(model, train_ds, device, features_list, cache_dir, "train")
        va_img_m, va_pf_m, va_gts, va_labels, _, _ = extract_anomalyclip_cache(model, val_ds, device, features_list, cache_dir, "val")
        n_train = len(tr_labels)

        for epoch in range(epochs):
            model.eval()
            prompt_learner.train()
            loss_list = []
            perm = np.random.permutation(n_train)

            for start in range(0, n_train, batch_size):
                batch_idx = perm[start:start + batch_size]
                image_features = torch.from_numpy(np.array(tr_img_m[batch_idx], dtype=np.float32)).to(device)
                pf_batch = torch.from_numpy(np.array(tr_pf_m[batch_idx], dtype=np.float32)).to(device)
                label = torch.from_numpy(tr_labels[batch_idx]).long().to(device)
                gt = torch.from_numpy(tr_gts[batch_idx]).float().to(device)

                prompts, tokenized_prompts, compound_prompts_text = prompt_learner(cls_id=None)
                text_features = model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()
                text_features = torch.stack(torch.chunk(text_features, dim=0, chunks=2), dim=1)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                text_probs = image_features.unsqueeze(1) @ text_features.permute(0, 2, 1)
                text_probs = text_probs[:, 0, ...] / 0.07
                if text_probs.dim() == 1:
                    text_probs = text_probs.unsqueeze(0)
                image_loss = F.cross_entropy(text_probs, label)

                similarity_map_list = []
                for idx in range(pf_batch.shape[1]):
                    patch_feature = pf_batch[:, idx]
                    similarity, _ = AnomalyCLIP_lib.compute_similarity(patch_feature, text_features[0])
                    similarity_map = AnomalyCLIP_lib.get_similarity_map(similarity[:, 1:, :], image_size).permute(0, 3, 1, 2)
                    similarity_map_list.append(similarity_map)

                seg_loss = 0.0
                for sim_m in similarity_map_list:
                    seg_loss = seg_loss + loss_focal(sim_m, gt) + loss_dice(sim_m[:, 1, :, :], gt) + loss_dice(sim_m[:, 0, :, :], 1.0 - gt)

                total_loss = lam * seg_loss + image_loss
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                loss_list.append(float(total_loss.detach().item()))

            val_m = evaluate_anomalyclip_cached(model, prompt_learner, va_img_m, va_pf_m, va_gts, va_labels, device, image_size)
            val_key = (val_m["Image-AUROC"] + val_m["Pixel-AUROC"] + val_m["Pixel-F1"] + val_m["Pixel-AP"]) / 4.0
            log_print(f"Epoch [{epoch+1}/{epochs}] loss={np.mean(loss_list):.4f} | Val I-AUROC={val_m['Image-AUROC']:.2f} P-AUROC={val_m['Pixel-AUROC']:.2f} P-F1={val_m['Pixel-F1']:.2f} P-AP={val_m['Pixel-AP']:.2f}")
            history.append({"epoch": epoch + 1, "loss": round(float(np.mean(loss_list)), 4), "val": val_m})

            if val_key >= best_val_score:
                best_val_score = val_key
                torch.save({"prompt_learner": prompt_learner.state_dict()}, best_ckpt)

            if (epoch + 1) == 8 or (smoke_test and (epoch + 1) == epochs):
                shutil.copyfile(best_ckpt, best_ckpt_8ep)
        del tr_img_m, tr_pf_m, va_img_m, va_pf_m
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)

    other_ds_name = "Bridge512_balance_CODEBRIM" if dataset_name == "Bridge512_balance_S2DS" else "Bridge512_balance_S2DS"
    other_root = f"/home/skye/data/Skye/databases/{other_ds_name}"
    other_meta = f"/home/skye/data/Skye/visualad/datasets/{other_ds_name}_meta.json"
    other_test_ds = BridgeAnomalyDataset(other_root, "test", image_size, preprocess, True, other_meta)
    if smoke_test:
        other_test_ds.data_all = other_test_ds.data_all[:8]
    other_test_loader = torch.utils.data.DataLoader(other_test_ds, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    # Evaluate 8-epoch checkpoint
    in_m_8ep, cross_m_8ep = None, None
    if os.path.exists(best_ckpt_8ep):
        ckpt8 = torch.load(best_ckpt_8ep, map_location=device)
        prompt_learner.load_state_dict(ckpt8["prompt_learner"])
        in_m_8ep = evaluate_anomalyclip(model, prompt_learner, test_loader, device, features_list, image_size, compute_pro=True)
        cross_m_8ep = evaluate_anomalyclip(model, prompt_learner, other_test_loader, device, features_list, image_size, compute_pro=True)
        log_print(f"[AnomalyCLIP (8Ep) In-Domain {dataset_name}] {json.dumps(in_m_8ep)}")
        log_print(f"[AnomalyCLIP (8Ep) Cross-Domain {dataset_name} -> {other_ds_name}] {json.dumps(cross_m_8ep)}")

    # Evaluate full official checkpoint
    ckpt = torch.load(best_ckpt, map_location=device)
    prompt_learner.load_state_dict(ckpt["prompt_learner"])
    in_m = evaluate_anomalyclip(model, prompt_learner, test_loader, device, features_list, image_size, compute_pro=True)
    log_print(f"[AnomalyCLIP (Official {epochs}Ep) In-Domain {dataset_name}] {json.dumps(in_m)}")
    cross_m = evaluate_anomalyclip(model, prompt_learner, other_test_loader, device, features_list, image_size, compute_pro=True)
    log_print(f"[AnomalyCLIP (Official {epochs}Ep) Cross-Domain {dataset_name} -> {other_ds_name}] {json.dumps(cross_m)}")

    summary = {
        "method": "AnomalyCLIP (ICLR 2024)",
        "variant": "comp_anomalyclip",
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
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()
    ep = 1 if args.smoke_test else args.epochs
    for s_ds, full_ds in [("S2DS", "Bridge512_balance_S2DS"), ("CODEBRIM", "Bridge512_balance_CODEBRIM")]:
        train_and_eval_anomalyclip(full_ds, s_ds, epochs=ep, smoke_test=args.smoke_test)
