#!/usr/bin/env python3
"""
Zero-Shot Cross-Dataset & Extended-Damage Generalization on DACL10k-v3 (validation, 975 images)
for the 2 selected external comparison methods:
  1. AdaptCLIP (AAAI 2026) - using existing S2DS & CODEBRIM checkpoints (best_model.pth)
  2. WPFormer (CVPR 2025)  - using existing S2DS & CODEBRIM checkpoints (best_model.pth)
Uses the exact same polygon rasterization, label taxonomy, evaluation metrics, and source-domain fixed threshold rules
as eval_dacl10k_zero_shot.py.
"""
import os
import sys
import glob
import json
import time
import argparse
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.ndimage import gaussian_filter
import torch
import torch.nn.functional as F
import torch.utils.data as data
from tqdm import tqdm

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
if CUR_DIR not in sys.path:
    sys.path.insert(0, CUR_DIR)

from eval_bridge_metrics_common import (
    norm01_np as norm01,
    topk_pool_np,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    auc,
)

DACL_ROOT = "/data/Skye/databases/dacl10k_v3_512/extracted"
DACL_VAL_IMG_DIR = os.path.join(DACL_ROOT, "images/valid")
DACL_VAL_ANN_DIR = os.path.join(DACL_ROOT, "dacl10k_v3_devphase_annotations_512/validation")

SHARED_4_MAP = {
    "Crack": "Crack",
    "Spalling": "Spalling",
    "Corrosion": "Rust",
    "Efflorescence": "Efflorescence",
}
SHARED_4_LABELS = ["Crack", "Spalling", "Rust", "Efflorescence"]
EXTENDED_DAMAGE_LABELS = [
    "ACrack", "Wetspot", "Cavity", "Hollowareas", "Rockpocket",
    "WConccor", "Weathering", "ExposedRebars", "Graffiti",
]
ALL_13_DAMAGE_LABELS = SHARED_4_LABELS + EXTENDED_DAMAGE_LABELS
STRUCTURAL_12_DAMAGE_LABELS = [l for l in ALL_13_DAMAGE_LABELS if l != "Graffiti"]
OBJECT_6_LABELS = ["Bearing", "EJoint", "Drainage", "PEquipment", "JTape", "Restformwork"]


def rasterize_polygons(shapes, target_labels_set, width=512, height=512, out_size=518):
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    has_any = False
    for sh in shapes:
        if sh.get("label") in target_labels_set:
            pts = [(float(x), float(y)) for x, y in sh.get("points", [])]
            if len(pts) >= 3:
                draw.polygon(pts, outline=1, fill=1)
                has_any = True
            elif len(pts) == 2:
                draw.line(pts, fill=1, width=2)
                has_any = True
    if not has_any:
        return np.zeros((out_size, out_size), dtype=np.uint8)
    if (width, height) != (out_size, out_size):
        mask = mask.resize((out_size, out_size), resample=Image.NEAREST)
    return (np.array(mask, dtype=np.uint8) > 0).astype(np.uint8)


class DACL10kCustomValDataset(data.Dataset):
    def __init__(self, preprocess):
        super().__init__()
        self.preprocess = preprocess
        self.json_paths = sorted(glob.glob(os.path.join(DACL_VAL_ANN_DIR, "*.json")))
        assert len(self.json_paths) == 975, f"Expected 975 valid JSONs, got {len(self.json_paths)}"

    def __len__(self):
        return len(self.json_paths)

    def __getitem__(self, idx):
        jp = self.json_paths[idx]
        with open(jp, "r", encoding="utf-8") as f:
            ann = json.load(f)
        img_path = os.path.join(DACL_VAL_IMG_DIR, ann["imageName"])
        img = Image.open(img_path).convert("RGB")
        img_tensor = self.preprocess(img)
        return {
            "img": img_tensor,
            "idx": idx,
            "img_name": ann["imageName"],
        }


def precompute_cc_slices(masks):
    cc_list = []
    for i in range(len(masks)):
        if masks[i].sum() == 0:
            continue
        labeled, _ = ndimage.label(masks[i] > 0)
        objs = ndimage.find_objects(labeled)
        for comp_id, slc in enumerate(objs, start=1):
            if slc is None:
                continue
            sub_mask = (labeled[slc] == comp_id)
            area = float(sub_mask.sum())
            if area > 0:
                cc_list.append((i, slc, sub_mask, area))
    return cc_list


def fast_aupro_precomputed(masks, amaps, cc_list, ignore_masks=None, num_th=40, max_fpr=0.3):
    if not cc_list:
        return 0.0, 0.0
    min_th, max_th = float(amaps.min()), float(amaps.max())
    if max_th <= min_th + 1e-12:
        return 0.0, 0.0

    thresholds = np.linspace(min_th, max_th, num_th, endpoint=False, dtype=np.float32)
    bins = np.concatenate([[-np.inf], np.nextafter(thresholds, np.inf), [np.inf]])

    comp_pros_list = []
    for (img_idx, slc, sub_mask, area) in cc_list:
        comp_vals = amaps[img_idx][slc][sub_mask]
        c_hist, _ = np.histogram(comp_vals, bins=bins)
        c_tp = np.cumsum(c_hist[::-1])[::-1][1:].astype(np.float64)
        comp_pros_list.append(c_tp / area)

    if ignore_masks is not None:
        neg_cond = (masks == 0) & (ignore_masks == 0)
    else:
        neg_cond = (masks == 0)
    neg_vals = amaps[neg_cond]
    total_neg = float(len(neg_vals))
    if total_neg == 0:
        return 0.0, 0.0

    neg_hist, _ = np.histogram(neg_vals, bins=bins)
    fp_arr = np.cumsum(neg_hist[::-1])[::-1][1:].astype(np.float64)
    fpr_arr = fp_arr / total_neg
    mean_pros = np.mean(np.stack(comp_pros_list, axis=0), axis=0)

    valid = (fpr_arr <= max_fpr)
    fprs = fpr_arr[valid]
    pros = mean_pros[valid]
    if len(fprs) < 2:
        return 0.0, 0.0

    order = np.argsort(fprs)
    fprs = fprs[order]
    pros = pros[order]
    fpr_span = float(fprs[-1] - fprs[0])
    aupro_repo = float(auc(fprs, pros) / fpr_span) if fpr_span > 1e-12 else 0.0

    fprs_ext = np.concatenate([[0.0], fprs, [max_fpr]])
    pros_ext = np.concatenate([[0.0], pros, [pros[-1]]])
    aupro_std03 = float(auc(fprs_ext, pros_ext) / max_fpr)
    return aupro_repo, aupro_std03


def precompute_dacl_masks(out_size=518):
    print("Pre-rasterizing 975 DACL10k-v3 validation JSON annotations...")
    t0 = time.time()
    json_paths = sorted(glob.glob(os.path.join(DACL_VAL_ANN_DIR, "*.json")))
    N = len(json_paths)

    masks_all_13 = np.zeros((N, out_size, out_size), dtype=np.uint8)
    masks_struct_12 = np.zeros((N, out_size, out_size), dtype=np.uint8)
    masks_shared_4 = np.zeros((N, out_size, out_size), dtype=np.uint8)
    masks_other_9 = np.zeros((N, out_size, out_size), dtype=np.uint8)

    tracked_indiv_labels = SHARED_4_LABELS + [
        "ACrack", "ExposedRebars", "Cavity", "Wetspot", "Weathering", "Hollowareas", "Rockpocket", "WConccor"
    ]
    indiv_masks = {lbl: np.zeros((N, out_size, out_size), dtype=np.uint8) for lbl in tracked_indiv_labels}

    set_all_13 = set(ALL_13_DAMAGE_LABELS)
    set_struct_12 = set(STRUCTURAL_12_DAMAGE_LABELS)
    set_shared_4 = set(SHARED_4_LABELS)
    set_other_9 = set(EXTENDED_DAMAGE_LABELS)

    for i, jp in enumerate(json_paths):
        with open(jp, "r", encoding="utf-8") as f:
            ann = json.load(f)
        w, h = ann.get("imageWidth", 512), ann.get("imageHeight", 512)
        shapes = ann.get("shapes", [])
        masks_all_13[i] = rasterize_polygons(shapes, set_all_13, w, h, out_size)
        masks_struct_12[i] = rasterize_polygons(shapes, set_struct_12, w, h, out_size)
        masks_shared_4[i] = rasterize_polygons(shapes, set_shared_4, w, h, out_size)
        masks_other_9[i] = rasterize_polygons(shapes, set_other_9, w, h, out_size)
        for lbl in tracked_indiv_labels:
            indiv_masks[lbl][i] = rasterize_polygons(shapes, {lbl}, w, h, out_size)

    print("Precomputing connected components for AUPRO...")
    cc_all_13 = precompute_cc_slices(masks_all_13)
    cc_struct_12 = precompute_cc_slices(masks_struct_12)
    cc_shared_4 = precompute_cc_slices(masks_shared_4)

    print(f"Pre-rasterized 975 masks & CC slices in {time.time() - t0:.1f}s.")
    return {
        "masks_all_13": masks_all_13,
        "masks_struct_12": masks_struct_12,
        "masks_shared_4": masks_shared_4,
        "masks_other_9": masks_other_9,
        "indiv_masks": indiv_masks,
        "cc_all_13": cc_all_13,
        "cc_struct_12": cc_struct_12,
        "cc_shared_4": cc_shared_4,
    }


def compute_dacl_metrics_for_protocol(
    amaps_norm, final_sp, gt_masks, cc_list, indiv_masks, source_fixed_th=0.5, ignore_masks=None
):
    gt_px_full = gt_masks.reshape(-1).astype(np.uint8)
    pr_px_full = amaps_norm.reshape(-1).astype(np.float32)

    if ignore_masks is not None:
        ign_px = (ignore_masks.reshape(-1) > 0) & (gt_px_full == 0)
        valid_px = ~ign_px
        gt_px = gt_px_full[valid_px]
        pr_px = pr_px_full[valid_px]
    else:
        gt_px = gt_px_full
        pr_px = pr_px_full

    pos_vals = pr_px[gt_px == 1]
    neg_vals = pr_px[gt_px == 0]
    total_pos = float(len(pos_vals))
    total_neg = float(len(neg_vals))

    fine_ths = np.linspace(0.0, 1.0, 4097, dtype=np.float32)
    bins_fine = np.concatenate([[-np.inf], fine_ths, [np.inf]])
    pos_hist_f, _ = np.histogram(pos_vals, bins=bins_fine)
    neg_hist_f, _ = np.histogram(neg_vals, bins=bins_fine)
    tp_f = np.cumsum(pos_hist_f[::-1])[::-1][1:].astype(np.float64)
    fp_f = np.cumsum(neg_hist_f[::-1])[::-1][1:].astype(np.float64)
    tpr_f = np.concatenate([[0.0], (tp_f / total_pos)[::-1], [1.0]])
    fpr_f = np.concatenate([[0.0], (fp_f / total_neg)[::-1], [1.0]])
    p_auroc = float(auc(fpr_f, tpr_f)) * 100.0

    rec_f = (tp_f / total_pos)[::-1]
    prec_f = (tp_f / np.maximum(tp_f + fp_f, 1e-12))[::-1]
    rec_inc = np.concatenate([[0.0], rec_f, [1.0]])
    prec_inc = np.concatenate([[1.0], prec_f, [total_pos / (total_pos + total_neg)]])
    p_ap = float(np.sum((rec_inc[1:] - rec_inc[:-1]) * prec_inc[1:])) * 100.0

    ths = np.linspace(0.01, 0.99, 99, dtype=np.float32)
    bins_99 = np.concatenate([[-np.inf], ths, [np.inf]])
    pos_hist_99, _ = np.histogram(pos_vals, bins=bins_99)
    neg_hist_99, _ = np.histogram(neg_vals, bins=bins_99)
    tp_99 = np.cumsum(pos_hist_99[::-1])[::-1][1:].astype(np.float64)
    fp_99 = np.cumsum(neg_hist_99[::-1])[::-1][1:].astype(np.float64)
    pred_pos_99 = tp_99 + fp_99
    f1_arr = (2.0 * tp_99) / (pred_pos_99 + total_pos + 1e-8)
    iou_arr = tp_99 / (pred_pos_99 + total_pos - tp_99 + 1e-8)
    best_idx = int(np.argmax(f1_arr))
    p_f1 = float(f1_arr[best_idx]) * 100.0
    p_iou = float(iou_arr[best_idx]) * 100.0
    best_th = float(ths[best_idx])

    src_idx = int(np.argmin(np.abs(ths - source_fixed_th)))
    p_f1_src_th = float(f1_arr[src_idx]) * 100.0
    p_iou_src_th = float(iou_arr[src_idx]) * 100.0

    aupro_repo, aupro_std03 = fast_aupro_precomputed(gt_masks, amaps_norm, cc_list, ignore_masks=ignore_masks)
    p_aupro = aupro_repo * 100.0
    p_aupro_std03 = aupro_std03 * 100.0
    pixel_mean_4 = (p_auroc + p_f1 + p_ap + p_aupro) / 4.0

    pred_best = (amaps_norm >= best_th)
    pred_src = (amaps_norm >= source_fixed_th)

    per_class_best = {}
    per_class_src = {}
    for cname, dacl_lbl in SHARED_4_MAP.items():
        cm = (indiv_masks[dacl_lbl] > 0)
        tot = float(cm.sum())
        if tot > 0:
            per_class_best[f"{cname}_Recall"] = round(float((pred_best & cm).sum()) / tot * 100.0, 2)
            per_class_src[f"{cname}_Recall"] = round(float((pred_src & cm).sum()) / tot * 100.0, 2)

    cm_crack_all = (indiv_masks["Crack"] > 0) | (indiv_masks["ACrack"] > 0)
    per_class_best["Crack_plus_ACrack_Recall"] = round(
        float((pred_best & cm_crack_all).sum()) / float(cm_crack_all.sum()) * 100.0, 2
    )

    shared4_macro_best = round(
        float(np.mean([per_class_best[f"{c}_Recall"] for c in ["Crack", "Spalling", "Corrosion", "Efflorescence"]])), 2
    )
    shared4_macro_src = round(
        float(np.mean([per_class_src[f"{c}_Recall"] for c in ["Crack", "Spalling", "Corrosion", "Efflorescence"]])), 2
    )
    per_class_best["Shared4_Macro_Recall"] = shared4_macro_best
    per_class_src["Shared4_Macro_Recall"] = shared4_macro_src

    extended_recalls = {}
    for elbl in ["ACrack", "ExposedRebars", "Cavity", "Wetspot", "Weathering", "Hollowareas", "Rockpocket", "WConccor"]:
        cm = (indiv_masks[elbl] > 0)
        tot = float(cm.sum())
        if tot > 0:
            extended_recalls[f"{elbl}_Recall"] = round(float((pred_best & cm).sum()) / tot * 100.0, 2)

    gt_sp = (gt_masks.reshape(gt_masks.shape[0], -1).sum(axis=1) > 0).astype(np.int64)
    pos_imgs = int(gt_sp.sum())
    neg_imgs = int(len(gt_sp) - pos_imgs)
    if pos_imgs > 0 and neg_imgs > 0:
        i_auroc = round(float(roc_auc_score(gt_sp, final_sp)) * 100.0, 2)
        i_ap = round(float(average_precision_score(gt_sp, final_sp)) * 100.0, 2)
        prec_s, rec_s, _ = precision_recall_curve(gt_sp, final_sp)
        i_f1 = round(float(np.max(2 * prec_s * rec_s / (prec_s + rec_s + 1e-8))) * 100.0, 2)
    else:
        i_auroc, i_ap, i_f1 = 0.0, 0.0, 0.0

    return {
        "Pixel-AUROC": round(p_auroc, 2),
        "Pixel-AP": round(p_ap, 2),
        "Pixel-AUPRO": round(p_aupro, 2),
        "Pixel-AUPRO-Std03": round(p_aupro_std03, 2),
        "Pixel-F1": round(p_f1, 2),
        "Pixel-IoU": round(p_iou, 2),
        "Pixel-Mean-4": round(pixel_mean_4, 2),
        "Best-Th": round(best_th, 3),
        "Source-Fixed-Th": round(source_fixed_th, 3),
        "Pixel-F1-at-Source-Th": round(p_f1_src_th, 2),
        "Pixel-IoU-at-Source-Th": round(p_iou_src_th, 2),
        "Per-Class-Shared4": per_class_best,
        "Per-Class-Shared4-at-Source-Th": per_class_src,
        "Per-Class-Extended": extended_recalls,
        "Image-Reference": {
            "Positive-Images": pos_imgs,
            "Negative-Images": neg_imgs,
            "Image-AUROC": i_auroc,
            "Image-AP": i_ap,
            "Image-F1": i_f1,
        },
    }


@torch.no_grad()
def infer_adaptclip_on_dacl(short_ds, device):
    ADAPTCLIP_DIR = "/home/skye/data/Skye/AdaptCLIP_paper_a55cfbf"
    if ADAPTCLIP_DIR not in sys.path:
        sys.path.insert(0, ADAPTCLIP_DIR)
    import adaptcliplib
    from adaptcliplib import TextualAdapter, VisualAdapter
    from run_comp_adaptclip import get_adaptclip_transform

    image_size = 518
    features_list = [6, 12, 18, 24]
    preprocess = get_adaptclip_transform(image_size)
    dataset = DACL10kCustomValDataset(preprocess)
    loader = data.DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    ckpt_dir = os.path.join(CUR_DIR, "outputs", f"comp_adaptclip_{short_ds}")
    with open(os.path.join(ckpt_dir, "test_metrics.json"), "r", encoding="utf-8") as f:
        src_meta = json.load(f)
    source_th = float(src_meta.get("val_best_th", src_meta["in_domain_test"].get("Best-Th", 0.35)))

    model, _ = adaptcliplib.load(
        "ViT-L/14@336px",
        device=device,
        download_root="/home/skye/data/Skye/Bayes-PFL-main/pretrained_weight",
    )
    model.visual.DAPM_replace(DPAM_layer=20)
    textual_learner = TextualAdapter(model.to("cpu"), image_size, 12)
    visual_learner = VisualAdapter(image_size, 14, input_dim=768, reduction=4)
    model.to(device)
    textual_learner.to(device)
    visual_learner.to(device)
    model.eval()
    textual_learner.prepare_static_text_feature(model)

    ckpt = torch.load(os.path.join(ckpt_dir, "best_model.pth"), map_location=device)
    textual_learner.load_state_dict(ckpt["textual_learner"])
    visual_learner.load_state_dict(ckpt["visual_learner"])
    textual_learner.eval()
    visual_learner.eval()

    static_text_features = textual_learner.static_text_features
    learned_prompts, tokenized_prompts = textual_learner()
    learned_text_features = model.encode_text(learned_prompts, tokenized_prompts).float()

    pr_sp_list, amaps_list = [], []
    for items in tqdm(loader, desc=f"AdaptCLIP_{short_ds}->DACL", leave=False):
        query_image = items["img"].to(device)
        bsz = query_image.shape[0]
        with torch.cuda.amp.autocast():
            query_feats, query_patch_feats = model.encode_image(query_image, features_list, DPAM_layer=20)
        query_feats = query_feats.float()
        query_patch_feats = [pf.float() for pf in query_patch_feats]

        global_vl_logit, local_vl_map = visual_learner(query_feats, query_patch_feats, static_text_features)
        local_vl_map = local_vl_map[:, 1].detach()
        global_vl_score = global_vl_logit.softmax(-1)[:, 1].detach()

        global_tl_logit, local_tl_map = textual_learner.compute_global_local_score(
            query_feats, query_patch_feats, learned_text_features
        )
        local_tl_map = local_tl_map[:, 1].detach()
        global_tl_score = global_tl_logit.softmax(-1)[:, 1].detach()

        pixel_anomaly_map = 0.5 * (local_vl_map + local_tl_map)
        amap_np = pixel_anomaly_map.float().cpu().numpy()

        smooth_batch = []
        for b in range(bsz):
            sm = gaussian_filter(amap_np[b], sigma=4.0)
            smooth_batch.append(sm)
            amaps_list.append(sm.astype(np.float16))

        smooth_t = torch.from_numpy(np.stack(smooth_batch, axis=0)).to(device)
        anomaly_map_max, _ = torch.max(smooth_t.view(bsz, -1), dim=1)
        image_anomaly_pred = (global_vl_score + global_tl_score + anomaly_map_max) / 3.0
        sp_np = image_anomaly_pred.float().cpu().numpy()
        if sp_np.ndim == 0:
            sp_np = np.array([float(sp_np)])
        for b in range(bsz):
            pr_sp_list.append(float(sp_np[b]))

    amaps_norm = norm01(np.stack(amaps_list, axis=0).astype(np.float32))
    amaps_list.clear()
    final_sp = np.array(pr_sp_list, dtype=np.float32)
    return amaps_norm, final_sp, source_th


@torch.no_grad()
def infer_wpformer_on_dacl(short_ds, device):
    WPFORMER_DIR = "/home/skye/data/Skye/WPFormer"
    if WPFORMER_DIR not in sys.path:
        sys.path.insert(0, WPFORMER_DIR)
    from run_comp_wpformer import get_wpformer_transform
    orig_cwd = os.getcwd()
    os.chdir(WPFORMER_DIR)
    try:
        from model.WPFormer import WPFormer
        model = WPFormer(method="pvt_v2_b2", channel=64).to(device)
    finally:
        os.chdir(orig_cwd)

    preprocess = get_wpformer_transform(512)
    dataset = DACL10kCustomValDataset(preprocess)
    loader = data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

    ckpt_dir = os.path.join(CUR_DIR, "outputs", f"comp_wpformer_{short_ds}")
    with open(os.path.join(ckpt_dir, "test_metrics.json"), "r", encoding="utf-8") as f:
        src_meta = json.load(f)
    source_th = float(src_meta.get("val_best_th", src_meta["in_domain_test"].get("Best-Th", 0.35)))

    model.load_state_dict(torch.load(os.path.join(ckpt_dir, "best_model.pth"), map_location=device))
    model.eval()

    amaps_list = []
    for items in tqdm(loader, desc=f"WPFormer_{short_ds}->DACL", leave=False):
        imgs = items["img"].to(device)
        with torch.cuda.amp.autocast():
            preds = model(imgs)[-1]  # [B, 1, 512, 512]
        probs = torch.sigmoid(preds.float())
        probs_up = F.interpolate(probs, size=(518, 518), mode="bilinear", align_corners=False)[:, 0]
        amap_np = probs_up.float().cpu().numpy()
        for b in range(imgs.shape[0]):
            sm = gaussian_filter(amap_np[b], sigma=1.5)
            amaps_list.append(sm.astype(np.float16))

    amaps_norm = norm01(np.stack(amaps_list, axis=0).astype(np.float32))
    amaps_list.clear()
    final_sp = topk_pool_np(amaps_norm, ratio=0.01)
    return amaps_norm, final_sp, source_th


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True, choices=["adaptclip", "wpformer"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dacl_masks = precompute_dacl_masks(out_size=518)

    out_path = os.path.join(CUR_DIR, "outputs", "dacl10k_v3_zero_shot_results.json")
    with open(out_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    if args.method == "adaptclip":
        m_key = "comp_adaptclip"
        m_label = "AdaptCLIP (AAAI 2026)"
        infer_fn = infer_adaptclip_on_dacl
    else:
        m_key = "comp_wpformer"
        m_label = "WPFormer (CVPR 2025)"
        infer_fn = infer_wpformer_on_dacl

    for short_ds, dir_key in [("S2DS", "S2DS_to_DACL10k"), ("CODEBRIM", "CODEBRIM_to_DACL10k")]:
        t0 = time.time()
        print(f"=== Evaluating {m_label} ({short_ds} -> DACL10k-v3) ===")
        amaps_norm, final_sp, src_th = infer_fn(short_ds, device)

        m_all13 = compute_dacl_metrics_for_protocol(
            amaps_norm, final_sp, dacl_masks["masks_all_13"], dacl_masks["cc_all_13"],
            dacl_masks["indiv_masks"], source_fixed_th=src_th, ignore_masks=None
        )
        m_struct12 = compute_dacl_metrics_for_protocol(
            amaps_norm, final_sp, dacl_masks["masks_struct_12"], dacl_masks["cc_struct_12"],
            dacl_masks["indiv_masks"], source_fixed_th=src_th, ignore_masks=None
        )
        m_shared4_ign = compute_dacl_metrics_for_protocol(
            amaps_norm, final_sp, dacl_masks["masks_shared_4"], dacl_masks["cc_shared_4"],
            dacl_masks["indiv_masks"], source_fixed_th=src_th, ignore_masks=dacl_masks["masks_other_9"]
        )

        results[dir_key][m_key] = {
            "model_key": m_key,
            "label": m_label,
            "source_dataset": short_ds,
            "source_fixed_th": src_th,
            "elapsed_sec": round(time.time() - t0, 1),
            "all_13_damages": m_all13,
            "structural_12_damages": m_struct12,
            "shared_4_ignore_other": m_shared4_ign,
        }
        p_m4 = m_all13["Pixel-Mean-4"]
        c_rec = m_all13["Per-Class-Shared4"]["Crack_Recall"]
        s4_rec = m_all13["Per-Class-Shared4"]["Shared4_Macro_Recall"]
        print(
            f"  -> [All-13 Damages] P-AUROC={m_all13['Pixel-AUROC']:.2f}, P-F1={m_all13['Pixel-F1']:.2f}, "
            f"P-AP={m_all13['Pixel-AP']:.2f}, P-AUPRO={m_all13['Pixel-AUPRO']:.2f} | Pixel-Mean-4={p_m4:.2f} | "
            f"Crack={c_rec:.2f}%, Shared4-Macro={s4_rec:.2f}% ({time.time()-t0:.1f}s)"
        )

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Finished {m_label} on DACL10k-v3 and updated {out_path}")


if __name__ == "__main__":
    main()
