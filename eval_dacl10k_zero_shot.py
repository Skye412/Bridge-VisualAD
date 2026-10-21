#!/usr/bin/env python3
"""
Zero-Shot Cross-Dataset & Extended-Damage Generalization Evaluation on DACL10k-v3 (validation split, 975 images).
Strictly evaluates existing trained checkpoints (S2DS -> DACL10k-v3 and CODEBRIM -> DACL10k-v3) without any re-training
or hyperparameter tuning on DACL10k-v3.
"""
import os
import sys
import glob
import json
import time
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.ndimage import gaussian_filter
import torch
import torch.utils.data as data
from torch.cuda.amp import autocast
from tqdm import tqdm

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.join(CUR_DIR, "VisualAD_repo")
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
if CUR_DIR not in sys.path:
    sys.path.insert(0, CUR_DIR)

import VisualAD_lib
from utils.transforms import get_transform
from utils.scoring import DEFAULT_TOPK_RATIO
from run_experiments import BridgeVisualADWrapper
from run_true_ablation_train import StandaloneAblationWrapper
from run_a3_to_a4_scratch_8ep import Scratch8EpFactorialWrapper
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

# Official DACL10k-v3 (WACV 2024) taxonomy:
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


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


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


class DACL10kValDataset(data.Dataset):
    def __init__(self, image_size=518):
        super().__init__()
        self.image_size = image_size
        self.preprocess, _ = get_transform(Namespace(image_size=image_size))
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
    """Precompute connected component bounding slices and boolean submasks once for fast AUPRO."""
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

    # Standard [0, max_fpr] integrated AUPRO with linear boundary interpolation at max_fpr (0.3)
    order_all = np.argsort(fpr_arr)
    f_all = fpr_arr[order_all]
    p_all = mean_pros[order_all]
    if fprs[-1] < max_fpr and f_all[-1] > max_fpr:
        p_at_03 = float(np.interp(max_fpr, f_all, p_all))
    else:
        p_at_03 = float(pros[-1])
    fprs_ext = np.concatenate([[0.0], fprs, [max_fpr]])
    pros_ext = np.concatenate([[0.0], pros, [p_at_03]])
    aupro_std03 = float(auc(fprs_ext, pros_ext) / max_fpr)
    if fpr_span <= 1e-12:
        aupro_repo = float(pros.mean())
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
def infer_amaps_and_scores(model_key, short_ds, dataloader, device):
    features_list = [6, 12, 18, 24]
    clip_model, _ = VisualAD_lib.load("ViT-L/14@336px", device=device, design_details={
        "Prompt_length": 4, "learnabel_text_embedding_depth": 9, "learnabel_text_embedding_length": 4
    })
    clip_model.eval()

    out_root = os.path.join(CUR_DIR, "outputs")
    ckpt_dir = os.path.join(out_root, f"{model_key}_{short_ds}")
    ckpt_path = os.path.join(ckpt_dir, "best_model.pth")
    metrics_path = os.path.join(ckpt_dir, "test_metrics.json")
    with open(metrics_path, "r", encoding="utf-8") as f:
        src_meta = json.load(f)
    source_th = float(src_meta.get("val_best_th", src_meta["in_domain_test"].get("Best-Th", 0.35)))

    state = torch.load(ckpt_path, map_location=device)

    if model_key == "baseline":
        wrapper = BridgeVisualADWrapper(clip_model, features_list, image_size=518, variant="baseline", device=device).to(device)
        wrapper.load_trainable_state_dict(state, device)
        wrapper.eval()
        amaps_list = []
        for items in tqdm(dataloader, desc=f"{model_key}_{short_ds}->DACL", leave=False):
            img = items["img"].to(device)
            with autocast(enabled=(device.type == "cuda")):
                out = wrapper.forward_step(img)
                amap_b = out["final_anomaly_map"].float().cpu().numpy()
            for b in range(img.shape[0]):
                amaps_list.append(gaussian_filter(amap_b[b], sigma=4.0).astype(np.float16))
        amaps_norm = norm01(np.stack(amaps_list, axis=0).astype(np.float32))
        amaps_list.clear()
        final_sp = topk_pool_np(amaps_norm, DEFAULT_TOPK_RATIO)
        return amaps_norm, final_sp, source_th, None

    elif model_key in ("a1_mp_bs", "a2_wd_saf"):
        wrapper = StandaloneAblationWrapper(clip_model, features_list, image_size=518, variant=model_key, device=device).to(device)
        wrapper.load_trainable_state_dict(state, device)
        wrapper.eval()
        pre_list, post_list = [], []
        for items in tqdm(dataloader, desc=f"{model_key}_{short_ds}->DACL", leave=False):
            img = items["img"].to(device)
            with autocast(enabled=(device.type == "cuda")):
                out = wrapper.forward_step(img)
                pre_b = out["pre_sum"].float().cpu().numpy()
                post_b = out["post_sum"].float().cpu().numpy()
            for b in range(img.shape[0]):
                pre_list.append(gaussian_filter(pre_b[b], sigma=4.0).astype(np.float16))
                post_list.append(gaussian_filter(post_b[b], sigma=4.0).astype(np.float16))
        pre_n = norm01(np.stack(pre_list, axis=0).astype(np.float32))
        pre_list.clear()
        post_n = norm01(np.stack(post_list, axis=0).astype(np.float32))
        post_list.clear()
        amaps_norm = norm01(0.65 * pre_n + 0.35 * post_n)
        final_sp = topk_pool_np(amaps_norm, DEFAULT_TOPK_RATIO)
        return amaps_norm, final_sp, source_th, None

    elif model_key in ("a3_agds_wgd", "a3_plus_cw_8ep", "a3_plus_scorer_8ep", "a4_scratch8ep"):
        wrapper = Scratch8EpFactorialWrapper(clip_model, features_list, variant_key=model_key, image_size=518, device=device).to(device)
        wrapper.load_trainable_state_dict(state, device)
        wrapper.eval()
        pre_list, post_list, prob_list, conf_list, sc_list = [], [], [], [], []
        for items in tqdm(dataloader, desc=f"{model_key}_{short_ds}->DACL", leave=False):
            img = items["img"].to(device)
            with autocast(enabled=(device.type == "cuda")):
                out = wrapper.forward_step(img)
                pre_b = out["pre_sum"].float().cpu().numpy()
                post_b = out["post_sum"].float().cpu().numpy()
                prob_b = out["prob_map"].float().cpu().numpy()
                conf_b = torch.sigmoid(out["log_conf"]).float().cpu().numpy()
                if wrapper.use_scorer and out["dyn_score"] is not None:
                    sc_b = out["dyn_score"].float().cpu().numpy()
                    if sc_b.ndim == 0:
                        sc_b = np.array([float(sc_b)])
                else:
                    sc_b = None
            for b in range(img.shape[0]):
                pre_list.append(gaussian_filter(pre_b[b], sigma=4.0).astype(np.float16))
                post_list.append(gaussian_filter(post_b[b], sigma=4.0).astype(np.float16))
                prob_list.append(gaussian_filter(prob_b[b], sigma=1.5).astype(np.float16))
                conf_list.append(gaussian_filter(conf_b[b], sigma=2.0).astype(np.float16))
                if sc_b is not None:
                    sc_list.append(float(sc_b[b]))

        pre_n = norm01(np.stack(pre_list, axis=0).astype(np.float32))
        pre_list.clear()
        post_n = norm01(np.stack(post_list, axis=0).astype(np.float32))
        post_list.clear()
        prob_n = norm01(np.stack(prob_list, axis=0).astype(np.float32))
        prob_list.clear()
        conf_n = norm01(np.stack(conf_list, axis=0).astype(np.float32))
        conf_list.clear()

        cos_hybrid = norm01(0.65 * pre_n + 0.35 * post_n)
        amaps_linear = norm01(0.82 * cos_hybrid + 0.18 * prob_n)

        if wrapper.use_cw_loss:
            prob_conf = norm01(prob_n * (0.90 + 0.10 * conf_n))
            if short_ds == "S2DS":
                amaps_norm = norm01(cos_hybrid * (0.72 + 0.28 * prob_conf) + 0.18 * prob_conf)
            else:
                amaps_norm = norm01(cos_hybrid * (0.80 + 0.05 * prob_conf) + 0.18 * prob_conf)
        else:
            amaps_norm = amaps_linear

        if wrapper.use_scorer and len(sc_list) > 0:
            sc_n = norm01(np.array(sc_list))
            tk_002 = topk_pool_np(cos_hybrid, 0.002)
            tk_010 = topk_pool_np(amaps_norm, 0.010)
            tk_050 = topk_pool_np(amaps_norm, 0.050)
            if short_ds == "S2DS":
                final_sp = 0.70 * tk_002 + 0.20 * tk_010 + 0.10 * sc_n
            else:
                final_sp = 0.20 * tk_002 + 0.45 * tk_010 + 0.25 * tk_050 + 0.10 * sc_n
        else:
            final_sp = topk_pool_np(amaps_norm, DEFAULT_TOPK_RATIO)

        return amaps_norm, final_sp, source_th, (amaps_linear if wrapper.use_cw_loss else None)
    else:
        raise ValueError(f"Unknown model_key: {model_key}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dacl_masks = precompute_dacl_masks(out_size=518)
    dataset = DACL10kValDataset(image_size=518)
    dataloader = data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

    models_order = [
        ("baseline", "A0 (VisualAD Baseline)"),
        ("a1_mp_bs", "A1 (+MP-BS)"),
        ("a2_wd_saf", "A2 (+MP-BS +WD-SAF)"),
        ("a3_agds_wgd", "A3 (+MP-BS +WD-SAF +AGDS-WGD)"),
        ("a3_plus_cw_8ep", "A3 + CW-Loss"),
        ("a3_plus_scorer_8ep", "A3 + Scorer"),
        ("a4_scratch8ep", "A4 (Bridge-VisualAD Full)"),
    ]

    out_path = os.path.join(CUR_DIR, "outputs", "dacl10k_v3_zero_shot_results.json")
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            results = json.load(f)
    else:
        results = {
            "dataset": "DACL10k-v3 validation (975 images, 512x512)",
            "protocols": {
                "primary_all_13_damages": {
                    "description": "All 13 official DACL10k bridge damage classes merged as binary anomaly mask; 6 bridge object/component classes treated as normal background.",
                    "positive_damage_labels": ALL_13_DAMAGE_LABELS,
                    "normal_object_labels": OBJECT_6_LABELS,
                },
                "sub_structural_12_damages": {
                    "description": "12 structural bridge damage classes (excluding superficial Graffiti) merged as binary anomaly mask.",
                    "positive_damage_labels": STRUCTURAL_12_DAMAGE_LABELS,
                },
                "secondary_shared_4_ignore_other": {
                    "description": "4 shared defect classes (Crack, Spalling, Rust, Efflorescence) as positive anomaly; pixels belonging exclusively to the other 9 damage classes set to Ignore.",
                    "positive_damage_labels": SHARED_4_LABELS,
                    "ignored_damage_labels": EXTENDED_DAMAGE_LABELS,
                },
            },
            "S2DS_to_DACL10k": {},
            "CODEBRIM_to_DACL10k": {},
        }

    for short_ds, dir_key in [("S2DS", "S2DS_to_DACL10k"), ("CODEBRIM", "CODEBRIM_to_DACL10k")]:
        for m_key, m_label in models_order:
            if m_key in results[dir_key]:
                print(f"[SKIP] {dir_key} / {m_key} already computed.")
                continue
            t0 = time.time()
            print(f"=== Evaluating {m_label} ({short_ds} -> DACL10k-v3) ===")
            amaps_norm, final_sp, src_th, amaps_linear = infer_amaps_and_scores(m_key, short_ds, dataloader, device)

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

            entry = {
                "model_key": m_key,
                "label": m_label,
                "source_dataset": short_ds,
                "source_fixed_th": src_th,
                "elapsed_sec": round(time.time() - t0, 1),
                "all_13_damages": m_all13,
                "structural_12_damages": m_struct12,
                "shared_4_ignore_other": m_shared4_ign,
            }
            if amaps_linear is not None:
                entry["all_13_damages_linear_fusion"] = compute_dacl_metrics_for_protocol(
                    amaps_linear, final_sp, dacl_masks["masks_all_13"], dacl_masks["cc_all_13"],
                    dacl_masks["indiv_masks"], source_fixed_th=src_th, ignore_masks=None
                )

            results[dir_key][m_key] = entry
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

    print(f"All DACL10k-v3 zero-shot evaluations saved to {out_path}")


if __name__ == "__main__":
    main()
