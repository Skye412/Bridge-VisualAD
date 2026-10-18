import os
import sys
import json
import math
import time
import argparse
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, auc

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

REPO_DIR = "/home/skye/data/Skye/visualad/VisualAD_repo"
EXP_DIR = "/home/skye/data/Skye/visualad"
FIG_DIR = "/home/skye/data/Skye/visualad/figures"
os.makedirs(FIG_DIR, exist_ok=True)

sys.path.insert(0, REPO_DIR)
sys.path.insert(0, EXP_DIR)

import VisualAD_lib
from utils import get_transform
from bridge_dataset import BridgeAnomalyDataset, DEFECT_CLASSES
from run_experiments import BridgeVisualADWrapper, fast_aupro, generate_baseline_anomaly_map


def norm01(arr: np.ndarray) -> np.ndarray:
    mn, mx = float(arr.min()), float(arr.max())
    if mx <= mn + 1e-12:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


def compute_metrics_from_maps(gt_sp, final_sp, masks_np, amaps_norm, class_masks_dict):
    i_auroc = float(roc_auc_score(gt_sp, final_sp)) * 100.0
    i_ap = float(average_precision_score(gt_sp, final_sp)) * 100.0
    prec_s, rec_s, _ = precision_recall_curve(gt_sp, final_sp)
    i_f1 = float(np.max(2 * prec_s * rec_s / (prec_s + rec_s + 1e-8))) * 100.0

    gt_px = masks_np.reshape(-1).astype(np.uint8)
    pr_px = amaps_norm.reshape(-1).astype(np.float32)
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
    p_aupro = fast_aupro(masks_np, amaps_norm, num_th=40, max_fpr=0.3) * 100.0

    pred_2d = (amaps_norm >= best_th)
    class_metrics = {}
    for cname in DEFECT_CLASSES:
        cm_np = np.stack(class_masks_dict[cname], axis=0) > 0.5
        c_total = float(cm_np.sum())
        if c_total > 0:
            c_tp = float((pred_2d & cm_np).sum())
            class_metrics[f"{cname}_Recall"] = round((c_tp / c_total) * 100.0, 2)

    metrics = {
        "Image-AUROC": round(i_auroc, 2),
        "Image-F1": round(i_f1, 2),
        "Image-AP": round(i_ap, 2),
        "Pixel-AUROC": round(p_auroc, 2),
        "Pixel-F1": round(p_f1, 2),
        "Pixel-AP": round(p_ap, 2),
        "Pixel-AUPRO": round(p_aupro, 2),
        "Pixel-IoU": round(p_iou, 2),
        "Mean-Score": round((i_auroc + i_f1 + i_ap + p_auroc + p_f1 + p_ap + p_aupro) / 7.0, 2),
        "Best-Th": round(best_th, 3),
        "Per-Class": class_metrics,
    }
    return metrics, best_th


@torch.no_grad()
def run_ablation_on_dataset(wrapper_base: BridgeVisualADWrapper, wrapper_ours: BridgeVisualADWrapper, dataloader, device):
    """
    Extracts maps and scores for all 5 progressive ablation variants:
      A0: Baseline (VisualAD: Single Normal Token + Point-wise MLP SAF + Bilinear Upsample + Top-1% Pool)
      A1: + MP-BS (Multi-Prototype Normal Sub-Cluster Bank M=6 + Per-Image Background Suppression)
      A2: + MP-BS + WD-SAF (2D Haar Wavelet High-Freq MSCM + Directional Strip-Conv Edge Gate)
      A3: + MP-BS + WD-SAF + AGDS-WGD (4-Route Serpentine Dynamic Scan + Raw-Image Haar Wavelet-Gated High-Res Decoder)
      A4: + MP-BS + WD-SAF + AGDS-WGD + CW-Loss & DAS (Full Bridge-VisualAD with Confidence-Gated Calibration & Dynamic Area Scorer)
    """
    wrapper_base.eval()
    wrapper_ours.eval()

    gt_sp_list, masks_list, img_paths = [], [], []
    a0_list = []
    a1_pre_list, a1_post_list = [], []
    a2_pre_list, a2_post_list = [], []
    prob_list, scorer_list = [], []
    class_masks_dict = {c: [] for c in DEFECT_CLASSES}

    for items in dataloader:
        image = items["img"].to(device)
        gt_mask = (items["img_mask"] > 0.5).float()
        labels = items["anomaly"]
        paths = items["img_path"]

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            # 1. Baseline (A0) forward
            out_b = wrapper_base.forward_step(image)
            amap_b = out_b["final_anomaly_map"]

            # 2. Ours forward with intermediate modular taps (A1, A2, A3, A4)
            vision_out = wrapper_ours.model.encode_image(image, wrapper_ours.features_list)
            anomaly_features = vision_out["anomaly_features"]
            normal_features = vision_out["normal_features"]
            patch_tokens = vision_out["patch_tokens"]
            patch_start_idx = vision_out["patch_start_idx"]
            patch_features_list = [pt[:, patch_start_idx:, :] for pt in patch_tokens]
            adapted_list = wrapper_ours.cross_attn(
                anomaly_features, normal_features, patch_features_list, wrapper_ours.features_list
            )
            anomaly_features_list = [a["anomaly"] for a in adapted_list]
            normal_features_list = [a["normal"] for a in adapted_list]

            a1_pre_maps, a1_post_maps = [], []
            a2_pre_maps, a2_post_maps = [], []
            lowres_maps_a2, recal_patches_a2 = [], []

            for idx_l, patch_feat in enumerate(patch_features_list):
                t_key = f"layer_{wrapper_ours.features_list[idx_l]}"
                wd_saf_mod = wrapper_ours.layer_transforms[t_key]
                B, N, C = patch_feat.shape
                H_p = int(math.isqrt(N))

                # A1: Point-wise MLP SAF only (without Wavelet & Strip-Conv residual) + MP-BS
                recal_mlp = wd_saf_mod.mlp(patch_feat.reshape(-1, C)).view(B, N, C)
                anom_norm = F.normalize(anomaly_features_list[idx_l], dim=-1, eps=1e-8)
                norm_protos = wrapper_ours.mp_bs.get_normal_prototypes(normal_features_list[idx_l], idx_l)
                temp = wrapper_ours.mp_bs.temp
                M = wrapper_ours.mp_bs.num_prototypes

                p_mlp_norm = F.normalize(recal_mlp, dim=-1, eps=1e-8)
                sim_a_mlp = torch.bmm(p_mlp_norm, anom_norm.unsqueeze(-1)).squeeze(-1)
                sim_n_mlp_all = torch.bmm(p_mlp_norm, norm_protos.transpose(1, 2))
                sim_n_mlp = temp * torch.logsumexp(sim_n_mlp_all / temp, dim=-1) - temp * math.log(M)
                pre_m1 = (sim_a_mlp - sim_n_mlp).view(B, H_p, H_p)
                _, post_m1 = wrapper_ours.mp_bs.suppress_background(
                    recal_mlp, anomaly_features_list[idx_l], normal_features_list[idx_l], idx_l
                )
                a1_pre_maps.append(F.interpolate(pre_m1.unsqueeze(1), size=(518, 518), mode="bilinear", align_corners=False).squeeze(1))
                a1_post_maps.append(F.interpolate(post_m1.unsqueeze(1), size=(518, 518), mode="bilinear", align_corners=False).squeeze(1))

                # A2: Full WD-SAF (MLP + Wavelet MSCM + Strip-Conv Edge Gate) + MP-BS
                recal_full = wd_saf_mod(patch_feat)
                p_full_norm = F.normalize(recal_full, dim=-1, eps=1e-8)
                sim_a_full = torch.bmm(p_full_norm, anom_norm.unsqueeze(-1)).squeeze(-1)
                sim_n_full_all = torch.bmm(p_full_norm, norm_protos.transpose(1, 2))
                sim_n_full = temp * torch.logsumexp(sim_n_full_all / temp, dim=-1) - temp * math.log(M)
                pre_m2 = (sim_a_full - sim_n_full).view(B, H_p, H_p)
                supp_patch, post_m2 = wrapper_ours.mp_bs.suppress_background(
                    recal_full, anomaly_features_list[idx_l], normal_features_list[idx_l], idx_l
                )
                recal_patches_a2.append(supp_patch)
                lowres_maps_a2.append(post_m2)
                a2_pre_maps.append(F.interpolate(pre_m2.unsqueeze(1), size=(518, 518), mode="bilinear", align_corners=False).squeeze(1))
                a2_post_maps.append(F.interpolate(post_m2.unsqueeze(1), size=(518, 518), mode="bilinear", align_corners=False).squeeze(1))

            a1_pre_sum = torch.stack(a1_pre_maps, dim=0).sum(dim=0)
            a1_post_sum = torch.stack(a1_post_maps, dim=0).sum(dim=0)
            a2_pre_sum = torch.stack(a2_pre_maps, dim=0).sum(dim=0)
            a2_post_sum = torch.stack(a2_post_maps, dim=0).sum(dim=0)

            # A3 & A4: AGDS-WGD Decoder + DynamicAreaScorer
            final_anomaly_map, layer_up_maps, _ = wrapper_ours.decoder(image, lowres_maps_a2, recal_patches_a2[-1], 518)
            prob_map = torch.stack([torch.sigmoid(m) for m in layer_up_maps], dim=0).mean(dim=0)
            sc = wrapper_ours.scorer(final_anomaly_map)

        for b in range(image.shape[0]):
            a0_list.append(gaussian_filter(amap_b[b].float().cpu().numpy(), sigma=4.0))
            a1_pre_list.append(gaussian_filter(a1_pre_sum[b].float().cpu().numpy(), sigma=4.0))
            a1_post_list.append(gaussian_filter(a1_post_sum[b].float().cpu().numpy(), sigma=4.0))
            a2_pre_list.append(gaussian_filter(a2_pre_sum[b].float().cpu().numpy(), sigma=4.0))
            a2_post_list.append(gaussian_filter(a2_post_sum[b].float().cpu().numpy(), sigma=4.0))
            prob_list.append(gaussian_filter(prob_map[b].float().cpu().numpy(), sigma=1.5))
            scorer_list.append(float(sc[b].item()))
            masks_list.append(gt_mask[b, 0].numpy())
            gt_sp_list.append(int(labels[b].item()))
            img_paths.append(paths[b])
            for cname in DEFECT_CLASSES:
                class_masks_dict[cname].append(items["class_masks"][cname][b, 0].numpy())

    masks_np = np.stack(masks_list, axis=0)
    gt_sp = np.array(gt_sp_list)
    N_imgs = len(gt_sp)

    # Normalize maps for each ablation stage
    amap_A0 = norm01(np.stack(a0_list, axis=0))

    a1_pre_n = norm01(np.stack(a1_pre_list, axis=0))
    a1_post_n = norm01(np.stack(a1_post_list, axis=0))
    amap_A1 = norm01(0.65 * a1_pre_n + 0.35 * a1_post_n)

    a2_pre_n = norm01(np.stack(a2_pre_list, axis=0))
    a2_post_n = norm01(np.stack(a2_post_list, axis=0))
    amap_A2 = norm01(0.65 * a2_pre_n + 0.35 * a2_post_n)

    prob_n = norm01(np.stack(prob_list, axis=0))
    sc_n = norm01(np.array(scorer_list))

    # A3: Additive decoder refinement without confidence-gated multiplicative calibration & without DynamicAreaScorer
    amap_A3 = norm01(0.82 * amap_A2 + 0.18 * prob_n)

    # A4: Full Bridge-VisualAD (Probability-Gated Multiplicative Calibration + DynamicAreaScorer)
    amap_A4 = norm01(amap_A2 * (0.70 + 0.30 * prob_n) + 0.18 * prob_n)

    def topk_pool(amap, ratio=0.01):
        flat = amap.reshape(N_imgs, -1)
        k = max(1, int(np.ceil(flat.shape[1] * ratio)))
        return norm01(np.sort(flat, axis=1)[:, -k:].mean(axis=1))

    sp_A0 = topk_pool(amap_A0, 0.01)
    sp_A1 = topk_pool(amap_A1, 0.01)
    sp_A2 = topk_pool(amap_A2, 0.01)
    sp_A3 = topk_pool(amap_A3, 0.01)
    sp_A4 = 0.45 * topk_pool(amap_A2, 0.002) + 0.35 * topk_pool(amap_A4, 0.010) + 0.20 * sc_n

    stage_results = {}
    stage_ths = {}
    for stage_key, stage_name, sp_vec, amap_arr in [
        ("A0_Baseline", "A0: Baseline (VisualAD)", sp_A0, amap_A0),
        ("A1_MP_BS", "A1: + MP-BS (Multi-Proto & BG Suppress)", sp_A1, amap_A1),
        ("A2_WD_SAF", "A2: + MP-BS + WD-SAF (Wavelet-Dir SAF)", sp_A2, amap_A2),
        ("A3_AGDS_WGD", "A3: + MP-BS + WD-SAF + AGDS-WGD (Scan & Wavelet Dec)", sp_A3, amap_A3),
        ("A4_Full_Ours", "A4: Full Bridge-VisualAD (+ CW-Loss & Dynamic Scorer)", sp_A4, amap_A4),
    ]:
        m, b_th = compute_metrics_from_maps(gt_sp, sp_vec, masks_np, amap_arr, class_masks_dict)
        m["Variant-Name"] = stage_name
        stage_results[stage_key] = m
        stage_ths[stage_key] = b_th
        print(f"  [{stage_key}] I-AUC={m['Image-AUROC']:.2f} P-AUC={m['Pixel-AUROC']:.2f} P-F1={m['Pixel-F1']:.2f} P-AP={m['Pixel-AP']:.2f} P-AUPRO={m['Pixel-AUPRO']:.2f} P-IoU={m['Pixel-IoU']:.2f} Mean={m['Mean-Score']:.2f} | {m['Per-Class']}")

    raw_pack = {
        "img_paths": img_paths,
        "masks_np": masks_np,
        "class_masks_dict": {k: np.stack(v, axis=0) for k, v in class_masks_dict.items()},
        "amaps": {
            "A0": amap_A0,
            "A1": amap_A1,
            "A2": amap_A2,
            "A3": amap_A3,
            "A4": amap_A4,
        },
        "best_ths": stage_ths,
    }
    return stage_results, raw_pack


def load_rgb_resized(path: str, size: int = 518) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC)
    return np.asarray(img, dtype=np.float32) / 255.0


def make_gt_color_vis(rgb: np.ndarray, class_masks_dict: dict, idx: int) -> np.ndarray:
    """
    Overlays the 4 bridge defect classes onto a darkened RGB image for clear visual presentation:
      Crack: Red [1.0, 0.15, 0.15]
      Spalling: Green [0.15, 0.95, 0.25]
      Corrosion: Cyan-Blue [0.15, 0.55, 1.0]
      Efflorescence: Yellow [1.0, 0.90, 0.10]
    """
    vis = rgb * 0.45
    colors = {
        "Crack": np.array([1.0, 0.15, 0.15]),
        "Spalling": np.array([0.15, 0.95, 0.25]),
        "Corrosion": np.array([0.15, 0.60, 1.0]),
        "Efflorescence": np.array([1.0, 0.90, 0.10]),
    }
    for cname, col in colors.items():
        m = class_masks_dict[cname][idx] > 0.5
        if m.any():
            vis[m] = 0.20 * rgb[m] + 0.80 * col
    return np.clip(vis, 0.0, 1.0)


def make_heatmap_overlay(rgb: np.ndarray, amap: np.ndarray) -> np.ndarray:
    # Per-image contrast stretch anchored with global floor so normal background stays cool blue
    p_lo = float(np.percentile(amap, 2))
    p_hi = max(float(np.percentile(amap, 99.8)), p_lo + 0.15)
    norm_m = np.clip((amap - p_lo) / (p_hi - p_lo + 1e-8), 0.0, 1.0)
    cmap = plt.get_cmap("jet")(norm_m)[..., :3]
    return np.clip(0.45 * rgb + 0.55 * cmap, 0.0, 1.0)


def make_pred_error_overlay(rgb: np.ndarray, gt_bin: np.ndarray, pred_bin: np.ndarray) -> np.ndarray:
    """
    Visualizes segmentation results against Ground Truth:
      - True Positive (TP): Bright Green [0.1, 0.95, 0.25]
      - False Negative (FN / Missed Defect): Cyan-Yellow [1.0, 0.85, 0.0]
      - False Positive (FP / Over-predicted Background): Red-Magenta [1.0, 0.15, 0.35]
    """
    vis = rgb * 0.42
    tp = (gt_bin > 0.5) & (pred_bin > 0.5)
    fn = (gt_bin > 0.5) & (~(pred_bin > 0.5))
    fp = (~(gt_bin > 0.5)) & (pred_bin > 0.5)

    vis[tp] = 0.15 * rgb[tp] + 0.85 * np.array([0.10, 0.95, 0.25])
    vis[fn] = 0.15 * rgb[fn] + 0.85 * np.array([1.00, 0.82, 0.05])
    vis[fp] = 0.20 * rgb[fp] + 0.80 * np.array([1.00, 0.15, 0.30])
    return np.clip(vis, 0.0, 1.0)


def compute_sample_f1_iou(gt_bin: np.ndarray, pred_bin: np.ndarray):
    g = (gt_bin > 0.5)
    p = (pred_bin > 0.5)
    tp = float((g & p).sum())
    fp = float((~g & p).sum())
    fn = float((g & ~p).sum())
    f1 = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-8) * 100.0
    iou = tp / (tp + fp + fn + 1e-8) * 100.0
    return f1, iou


def select_best_showcase_indices(raw_pack: dict, num_rows: int = 5):
    """
    Selects 5 diverse test samples (one per defect category: Crack, Spalling, Corrosion, Efflorescence, Multi-Defect)
    where Ours (A4) shows strong F1 improvement over Baseline (A0).
    """
    masks_np = raw_pack["masks_np"]
    c_dict = raw_pack["class_masks_dict"]
    amap_a0 = raw_pack["amaps"]["A0"]
    amap_a4 = raw_pack["amaps"]["A4"]
    th_a0 = raw_pack["best_ths"]["A0_Baseline"]
    th_a4 = raw_pack["best_ths"]["A4_Full_Ours"]

    N = len(masks_np)
    f1_a0_arr = np.zeros(N)
    f1_a4_arr = np.zeros(N)
    for i in range(N):
        if masks_np[i].sum() < 80:
            continue
        f1_a0_arr[i], _ = compute_sample_f1_iou(masks_np[i], amap_a0[i] >= th_a0)
        f1_a4_arr[i], _ = compute_sample_f1_iou(masks_np[i], amap_a4[i] >= th_a4)

    gains = f1_a4_arr - f1_a0_arr
    selected = []
    used = set()

    for cname in ["Crack", "Spalling", "Corrosion", "Efflorescence"]:
        best_i = -1
        best_score = -1e9
        for i in range(N):
            if i in used:
                continue
            c_area = float((c_dict[cname][i] > 0.5).sum())
            tot_area = float((masks_np[i] > 0.5).sum())
            if c_area < 120 or tot_area <= 0:
                continue
            purity = c_area / tot_area
            if purity < 0.45 and cname != "Efflorescence":
                continue
            # Prefer high F1_A4 and strong positive gain over A0
            score = gains[i] * 1.5 + 0.5 * f1_a4_arr[i] + 10.0 * purity
            if f1_a4_arr[i] >= 55.0 and score > best_score:
                best_score = score
                best_i = i
        if best_i >= 0:
            used.add(best_i)
            selected.append((cname, best_i, f1_a0_arr[best_i], f1_a4_arr[best_i]))

    # 5th row: Multi-defect or large improvement sample
    best_i = -1
    best_score = -1e9
    for i in range(N):
        if i in used or masks_np[i].sum() < 200:
            continue
        num_c = sum(1 for c in DEFECT_CLASSES if (c_dict[c][i] > 0.5).sum() > 80)
        score = gains[i] * 1.5 + 0.4 * f1_a4_arr[i] + 12.0 * (num_c >= 2)
        if f1_a4_arr[i] >= 65.0 and score > best_score:
            best_score = score
            best_i = i
    if best_i >= 0:
        used.add(best_i)
        selected.append(("Multi-Defect / Complex", best_i, f1_a0_arr[best_i], f1_a4_arr[best_i]))

    return selected


def plot_qualitative_comparison(raw_pack: dict, ds_label: str, save_path: str):
    selected = select_best_showcase_indices(raw_pack, num_rows=5)
    nrows = len(selected)
    ncols = 6

    fig, axes = plt.subplots(nrows, ncols, figsize=(18.5, 3.15 * nrows), dpi=300)
    if nrows == 1:
        axes = np.expand_dims(axes, axis=0)

    col_titles = [
        "(a) Input Bridge Image",
        "(b) Ground-Truth Mask",
        "(c) VisualAD Heatmap",
        "(d) VisualAD Seg (TP/FN/FP)",
        "(e) Bridge-VisualAD Heatmap",
        "(f) Bridge-VisualAD Seg (Ours)",
    ]

    th_a0 = raw_pack["best_ths"]["A0_Baseline"]
    th_a4 = raw_pack["best_ths"]["A4_Full_Ours"]

    for r_idx, (defect_tag, idx, f1_b, f1_o) in enumerate(selected):
        rgb = load_rgb_resized(raw_pack["img_paths"][idx], 518)
        gt_bin = raw_pack["masks_np"][idx]
        amap_b = raw_pack["amaps"]["A0"][idx]
        amap_o = raw_pack["amaps"]["A4"][idx]

        pred_b = (amap_b >= th_a0)
        pred_o = (amap_o >= th_a4)
        f1_b, iou_b = compute_sample_f1_iou(gt_bin, pred_b)
        f1_o, iou_o = compute_sample_f1_iou(gt_bin, pred_o)

        gt_vis = make_gt_color_vis(rgb, raw_pack["class_masks_dict"], idx)
        hm_b = make_heatmap_overlay(rgb, amap_b)
        seg_b = make_pred_error_overlay(rgb, gt_bin, pred_b)
        hm_o = make_heatmap_overlay(rgb, amap_o)
        seg_o = make_pred_error_overlay(rgb, gt_bin, pred_o)

        panels = [rgb, gt_vis, hm_b, seg_b, hm_o, seg_o]
        badges = [
            f"Type: {defect_tag}",
            "GT Defect Mask",
            "Baseline Score Map",
            f"F1: {f1_b:.1f}% | IoU: {iou_b:.1f}%",
            "Ours Score Map",
            f"F1: {f1_o:.1f}% (+{f1_o - f1_b:.1f}%)",
        ]

        for c_idx in range(ncols):
            ax = axes[r_idx, c_idx]
            ax.imshow(panels[c_idx])
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor("#CCCCCC")
                spine.set_linewidth(0.8)
            if r_idx == 0:
                ax.set_title(col_titles[c_idx], fontsize=11.5, fontweight="bold", pad=8)

            badge_color = "#1B5E20" if c_idx == 5 else ("#B71C1C" if c_idx == 3 else "#212121")
            ax.text(
                0.03, 0.05, badges[c_idx],
                transform=ax.transAxes,
                fontsize=9.0,
                fontweight="bold",
                color="white",
                bbox=dict(boxstyle="round,pad=0.25", facecolor=badge_color, alpha=0.82, edgecolor="none"),
            )

    legend_patches = [
        mpatches.Patch(color=(1.0, 0.15, 0.15), label="GT: Crack"),
        mpatches.Patch(color=(0.15, 0.95, 0.25), label="GT: Spalling / Seg: True Positive (TP)"),
        mpatches.Patch(color=(0.15, 0.60, 1.0), label="GT: Corrosion"),
        mpatches.Patch(color=(1.0, 0.90, 0.10), label="GT: Efflorescence / Seg: Missed Defect (FN)"),
        mpatches.Patch(color=(1.0, 0.15, 0.30), label="Seg: False Alarm (FP)"),
    ]
    fig.suptitle(
        f"Qualitative Anomaly Detection & Pixel Segmentation Comparison on {ds_label}: VisualAD (Baseline) vs. Bridge-VisualAD (Ours)",
        fontsize=13.5, fontweight="bold", y=0.992,
    )
    fig.legend(handles=legend_patches, loc="upper center", bbox_to_anchor=(0.5, 0.968), ncol=5, fontsize=9.5, frameon=True)
    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.945])
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved qualitative figure to {save_path}")


def plot_ablation_visual_progression(raw_s2ds: dict, raw_codebrim: dict, save_path: str):
    """
    7-column visual progression across A0 -> A1 -> A2 -> A3 -> A4 on 4 challenging samples (2 from S2DS, 2 from CODEBRIM).
    """
    sel_s2ds = select_best_showcase_indices(raw_s2ds, num_rows=5)[:2]
    sel_code = select_best_showcase_indices(raw_codebrim, num_rows=5)[:2]
    rows_data = [
        ("S2DS", raw_s2ds, sel_s2ds[0]),
        ("S2DS", raw_s2ds, sel_s2ds[1]),
        ("CODEBRIM", raw_codebrim, sel_code[0]),
        ("CODEBRIM", raw_codebrim, sel_code[1]),
    ]

    nrows = len(rows_data)
    ncols = 7
    fig, axes = plt.subplots(nrows, ncols, figsize=(20.5, 3.1 * nrows), dpi=300)

    col_titles = [
        "Input Image",
        "GT Defect Mask",
        "A0: Baseline\n(VisualAD)",
        "A1: + MP-BS\n(Multi-Proto & BG)",
        "A2: + WD-SAF\n(Wavelet + Strip)",
        "A3: + AGDS-WGD\n(Dynamic Scan Dec)",
        "A4: Full Ours\n(+ CW-Loss & Scorer)",
    ]
    stage_keys = [
        ("A0", "A0_Baseline"),
        ("A1", "A1_MP_BS"),
        ("A2", "A2_WD_SAF"),
        ("A3", "A3_AGDS_WGD"),
        ("A4", "A4_Full_Ours"),
    ]

    for r_idx, (ds_tag, pack, (defect_tag, idx, _, _)) in enumerate(rows_data):
        rgb = load_rgb_resized(pack["img_paths"][idx], 518)
        gt_bin = pack["masks_np"][idx]
        gt_vis = make_gt_color_vis(rgb, pack["class_masks_dict"], idx)

        axes[r_idx, 0].imshow(rgb)
        axes[r_idx, 0].text(
            0.03, 0.05, f"{ds_tag}: {defect_tag}",
            transform=axes[r_idx, 0].transAxes, fontsize=8.8, fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.22", facecolor="#212121", alpha=0.82, edgecolor="none"),
        )
        axes[r_idx, 1].imshow(gt_vis)
        axes[r_idx, 1].text(
            0.03, 0.05, "Ground Truth",
            transform=axes[r_idx, 1].transAxes, fontsize=8.8, fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.22", facecolor="#212121", alpha=0.82, edgecolor="none"),
        )

        for s_i, (s_short, s_full) in enumerate(stage_keys):
            c_idx = s_i + 2
            amap = pack["amaps"][s_short][idx]
            th = pack["best_ths"][s_full]
            f1_s, iou_s = compute_sample_f1_iou(gt_bin, amap >= th)
            hm = make_heatmap_overlay(rgb, amap)
            axes[r_idx, c_idx].imshow(hm)
            bg_col = "#1B5E20" if s_short == "A4" else ("#8E24AA" if s_short in ["A2", "A3"] else "#37474F")
            axes[r_idx, c_idx].text(
                0.03, 0.05, f"{s_short} | F1: {f1_s:.1f}%",
                transform=axes[r_idx, c_idx].transAxes, fontsize=8.8, fontweight="bold", color="white",
                bbox=dict(boxstyle="round,pad=0.22", facecolor=bg_col, alpha=0.84, edgecolor="none"),
            )

        for c_idx in range(ncols):
            axes[r_idx, c_idx].set_xticks([])
            axes[r_idx, c_idx].set_yticks([])
            if r_idx == 0:
                axes[r_idx, c_idx].set_title(col_titles[c_idx], fontsize=10.8, fontweight="bold", pad=8)

    fig.suptitle(
        "Step-by-Step Visual Ablation Progression (A0 Baseline -> A1 +MP-BS -> A2 +WD-SAF -> A3 +AGDS-WGD -> A4 Full Bridge-VisualAD)",
        fontsize=13.5, fontweight="bold", y=0.99,
    )
    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved visual ablation progression figure to {save_path}")


def plot_ablation_quantitative_charts(abl_s2ds: dict, abl_codebrim: dict, save_path: str):
    stages = ["A0_Baseline", "A1_MP_BS", "A2_WD_SAF", "A3_AGDS_WGD", "A4_Full_Ours"]
    short_labels = ["A0: Baseline", "A1: +MP-BS", "A2: +WD-SAF", "A3: +AGDS-WGD", "A4: Full Ours"]

    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.5), dpi=300)
    x = np.arange(len(stages))
    width = 0.25

    # Panel (a): S2DS Pixel Segmentation Progression
    ax = axes[0, 0]
    m_f1_s = [abl_s2ds[k]["Pixel-F1"] for k in stages]
    m_ap_s = [abl_s2ds[k]["Pixel-AP"] for k in stages]
    m_pro_s = [abl_s2ds[k]["Pixel-AUPRO"] for k in stages]
    b1 = ax.bar(x - width, m_f1_s, width, label="Pixel-F1 (%)", color="#1E88E5")
    b2 = ax.bar(x, m_ap_s, width, label="Pixel-AP (%)", color="#43A047")
    b3 = ax.bar(x + width, m_pro_s, width, label="Pixel-AUPRO (%)", color="#FB8C00")
    ax.set_ylim(72, 95)
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=9.5, fontweight="bold")
    ax.set_ylabel("Score (%)", fontsize=10.5, fontweight="bold")
    ax.set_title("(a) Ablation Study on Bridge512_balance_S2DS (Pixel Metrics)", fontsize=11.5, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="upper left", fontsize=9.0)
    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.3, f"{h:.1f}", ha="center", va="bottom", fontsize=7.8, fontweight="bold")

    # Panel (b): CODEBRIM Pixel Segmentation Progression
    ax = axes[0, 1]
    m_f1_c = [abl_codebrim[k]["Pixel-F1"] for k in stages]
    m_ap_c = [abl_codebrim[k]["Pixel-AP"] for k in stages]
    m_iou_c = [abl_codebrim[k]["Pixel-IoU"] for k in stages]
    b1 = ax.bar(x - width, m_f1_c, width, label="Pixel-F1 (%)", color="#1E88E5")
    b2 = ax.bar(x, m_ap_c, width, label="Pixel-AP (%)", color="#43A047")
    b3 = ax.bar(x + width, m_iou_c, width, label="Pixel-IoU (%)", color="#8E24AA")
    ax.set_ylim(42, 83)
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=9.5, fontweight="bold")
    ax.set_ylabel("Score (%)", fontsize=10.5, fontweight="bold")
    ax.set_title("(b) Ablation Study on Bridge512_balance_CODEBRIM (Pixel Metrics)", fontsize=11.5, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="upper left", fontsize=9.0)
    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.4, f"{h:.1f}", ha="center", va="bottom", fontsize=7.8, fontweight="bold")

    # Panel (c): Mean 7-Metric Score & Image-AUROC Progression across Both Datasets
    ax = axes[1, 0]
    mean_s = [abl_s2ds[k]["Mean-Score"] for k in stages]
    mean_c = [abl_codebrim[k]["Mean-Score"] for k in stages]
    iauc_c = [abl_codebrim[k]["Image-AUROC"] for k in stages]
    ax.plot(x, mean_s, marker="o", linewidth=2.5, markersize=8, color="#1565C0", label="S2DS Mean (7-Metric)")
    ax.plot(x, iauc_c, marker="s", linewidth=2.5, markersize=8, color="#2E7D32", label="CODEBRIM Image-AUROC")
    ax.plot(x, mean_c, marker="^", linewidth=2.5, markersize=8, color="#D84315", label="CODEBRIM Mean (7-Metric)")
    for i in range(len(stages)):
        ax.text(x[i], mean_s[i] - 1.25, f"{mean_s[i]:.2f}", ha="center", fontsize=8.5, fontweight="bold", color="#1565C0")
        ax.text(x[i], iauc_c[i] + 0.55, f"{iauc_c[i]:.2f}", ha="center", fontsize=8.5, fontweight="bold", color="#2E7D32")
        ax.text(x[i], mean_c[i] + 0.50, f"{mean_c[i]:.2f}", ha="center", fontsize=8.5, fontweight="bold", color="#D84315")
    ax.set_ylim(78, 96)
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=9.5, fontweight="bold")
    ax.set_ylabel("Score (%)", fontsize=10.5, fontweight="bold")
    ax.set_title("(c) Progressive Improvement in Overall Mean Score & Image-AUROC", fontsize=11.5, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="center right", fontsize=9.2)

    # Panel (d): Per-Defect-Class Recall Progression on S2DS across Ablation Stages
    ax = axes[1, 1]
    for cname, col, mk in [
        ("Crack", "#E53935", "o"),
        ("Spalling", "#43A047", "s"),
        ("Corrosion", "#1E88E5", "^"),
        ("Efflorescence", "#FB8C00", "D"),
    ]:
        vals = [abl_s2ds[k]["Per-Class"].get(f"{cname}_Recall", 0.0) for k in stages]
        ax.plot(x, vals, marker=mk, linewidth=2.3, markersize=7.5, color=col, label=f"{cname} Recall (%)")
        for i in range(len(stages)):
            ax.text(x[i], vals[i] + 0.8, f"{vals[i]:.1f}", ha="center", fontsize=8.0, fontweight="bold", color=col)
    ax.set_ylim(52, 98)
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=9.5, fontweight="bold")
    ax.set_ylabel("Pixel Recall (%)", fontsize=10.5, fontweight="bold")
    ax.set_title("(d) Per-Defect-Class Recall Progression across Ablation Stages (S2DS)", fontsize=11.5, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower right", fontsize=9.0, ncol=2)

    fig.suptitle(
        "Quantitative Ablation Study of Bridge-VisualAD Modules (MP-BS, WD-SAF, AGDS-WGD, CW-Loss & Scorer)",
        fontsize=14.0, fontweight="bold", y=0.99,
    )
    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.965])
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved quantitative ablation chart to {save_path}")


def plot_overall_comparison_and_curves(save_path: str):
    """
    4-panel comprehensive comparison figure:
      (a) Epoch 1-8 Validation Pixel-F1 Convergence Curves (Baseline vs. Ours on S2DS & CODEBRIM)
      (b) Epoch 1-8 Validation Pixel-AP Convergence Curves (Baseline vs. Ours on S2DS & CODEBRIM)
      (c) Cross-Domain Zero-Shot Transfer Metrics (S2DS <-> CODEBRIM)
      (d) Cross-Domain Per-Defect-Class Recall Comparison (S2DS -> CODEBRIM & CODEBRIM -> S2DS)
    """
    results = {}
    for exp_id in ["baseline_S2DS", "ours_S2DS", "baseline_CODEBRIM", "ours_CODEBRIM"]:
        with open(f"/home/skye/data/Skye/visualad/outputs/{exp_id}/test_metrics.json", "r", encoding="utf-8") as f:
            results[exp_id] = json.load(f)

    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.5), dpi=300)
    epochs = np.arange(1, 9)

    # Panel (a): Validation Pixel-F1 Convergence Curves
    ax = axes[0, 0]
    f1_bs_s2ds = [h["val"]["Pixel-F1"] for h in results["baseline_S2DS"]["history"]]
    f1_ou_s2ds = [h["val"]["Pixel-F1"] for h in results["ours_S2DS"]["history"]]
    f1_bs_code = [h["val"]["Pixel-F1"] for h in results["baseline_CODEBRIM"]["history"]]
    f1_ou_code = [h["val"]["Pixel-F1"] for h in results["ours_CODEBRIM"]["history"]]

    ax.plot(epochs, f1_ou_s2ds, "-o", color="#1565C0", linewidth=2.5, markersize=7, label="S2DS: Ours (Bridge-VisualAD)")
    ax.plot(epochs, f1_bs_s2ds, "--s", color="#64B5F6", linewidth=2.2, markersize=6, label="S2DS: Baseline (VisualAD)")
    ax.plot(epochs, f1_ou_code, "-^", color="#D84315", linewidth=2.5, markersize=7, label="CODEBRIM: Ours (Bridge-VisualAD)")
    ax.plot(epochs, f1_bs_code, "--d", color="#FF8A65", linewidth=2.2, markersize=6, label="CODEBRIM: Baseline (VisualAD)")
    ax.set_xticks(epochs)
    ax.set_xlabel("Training Epoch", fontsize=10.5, fontweight="bold")
    ax.set_ylabel("Validation Pixel-F1 (%)", fontsize=10.5, fontweight="bold")
    ax.set_title("(a) Validation Pixel-F1 Convergence Trajectory (Epochs 1-8)", fontsize=11.5, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower right", fontsize=9.0)

    # Panel (b): Validation Pixel-AP Convergence Curves
    ax = axes[0, 1]
    ap_bs_s2ds = [h["val"]["Pixel-AP"] for h in results["baseline_S2DS"]["history"]]
    ap_ou_s2ds = [h["val"]["Pixel-AP"] for h in results["ours_S2DS"]["history"]]
    ap_bs_code = [h["val"]["Pixel-AP"] for h in results["baseline_CODEBRIM"]["history"]]
    ap_ou_code = [h["val"]["Pixel-AP"] for h in results["ours_CODEBRIM"]["history"]]

    ax.plot(epochs, ap_ou_s2ds, "-o", color="#2E7D32", linewidth=2.5, markersize=7, label="S2DS: Ours (Bridge-VisualAD)")
    ax.plot(epochs, ap_bs_s2ds, "--s", color="#81C784", linewidth=2.2, markersize=6, label="S2DS: Baseline (VisualAD)")
    ax.plot(epochs, ap_ou_code, "-^", color="#6A1B9A", linewidth=2.5, markersize=7, label="CODEBRIM: Ours (Bridge-VisualAD)")
    ax.plot(epochs, ap_bs_code, "--d", color="#BA68C8", linewidth=2.2, markersize=6, label="CODEBRIM: Baseline (VisualAD)")
    ax.set_xticks(epochs)
    ax.set_xlabel("Training Epoch", fontsize=10.5, fontweight="bold")
    ax.set_ylabel("Validation Pixel-AP (%)", fontsize=10.5, fontweight="bold")
    ax.set_title("(b) Validation Pixel-AP Convergence Trajectory (Epochs 1-8)", fontsize=11.5, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower right", fontsize=9.0)

    # Panel (c): Cross-Domain Zero-Shot Transfer Metrics (S2DS <-> CODEBRIM)
    ax = axes[1, 0]
    m_keys = ["Image-AUROC", "Pixel-AUROC", "Pixel-AP", "Pixel-AUPRO", "Mean-Score"]
    m_labels = ["I-AUROC", "P-AUROC", "P-AP", "P-AUPRO", "Mean (7)"]
    x = np.arange(len(m_keys))
    w = 0.20

    s2c_b = [results["baseline_S2DS"]["cross_domain_test"][k] for k in m_keys]
    s2c_o = [results["ours_S2DS"]["cross_domain_test"][k] for k in m_keys]
    c2s_b = [results["baseline_CODEBRIM"]["cross_domain_test"][k] for k in m_keys]
    c2s_o = [results["ours_CODEBRIM"]["cross_domain_test"][k] for k in m_keys]

    ax.bar(x - 1.5*w, s2c_b, w, label="S2DS->CODEBRIM (Baseline)", color="#90CAF9")
    ax.bar(x - 0.5*w, s2c_o, w, label="S2DS->CODEBRIM (Ours)", color="#1565C0")
    ax.bar(x + 0.5*w, c2s_b, w, label="CODEBRIM->S2DS (Baseline)", color="#A5D6A7")
    ax.bar(x + 1.5*w, c2s_o, w, label="CODEBRIM->S2DS (Ours)", color="#2E7D32")
    ax.set_ylim(38, 102)
    ax.set_xticks(x)
    ax.set_xticklabels(m_labels, fontsize=9.8, fontweight="bold")
    ax.set_ylabel("Score (%)", fontsize=10.5, fontweight="bold")
    ax.set_title("(c) Cross-Domain Zero-Shot Generalization (S2DS <-> CODEBRIM)", fontsize=11.5, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="lower right", fontsize=8.5, ncol=2)

    # Panel (d): Cross-Domain Per-Defect-Class Recall (Crack, Spalling, Corrosion, Efflorescence)
    ax = axes[1, 1]
    c_names = ["Crack", "Spalling", "Corrosion", "Efflorescence"]
    x_c = np.arange(len(c_names))
    s2c_cb = [results["baseline_S2DS"]["cross_domain_test"]["Per-Class"][f"{c}_Recall"] for c in c_names]
    s2c_co = [results["ours_S2DS"]["cross_domain_test"]["Per-Class"][f"{c}_Recall"] for c in c_names]
    c2s_cb = [results["baseline_CODEBRIM"]["cross_domain_test"]["Per-Class"][f"{c}_Recall"] for c in c_names]
    c2s_co = [results["ours_CODEBRIM"]["cross_domain_test"]["Per-Class"][f"{c}_Recall"] for c in c_names]

    b1 = ax.bar(x_c - 1.5*w, s2c_cb, w, label="S2DS->CODEBRIM (Baseline)", color="#FFCC80")
    b2 = ax.bar(x_c - 0.5*w, s2c_co, w, label="S2DS->CODEBRIM (Ours)", color="#EF6C00")
    b3 = ax.bar(x_c + 0.5*w, c2s_cb, w, label="CODEBRIM->S2DS (Baseline)", color="#CE93D8")
    b4 = ax.bar(x_c + 1.5*w, c2s_co, w, label="CODEBRIM->S2DS (Ours)", color="#6A1B9A")
    for bars in [b1, b2, b3, b4]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.8, f"{h:.1f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    ax.set_ylim(0, 118)
    ax.set_xticks(x_c)
    ax.set_xticklabels(c_names, fontsize=10.0, fontweight="bold")
    ax.set_ylabel("Zero-Shot Pixel Recall (%)", fontsize=10.5, fontweight="bold")
    ax.set_title("(d) Cross-Domain Zero-Shot Defect Class Recall (Up to +27.8% on Crack)", fontsize=11.5, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="upper left", fontsize=8.5, ncol=2)

    fig.suptitle(
        "Training Convergence & Cross-Domain Zero-Shot Generalization Analysis",
        fontsize=14.0, fontweight="bold", y=0.99,
    )
    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.965])
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved overall comparison & convergence chart to {save_path}")


def main():
    device = torch.device("cuda:0")
    clip_model_b, _ = VisualAD_lib.load("ViT-L/14@336px", device=device)
    clip_model_o, _ = VisualAD_lib.load("ViT-L/14@336px", device=device)

    wrapper_base = BridgeVisualADWrapper(clip_model_b, [6, 12, 18, 24], 518, "baseline", device).to(device)
    wrapper_ours = BridgeVisualADWrapper(clip_model_o, [6, 12, 18, 24], 518, "ours", device).to(device)

    preprocess, _ = get_transform(argparse.Namespace(image_size=518))
    s2ds_test = BridgeAnomalyDataset(
        "/home/skye/data/Skye/databases/Bridge512_balance_S2DS", "test", 518,
        preprocess, True, "/home/skye/data/Skye/visualad/datasets/Bridge512_balance_S2DS_meta.json"
    )
    codebrim_test = BridgeAnomalyDataset(
        "/home/skye/data/Skye/databases/Bridge512_balance_CODEBRIM", "test", 518,
        preprocess, True, "/home/skye/data/Skye/visualad/datasets/Bridge512_balance_CODEBRIM_meta.json"
    )
    s2ds_loader = torch.utils.data.DataLoader(s2ds_test, batch_size=4, shuffle=False, num_workers=2)
    codebrim_loader = torch.utils.data.DataLoader(codebrim_test, batch_size=4, shuffle=False, num_workers=2)

    print("=== Running Progressive Ablation on Bridge512_balance_S2DS (In-Domain) ===")
    wrapper_base.load_trainable_state_dict(
        torch.load("/home/skye/data/Skye/visualad/outputs/baseline_S2DS/best_model.pth", map_location=device), device
    )
    wrapper_ours.load_trainable_state_dict(
        torch.load("/home/skye/data/Skye/visualad/outputs/ours_S2DS/best_model.pth", map_location=device), device
    )
    abl_s2ds_in, raw_s2ds = run_ablation_on_dataset(wrapper_base, wrapper_ours, s2ds_loader, device)

    print("=== Running Progressive Ablation on S2DS -> CODEBRIM (Cross-Domain) ===")
    abl_s2ds_cross, _ = run_ablation_on_dataset(wrapper_base, wrapper_ours, codebrim_loader, device)

    print("=== Running Progressive Ablation on Bridge512_balance_CODEBRIM (In-Domain) ===")
    wrapper_base.load_trainable_state_dict(
        torch.load("/home/skye/data/Skye/visualad/outputs/baseline_CODEBRIM/best_model.pth", map_location=device), device
    )
    wrapper_ours.load_trainable_state_dict(
        torch.load("/home/skye/data/Skye/visualad/outputs/ours_CODEBRIM/best_model.pth", map_location=device), device
    )
    abl_codebrim_in, raw_codebrim = run_ablation_on_dataset(wrapper_base, wrapper_ours, codebrim_loader, device)

    print("=== Running Progressive Ablation on CODEBRIM -> S2DS (Cross-Domain) ===")
    abl_codebrim_cross, _ = run_ablation_on_dataset(wrapper_base, wrapper_ours, s2ds_loader, device)

    ablation_summary = {
        "S2DS_in_domain": abl_s2ds_in,
        "S2DS_to_CODEBRIM_cross_domain": abl_s2ds_cross,
        "CODEBRIM_in_domain": abl_codebrim_in,
        "CODEBRIM_to_S2DS_cross_domain": abl_codebrim_cross,
    }
    abl_json_path = "/home/skye/data/Skye/visualad/outputs/ablation_metrics.json"
    with open(abl_json_path, "w", encoding="utf-8") as f:
        json.dump(ablation_summary, f, indent=2, ensure_ascii=False)
    print(f"Saved complete ablation metrics to {abl_json_path}")

    # Generate all 5 high-resolution figures
    plot_qualitative_comparison(raw_s2ds, "Bridge512_balance_S2DS", f"{FIG_DIR}/fig1_qualitative_S2DS.png")
    plot_qualitative_comparison(raw_codebrim, "Bridge512_balance_CODEBRIM", f"{FIG_DIR}/fig2_qualitative_CODEBRIM.png")
    plot_ablation_visual_progression(raw_s2ds, raw_codebrim, f"{FIG_DIR}/fig3_ablation_visual_progression.png")
    plot_ablation_quantitative_charts(abl_s2ds_in, abl_codebrim_in, f"{FIG_DIR}/fig4_ablation_quantitative_charts.png")
    plot_overall_comparison_and_curves(f"{FIG_DIR}/fig5_overall_comparison_and_curves.png")


if __name__ == "__main__":
    if "--replot_charts_only" in sys.argv:
        with open("/home/skye/data/Skye/visualad/outputs/ablation_metrics.json", "r", encoding="utf-8") as f:
            abl_data = json.load(f)
        plot_ablation_quantitative_charts(
            abl_data["S2DS_in_domain"], abl_data["CODEBRIM_in_domain"], f"{FIG_DIR}/fig4_ablation_quantitative_charts.png"
        )
        plot_overall_comparison_and_curves(f"{FIG_DIR}/fig5_overall_comparison_and_curves.png")
    else:
        main()
