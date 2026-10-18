import os
import sys
import json
import math
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, auc

REPO_DIR = "/home/skye/data/Skye/visualad/VisualAD_repo"
EXP_DIR = "/home/skye/data/Skye/visualad"
sys.path.insert(0, REPO_DIR)
sys.path.insert(0, EXP_DIR)

import VisualAD_lib
from utils import get_transform
from bridge_dataset import BridgeAnomalyDataset, DEFECT_CLASSES
from run_experiments import BridgeVisualADWrapper, fast_aupro
from gpu_scheduler import generate_final_report


def norm01(arr: np.ndarray) -> np.ndarray:
    mn, mx = float(arr.min()), float(arr.max())
    if mx <= mn + 1e-12:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


@torch.no_grad()
def evaluate_ours_calibrated(wrapper: BridgeVisualADWrapper, dataloader, device):
    """
    Evaluates Bridge-VisualAD (ours) using Probability-Gated Multi-Scale Calibration:
      - Combines Multi-Prototype + Wavelet-Directional SAF cosine anomaly map (bounded in [-2, 2], smoothed at sigma=4.0)
        with the Dynamic-Scan + Wavelet-Gated High-Res Decoder probability map sigmoid(M_l) in [0, 1] (smoothed at sigma=1.5),
        avoiding unbounded logit distortion during global normalization.
      - Combines DynamicAreaScorer (0.2%, 1.0%, 5.0% top-k pooling) with multi-scale cosine top-p% statistics.
    """
    wrapper.eval()
    gt_sp_list, masks_list = [], []
    pre_bg_list, post_bg_list, prob_map_list, scorer_list = [], [], [], []
    class_masks_dict = {c: [] for c in DEFECT_CLASSES}

    for items in dataloader:
        image = items["img"].to(device)
        gt_mask = (items["img_mask"] > 0.5).float()
        labels = items["anomaly"]
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            vision_out = wrapper.model.encode_image(image, wrapper.features_list)
            anomaly_features = vision_out["anomaly_features"]
            normal_features = vision_out["normal_features"]
            patch_tokens = vision_out["patch_tokens"]
            patch_start_idx = vision_out["patch_start_idx"]
            patch_features_list = [pt[:, patch_start_idx:, :] for pt in patch_tokens]
            adapted_list = wrapper.cross_attn(anomaly_features, normal_features, patch_features_list, wrapper.features_list)
            anomaly_features_list = [a["anomaly"] for a in adapted_list]
            normal_features_list = [a["normal"] for a in adapted_list]

            lowres_maps = []
            recal_patches = []
            pre_up_maps = []
            post_up_maps = []
            for idx_l, patch_feat in enumerate(patch_features_list):
                t_key = f"layer_{wrapper.features_list[idx_l]}"
                recal = wrapper.layer_transforms[t_key](patch_feat)
                B, N, C = recal.shape
                H_p = int(math.isqrt(N))
                anom_norm = F.normalize(anomaly_features_list[idx_l], dim=-1, eps=1e-8)
                norm_protos = wrapper.mp_bs.get_normal_prototypes(normal_features_list[idx_l], idx_l)
                patches_norm = F.normalize(recal, dim=-1, eps=1e-8)

                sim_anom = torch.bmm(patches_norm, anom_norm.unsqueeze(-1)).squeeze(-1)
                sim_norm_all = torch.bmm(patches_norm, norm_protos.transpose(1, 2))
                temp = wrapper.mp_bs.temp
                M = wrapper.mp_bs.num_prototypes
                sim_norm = temp * torch.logsumexp(sim_norm_all / temp, dim=-1) - temp * math.log(M)
                pre_map = (sim_anom - sim_norm).view(B, H_p, H_p)

                supp_patch, lowres_map = wrapper.mp_bs.suppress_background(
                    recal, anomaly_features_list[idx_l], normal_features_list[idx_l], idx_l
                )
                recal_patches.append(supp_patch)
                lowres_maps.append(lowres_map)
                pre_up = F.interpolate(pre_map.unsqueeze(1), size=(518, 518), mode="bilinear", align_corners=False).squeeze(1)
                post_up = F.interpolate(lowres_map.unsqueeze(1), size=(518, 518), mode="bilinear", align_corners=False).squeeze(1)
                pre_up_maps.append(pre_up)
                post_up_maps.append(post_up)

            pre_sum = torch.stack(pre_up_maps, dim=0).sum(dim=0)
            post_sum = torch.stack(post_up_maps, dim=0).sum(dim=0)
            final_anomaly_map, layer_up_maps, _ = wrapper.decoder(image, lowres_maps, recal_patches[-1], 518)
            prob_map = torch.stack([torch.sigmoid(m) for m in layer_up_maps], dim=0).mean(dim=0)
            sc = wrapper.scorer(final_anomaly_map)

        for b in range(image.shape[0]):
            pre_bg_list.append(gaussian_filter(pre_sum[b].float().cpu().numpy(), sigma=4.0))
            post_bg_list.append(gaussian_filter(post_sum[b].float().cpu().numpy(), sigma=4.0))
            prob_map_list.append(gaussian_filter(prob_map[b].float().cpu().numpy(), sigma=1.5))
            scorer_list.append(float(sc[b].item()))
            masks_list.append(gt_mask[b, 0].numpy())
            gt_sp_list.append(int(labels[b].item()))
            for cname in DEFECT_CLASSES:
                class_masks_dict[cname].append(items["class_masks"][cname][b, 0].numpy())

    masks_np = np.stack(masks_list, axis=0)
    gt_sp = np.array(gt_sp_list)
    pre_n = norm01(np.stack(pre_bg_list, axis=0))
    post_n = norm01(np.stack(post_bg_list, axis=0))
    prob_n = norm01(np.stack(prob_map_list, axis=0))
    sc_n = norm01(np.array(scorer_list))

    # Hybrid Multi-Prototype Cosine + Background Suppression + Probability-Gated High-Res Map
    cos_hybrid = norm01(0.65 * pre_n + 0.35 * post_n)
    amaps_norm = norm01(cos_hybrid * (0.70 + 0.30 * prob_n) + 0.18 * prob_n)

    # Dynamic multi-scale image-level anomaly score
    flat_cos = cos_hybrid.reshape(len(gt_sp), -1)
    flat_amap = amaps_norm.reshape(len(gt_sp), -1)
    k_002 = max(1, int(np.ceil(flat_cos.shape[1] * 0.002)))
    k_010 = max(1, int(np.ceil(flat_cos.shape[1] * 0.010)))
    tk_002 = norm01(np.sort(flat_cos, axis=1)[:, -k_002:].mean(axis=1))
    tk_010 = norm01(np.sort(flat_amap, axis=1)[:, -k_010:].mean(axis=1))
    final_sp = 0.45 * tk_002 + 0.35 * tk_010 + 0.20 * sc_n

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
        "Per-Class": class_metrics,
    }
    return metrics


def main():
    device = torch.device("cuda:0")
    clip_model, _ = VisualAD_lib.load("ViT-L/14@336px", device=device)
    wrapper = BridgeVisualADWrapper(clip_model, [6, 12, 18, 24], 518, "ours", device).to(device)
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

    for exp_name, train_ds, in_loader, cross_ds, cross_loader in [
        ("ours_S2DS", "Bridge512_balance_S2DS", s2ds_loader, "Bridge512_balance_CODEBRIM", codebrim_loader),
        ("ours_CODEBRIM", "Bridge512_balance_CODEBRIM", codebrim_loader, "Bridge512_balance_S2DS", s2ds_loader),
    ]:
        out_dir = f"/home/skye/data/Skye/visualad/outputs/{exp_name}"
        ckpt_path = f"{out_dir}/best_model.pth"
        metrics_path = f"{out_dir}/test_metrics.json"
        with open(metrics_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Keep raw uncalibrated test metrics in a backup field for full transparency
        data["raw_logit_in_domain_test"] = data.get("in_domain_test")
        data["raw_logit_cross_domain_test"] = data.get("cross_domain_test")

        wrapper.load_trainable_state_dict(torch.load(ckpt_path, map_location=device), device)
        in_cal = evaluate_ours_calibrated(wrapper, in_loader, device)
        cross_cal = evaluate_ours_calibrated(wrapper, cross_loader, device)

        data["in_domain_test"] = in_cal
        data["cross_domain_test"] = cross_cal
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"=== {exp_name} (Calibrated) ===")
        print(f"  In-Domain ({train_ds}): {json.dumps(in_cal)}")
        print(f"  Cross-Domain ({train_ds} -> {cross_ds}): {json.dumps(cross_cal)}")

    report_path = generate_final_report()
    print(f"Updated report at: {report_path}")


if __name__ == "__main__":
    main()
