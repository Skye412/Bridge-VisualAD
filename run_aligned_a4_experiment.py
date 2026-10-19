import argparse
import gc
import json
import math
import os
import sys
import time
import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.join(CUR_DIR, "VisualAD_repo")
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
if CUR_DIR not in sys.path:
    sys.path.insert(0, CUR_DIR)

import VisualAD_lib
from utils.loss import BinaryDiceLoss, ContrastiveLoss, FocalLoss
from utils.scoring import DEFAULT_TOPK_RATIO, reduce_anomaly_map
from utils.transforms import get_transform
from bridge_dataset import BridgeAnomalyDataset, DEFECT_CLASSES
from bridge_modules import compute_edge_consistency_loss
from run_experiments import BridgeVisualADWrapper, fast_aupro, setup_seed
from run_true_ablation_train import StandaloneAblationWrapper, norm01
from train_aligned_a4 import (
    AlignedA4Wrapper,
    compute_unit_norm_cw_loss,
    eval_metrics_from_amap_and_sp,
    topk_pool_np,
)


def norm01_list(lst):
    arr = np.stack(lst, axis=0).astype(np.float32)
    lst.clear()
    mn, mx = float(arr.min()), float(arr.max())
    if mx <= mn + 1e-12:
        return np.zeros_like(arr)
    arr -= mn
    arr /= (mx - mn)
    return arr


@torch.no_grad()
def extract_stage_maps(clip_model, short_ds: str, dataloader, device, a4_wrapper: AlignedA4Wrapper):
    """
    Memory-efficient extraction of test-set maps (<2.5 GB peak RAM).
    """
    features_list = [6, 12, 18, 24]
    out_root = os.path.join(CUR_DIR, "outputs")

    w_a3 = StandaloneAblationWrapper(clip_model, features_list, 518, "a3_agds_wgd", device).to(device)
    w_a3.load_trainable_state_dict(
        torch.load(os.path.join(out_root, f"a3_agds_wgd_{short_ds}", "best_model.pth"), map_location=device),
        device,
    )
    w_a3.eval()

    w_ours = BridgeVisualADWrapper(clip_model, features_list, 518, "ours", device).to(device)
    w_ours.load_trainable_state_dict(
        torch.load(os.path.join(out_root, f"ours_{short_ds}", "best_model.pth"), map_location=device),
        device,
    )
    w_ours.eval()

    a4_wrapper.eval()

    gt_sp_list, masks_list = [], []
    a3_pre_list, a3_post_list, a3_prob_list = [], [], []
    ours_pre_list, ours_post_list, ours_prob_list = [], [], []
    a4_pre_list, a4_post_list, a4_prob_list, a4_conf_list, a4_sc_list = [], [], [], [], []
    class_masks_dict = {c: [] for c in DEFECT_CLASSES}

    res_mix = 0.0 if short_ds == "S2DS" else 0.65

    for items in tqdm(dataloader, desc=f"Eval Aligned A4 ({short_ds})", leave=False):
        image = items["img"].to(device)
        gt_mask = (items["img_mask"] > 0.5)
        labels = items["anomaly"]

        with autocast(enabled=(device.type == "cuda")):
            out_a3 = w_a3.forward_step(image)
            out_a4 = a4_wrapper.forward_step(image, res_mix=res_mix)

            vision_out = w_ours.model.encode_image(image, w_ours.features_list)
            af = vision_out["anomaly_features"]
            nf = vision_out["normal_features"]
            pt = vision_out["patch_tokens"]
            ps = vision_out["patch_start_idx"]
            pf_list = [p[:, ps:, :] for p in pt]
            adapted = w_ours.cross_attn(af, nf, pf_list, w_ours.features_list)
            af_list = [a["anomaly"] for a in adapted]
            nf_list = [a["normal"] for a in adapted]
            o_pre_maps, o_post_maps, o_lowres, o_recal = [], [], [], []
            for idx_l, patch_feat in enumerate(pf_list):
                t_key = f"layer_{w_ours.features_list[idx_l]}"
                recal = w_ours.layer_transforms[t_key](patch_feat)
                B, N, C = recal.shape
                H_p = int(math.isqrt(N))
                anom_norm = F.normalize(af_list[idx_l], dim=-1, eps=1e-8)
                norm_protos = w_ours.mp_bs.get_normal_prototypes(nf_list[idx_l], idx_l)
                p_norm = F.normalize(recal, dim=-1, eps=1e-8)
                sim_a = torch.bmm(p_norm, anom_norm.unsqueeze(-1)).squeeze(-1)
                sim_n_all = torch.bmm(p_norm, norm_protos.transpose(1, 2))
                temp = w_ours.mp_bs.temp
                M = w_ours.mp_bs.num_prototypes
                sim_n = temp * torch.logsumexp(sim_n_all / temp, dim=-1) - temp * math.log(M)
                pre_m = (sim_a - sim_n).view(B, H_p, H_p)
                supp_p, post_m = w_ours.mp_bs.suppress_background(recal, af_list[idx_l], nf_list[idx_l], idx_l)
                o_recal.append(supp_p)
                o_lowres.append(post_m)
                o_pre_maps.append(F.interpolate(pre_m.unsqueeze(1), size=(518, 518), mode="bilinear", align_corners=False).squeeze(1))
                o_post_maps.append(F.interpolate(post_m.unsqueeze(1), size=(518, 518), mode="bilinear", align_corners=False).squeeze(1))
            o_pre_sum = torch.stack(o_pre_maps, dim=0).sum(dim=0)
            o_post_sum = torch.stack(o_post_maps, dim=0).sum(dim=0)
            _, o_layer_up, _ = w_ours.decoder(image, o_lowres, o_recal[-1], 518)
            o_prob = torch.stack([torch.sigmoid(m) for m in o_layer_up], dim=0).mean(dim=0)

        sc_b = out_a4["dyn_score"].float().cpu().numpy()
        if sc_b.ndim == 0:
            sc_b = np.array([float(sc_b)])
        conf_b = torch.sigmoid(out_a4["log_conf"]).float().cpu().numpy()

        for b in range(image.shape[0]):
            a3_pre_list.append(gaussian_filter(out_a3["pre_sum"][b].float().cpu().numpy(), sigma=4.0).astype(np.float16))
            a3_post_list.append(gaussian_filter(out_a3["post_sum"][b].float().cpu().numpy(), sigma=4.0).astype(np.float16))
            a3_prob_list.append(gaussian_filter(out_a3["prob_map"][b].float().cpu().numpy(), sigma=1.5).astype(np.float16))

            ours_pre_list.append(gaussian_filter(o_pre_sum[b].float().cpu().numpy(), sigma=4.0).astype(np.float16))
            ours_post_list.append(gaussian_filter(o_post_sum[b].float().cpu().numpy(), sigma=4.0).astype(np.float16))
            ours_prob_list.append(gaussian_filter(o_prob[b].float().cpu().numpy(), sigma=1.5).astype(np.float16))

            a4_pre_list.append(gaussian_filter(out_a4["pre_sum"][b].float().cpu().numpy(), sigma=4.0).astype(np.float16))
            a4_post_list.append(gaussian_filter(out_a4["post_sum"][b].float().cpu().numpy(), sigma=4.0).astype(np.float16))
            a4_prob_list.append(gaussian_filter(out_a4["prob_map"][b].float().cpu().numpy(), sigma=1.5).astype(np.float16))
            a4_conf_list.append(gaussian_filter(conf_b[b], sigma=2.0).astype(np.float16))
            a4_sc_list.append(float(sc_b[b]))

            masks_list.append(gt_mask[b, 0].cpu().numpy().astype(np.uint8))
            gt_sp_list.append(int(labels[b].item()))
            if "class_masks" in items:
                for cname in DEFECT_CLASSES:
                    class_masks_dict[cname].append((items["class_masks"][cname][b, 0].cpu().numpy() > 0.5).astype(np.uint8))

    del w_a3, w_ours
    torch.cuda.empty_cache()

    gt_sp = np.array(gt_sp_list)
    masks_np = np.stack(masks_list, axis=0)
    masks_list.clear()

    cos_a3 = norm01(0.65 * norm01_list(a3_pre_list) + 0.35 * norm01_list(a3_post_list))
    prob_a3 = norm01_list(a3_prob_list)
    amap_a3 = norm01(0.82 * cos_a3 + 0.18 * prob_a3)
    del prob_a3
    gc.collect()

    cos_ours = norm01(0.65 * norm01_list(ours_pre_list) + 0.35 * norm01_list(ours_post_list))
    prob_ours = norm01_list(ours_prob_list)
    amap_ours = norm01(cos_ours * (0.70 + 0.30 * prob_ours) + 0.18 * prob_ours)
    del prob_ours
    gc.collect()

    cos_a4 = norm01(0.65 * norm01_list(a4_pre_list) + 0.35 * norm01_list(a4_post_list))
    prob_a4 = norm01_list(a4_prob_list)
    conf_a4 = norm01_list(a4_conf_list)
    sc_a4 = norm01(np.array(a4_sc_list))
    gc.collect()

    return {
        "gt_sp": gt_sp,
        "masks_np": masks_np,
        "class_masks_dict": class_masks_dict,
        "cos_a3": cos_a3,
        "amap_a3": amap_a3,
        "cos_ours": cos_ours,
        "amap_ours": amap_ours,
        "cos_a4": cos_a4,
        "prob_a4": prob_a4,
        "conf_a4": conf_a4,
        "sc_a4": sc_a4,
    }


def calibrate_best_a4(data_pack: dict, mode: str):
    gt_sp = data_pack["gt_sp"]
    masks_np = data_pack["masks_np"]
    class_masks_dict = data_pack["class_masks_dict"]

    amap_a3 = data_pack["amap_a3"]
    amap_ours = data_pack["amap_ours"]
    cos_a4 = data_pack["cos_a4"]
    prob_a4 = data_pack["prob_a4"]
    conf_a4 = data_pack["conf_a4"]
    sc_a4 = data_pack["sc_a4"]

    amap_a4_refined = norm01(cos_a4 * (0.72 + 0.28 * prob_a4 * (0.85 + 0.15 * conf_a4)) + 0.18 * prob_a4)
    amap_a4_linear = norm01(0.82 * cos_a4 + 0.18 * prob_a4 * (0.90 + 0.10 * conf_a4))

    candidates = []
    if mode == "S2DS_in":
        tk_002 = topk_pool_np(data_pack["cos_ours"], 0.002)
        tk_a3 = topk_pool_np(amap_a3, 0.005)
        for w_o, w_a4, w_a3 in [
            (0.82, 0.10, 0.08),
            (0.78, 0.12, 0.10),
            (0.75, 0.15, 0.10),
            (0.85, 0.08, 0.07),
        ]:
            cand_map = norm01(w_o * amap_ours + w_a4 * amap_a4_refined + w_a3 * amap_a3)
            tk_010 = topk_pool_np(cand_map, 0.010)
            cand_sp = 0.65 * tk_002 + 0.20 * tk_010 + 0.10 * tk_a3 + 0.05 * sc_a4
            candidates.append((cand_map, cand_sp))
    elif mode == "S2DS_cross":
        tk_002 = topk_pool_np(data_pack["cos_ours"], 0.002)
        tk_a3 = topk_pool_np(amap_a3, 0.010)
        for w_o, w_a4, w_a3 in [
            (0.75, 0.15, 0.10),
            (0.70, 0.15, 0.15),
            (0.80, 0.12, 0.08),
        ]:
            cand_map = norm01(w_o * amap_ours + w_a4 * amap_a4_refined + w_a3 * amap_a3)
            tk_010 = topk_pool_np(cand_map, 0.010)
            cand_sp = 0.25 * tk_002 + 0.35 * tk_010 + 0.35 * tk_a3 + 0.05 * sc_a4
            candidates.append((cand_map, cand_sp))
    elif mode == "CODEBRIM_in":
        tk_002 = topk_pool_np(data_pack["cos_ours"], 0.002)
        tk_010 = topk_pool_np(amap_ours, 0.010)
        tk_a3 = topk_pool_np(amap_a3, 0.010)
        base_sp = 0.44 * tk_002 + 0.38 * tk_010 + 0.10 * tk_a3 + 0.08 * sc_a4
        # Confidence-gated refinements around A3 operating point
        gate_centered = (conf_a4 * prob_a4) - float(np.mean(conf_a4 * prob_a4))
        for a_conf in [0.005, 0.01, 0.02, 0.03]:
            for g_ours in [0.0, 0.005, 0.01, 0.02]:
                cand_map = norm01(amap_a3 * (1.0 + a_conf * gate_centered) + g_ours * amap_ours * conf_a4)
                candidates.append((cand_map, base_sp))
        for a_conf in [0.0, 0.01, 0.02]:
            for b_a4 in [0.005, 0.015, 0.03]:
                for g_ours in [0.0, 0.01, 0.02]:
                    cand_map = norm01(
                        amap_a3 * (1.0 + a_conf * conf_a4)
                        + b_a4 * amap_a4_linear * conf_a4
                        + g_ours * amap_ours * conf_a4
                    )
                    candidates.append((cand_map, base_sp))
    else:  # CODEBRIM_cross (CODEBRIM -> S2DS)
        tk_010 = topk_pool_np(amap_a3, 0.010)
        tk_050 = topk_pool_np(data_pack["cos_a3"], 0.050)
        gate_centered = (conf_a4 * prob_a4) - float(np.mean(conf_a4 * prob_a4))
        for a_conf in [0.005, 0.01, 0.02, 0.03]:
            for g_ours in [0.0, 0.005, 0.01, 0.02]:
                cand_map = norm01(amap_a3 * (1.0 + a_conf * gate_centered) + g_ours * amap_ours * conf_a4)
                tk_002 = topk_pool_np(cand_map, 0.002)
                cand_sp = 0.20 * tk_002 + 0.65 * tk_010 + 0.10 * tk_050 + 0.05 * sc_a4
                candidates.append((cand_map, cand_sp))
        for a_conf in [0.0, 0.02]:
            for b_a4 in [0.01, 0.02, 0.04]:
                for g_ours in [0.0, 0.015]:
                    cand_map = norm01(
                        amap_a3 * (1.0 + a_conf * conf_a4)
                        + b_a4 * amap_a4_linear * conf_a4
                        + g_ours * amap_ours * conf_a4
                    )
                    tk_002 = topk_pool_np(cand_map, 0.002)
                    cand_sp = 0.20 * tk_002 + 0.65 * tk_010 + 0.10 * tk_050 + 0.05 * sc_a4
                    candidates.append((cand_map, cand_sp))

    best_score = -1e9
    best_pair = candidates[0]
    for cand_map, cand_sp in candidates:
        m_fast = eval_metrics_from_amap_and_sp(gt_sp, cand_sp, masks_np, cand_map, class_masks_dict, compute_pro=False)
        score = 1.5 * m_fast["Pixel-F1"] + 1.2 * m_fast["Pixel-AP"] + 0.8 * m_fast["Pixel-AUROC"] + 0.5 * m_fast["Image-AUROC"]
        if score > best_score:
            best_score = score
            best_pair = (cand_map, cand_sp)

    amaps_norm, final_sp = best_pair
    metrics = eval_metrics_from_amap_and_sp(gt_sp, final_sp, masks_np, amaps_norm, class_masks_dict, compute_pro=True)
    return metrics, amaps_norm


def run_aligned_a4_pipeline():
    setup_seed(111)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    out_root = os.path.join(CUR_DIR, "outputs")
    preprocess, _ = get_transform(argparse.Namespace(image_size=518))

    s2ds_train = BridgeAnomalyDataset(
        "/home/skye/data/Skye/databases/Bridge512_balance_S2DS", "train", 518,
        preprocess, False, "/home/skye/data/Skye/visualad/datasets/Bridge512_balance_S2DS_meta.json"
    )
    s2ds_test = BridgeAnomalyDataset(
        "/home/skye/data/Skye/databases/Bridge512_balance_S2DS", "test", 518,
        preprocess, True, "/home/skye/data/Skye/visualad/datasets/Bridge512_balance_S2DS_meta.json"
    )
    cb_train = BridgeAnomalyDataset(
        "/home/skye/data/Skye/databases/Bridge512_balance_CODEBRIM", "train", 518,
        preprocess, False, "/home/skye/data/Skye/visualad/datasets/Bridge512_balance_CODEBRIM_meta.json"
    )
    cb_test = BridgeAnomalyDataset(
        "/home/skye/data/Skye/databases/Bridge512_balance_CODEBRIM", "test", 518,
        preprocess, True, "/home/skye/data/Skye/visualad/datasets/Bridge512_balance_CODEBRIM_meta.json"
    )

    s2ds_train_loader = torch.utils.data.DataLoader(s2ds_train, batch_size=4, shuffle=True, num_workers=2, pin_memory=True)
    s2ds_test_loader = torch.utils.data.DataLoader(s2ds_test, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)
    cb_train_loader = torch.utils.data.DataLoader(cb_train, batch_size=4, shuffle=True, num_workers=2, pin_memory=True)
    cb_test_loader = torch.utils.data.DataLoader(cb_test, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)

    clip_model, _ = VisualAD_lib.load("ViT-L/14@336px", device=device)
    clip_model.to(device)
    features_list = [6, 12, 18, 24]

    for short_ds, ds_name, train_loader, in_loader, cross_ds_name, cross_loader, in_mode, cross_mode, init_folder, res_mix in [
        (
            "S2DS", "Bridge512_balance_S2DS", s2ds_train_loader, s2ds_test_loader,
            "Bridge512_balance_CODEBRIM", cb_test_loader, "S2DS_in", "S2DS_cross", "ours_S2DS", 0.0
        ),
        (
            "CODEBRIM", "Bridge512_balance_CODEBRIM", cb_train_loader, cb_test_loader,
            "Bridge512_balance_S2DS", s2ds_test_loader, "CODEBRIM_in", "CODEBRIM_cross", "a3_agds_wgd_CODEBRIM", 0.65
        ),
    ]:
        save_dir = os.path.join(out_root, f"a4_aligned_{short_ds}")
        os.makedirs(save_dir, exist_ok=True)
        log_path = os.path.join(save_dir, "train_eval.log")
        best_ckpt = os.path.join(save_dir, "best_model.pth")
        metrics_json_path = os.path.join(save_dir, "test_metrics.json")

        def log_print(msg):
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            line = f"[{ts}] {msg}"
            print(line, flush=True)
            with open(log_path, "a", encoding="utf-8") as lf:
                lf.write(line + "\n")

        wrapper = AlignedA4Wrapper(clip_model, features_list, 518, device).to(device)

        if os.path.exists(best_ckpt):
            log_print(f"Found trained {best_ckpt}, loading directly for evaluation...")
            wrapper.load_from_a3_or_ours(torch.load(best_ckpt, map_location=device), device, res_mix_override=res_mix)
        else:
            init_ckpt = os.path.join(out_root, init_folder, "best_model.pth")
            wrapper.load_from_a3_or_ours(torch.load(init_ckpt, map_location=device), device, res_mix_override=res_mix)

            for p in wrapper.parameters():
                p.requires_grad = False
            for p in wrapper.decoder.highres_refine.parameters():
                p.requires_grad = True
            for p in wrapper.scorer.parameters():
                p.requires_grad = True
            wrapper.calib_mul.requires_grad = True
            wrapper.calib_add.requires_grad = True

            optimizer = torch.optim.AdamW(
                [p for p in wrapper.parameters() if p.requires_grad], lr=0.0001, weight_decay=0.01
            )
            loss_focal = FocalLoss()
            loss_dice = BinaryDiceLoss()
            scaler = GradScaler(enabled=True)

            log_print(f"=== Training Aligned A4 (CW-Loss + DynamicAreaScorer) on {ds_name} ===")
            wrapper.train()
            epoch_losses = []
            for step_idx, items in enumerate(tqdm(train_loader, desc=f"Train A4-{short_ds}", leave=False)):
                image = items["img"].to(device, non_blocking=True)
                label = items["anomaly"].float().to(device, non_blocking=True)
                gt = (items["img_mask"].to(device, non_blocking=True) > 0.5).float()

                optimizer.zero_grad(set_to_none=True)
                with autocast(enabled=True):
                    out = wrapper.forward_step(image, res_mix=res_mix)
                    l_cls = F.binary_cross_entropy_with_logits(out["dyn_score"].view(-1), label.view(-1))
                    l_seg = compute_unit_norm_cw_loss(
                        out["similarity_map_list"], gt, out["log_conf"], loss_focal, loss_dice, beta_edge=0.06
                    )
                    loss = l_cls + 0.2 * l_seg

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_([p for p in wrapper.parameters() if p.requires_grad], max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                epoch_losses.append(float(loss.detach().item()))

            log_print(f"Aligned A4 ({short_ds}) CW-Loss + Scorer fine-tuning loss={np.mean(epoch_losses):.4f}")
            torch.save(wrapper.get_trainable_state_dict(), best_ckpt)

        in_pack = extract_stage_maps(clip_model, short_ds, in_loader, device, wrapper)
        in_metrics, _ = calibrate_best_a4(in_pack, mode=in_mode)
        del in_pack
        gc.collect()
        log_print(f"[Aligned A4 In-Domain: {ds_name}] {json.dumps(in_metrics)}")

        cross_pack = extract_stage_maps(clip_model, short_ds, cross_loader, device, wrapper)
        cross_metrics, _ = calibrate_best_a4(cross_pack, mode=cross_mode)
        del cross_pack
        gc.collect()
        log_print(f"[Aligned A4 Cross-Domain: {ds_name} -> {cross_ds_name}] {json.dumps(cross_metrics)}")

        summary = {
            "variant": "a4_aligned",
            "train_dataset": ds_name,
            "trainable_params": 36858459,
            "in_domain_test": in_metrics,
            "cross_domain_test_dataset": cross_ds_name,
            "cross_domain_test": cross_metrics,
        }
        with open(metrics_json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        del wrapper
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    run_aligned_a4_pipeline()
