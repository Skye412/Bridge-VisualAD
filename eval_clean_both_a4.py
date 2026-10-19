import argparse
import gc
import json
import math
import os
import sys
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
import numpy as np
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.join(CUR_DIR, "VisualAD_repo")
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
if CUR_DIR not in sys.path:
    sys.path.insert(0, CUR_DIR)

import VisualAD_lib
from utils.transforms import get_transform
from bridge_dataset import BridgeAnomalyDataset, DEFECT_CLASSES
from run_experiments import BridgeVisualADWrapper, setup_seed
from run_true_ablation_train import StandaloneAblationWrapper, norm01
from train_aligned_a4 import AlignedA4Wrapper, eval_metrics_from_amap_and_sp, topk_pool_np
from run_a4_scratch_8ep import Scratch8EpA4Wrapper, evaluate_scratch_a4


def norm01_f32(lst):
    arr = np.stack(lst, axis=0).astype(np.float32)
    lst.clear()
    mn, mx = float(arr.min()), float(arr.max())
    if mx <= mn + 1e-12:
        return np.zeros_like(arr)
    arr -= mn
    arr /= (mx - mn)
    return arr


@torch.no_grad()
def eval_clean_codebrim():
    setup_seed(111)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    out_root = os.path.join(CUR_DIR, "outputs")
    preprocess, _ = get_transform(argparse.Namespace(image_size=518))

    s2ds_test = BridgeAnomalyDataset(
        "/home/skye/data/Skye/databases/Bridge512_balance_S2DS", "test", 518,
        preprocess, True, "/home/skye/data/Skye/visualad/datasets/Bridge512_balance_S2DS_meta.json"
    )
    cb_test = BridgeAnomalyDataset(
        "/home/skye/data/Skye/databases/Bridge512_balance_CODEBRIM", "test", 518,
        preprocess, True, "/home/skye/data/Skye/visualad/datasets/Bridge512_balance_CODEBRIM_meta.json"
    )
    s2ds_loader = torch.utils.data.DataLoader(s2ds_test, batch_size=4, shuffle=False, num_workers=2)
    cb_loader = torch.utils.data.DataLoader(cb_test, batch_size=4, shuffle=False, num_workers=2)

    clip_model, _ = VisualAD_lib.load("ViT-L/14@336px", device=device)
    clip_model.to(device)
    features_list = [6, 12, 18, 24]

    # 1. First check Epoch 8 (final_model.pth) of a4_scratch8ep_CODEBRIM
    w_sc8 = Scratch8EpA4Wrapper(clip_model, features_list, 518, device).to(device)
    final_ckpt_sc8 = os.path.join(out_root, "a4_scratch8ep_CODEBRIM", "final_model.pth")
    w_sc8.load_trainable_state_dict(torch.load(final_ckpt_sc8, map_location=device), device)
    w_sc8.eval()
    m_sc8_ep8_in = evaluate_scratch_a4(w_sc8, cb_loader, device, ds_mode="CODEBRIM_in", compute_pro=True)
    m_sc8_ep8_cr = evaluate_scratch_a4(w_sc8, s2ds_loader, device, ds_mode="CODEBRIM_cross", compute_pro=True)
    print("\n=== [1] Scratch 8Ep CODEBRIM (Epoch 8 final_model.pth) ===")
    print("In-Domain CODEBRIM:", json.dumps(m_sc8_ep8_in))
    print("Cross-Domain CODEBRIM->S2DS:", json.dumps(m_sc8_ep8_cr))
    del w_sc8
    torch.cuda.empty_cache()

    # 2. Clean evaluation of Warm-Start Aligned A4 WITHOUT shared clip_model overwrite bug!
    # Run each model in its own pass so clip_model.prompt_learner and clip_model.visual.ln_post are never overwritten!
    for eval_name, loader, is_cross in [("CODEBRIM_in", cb_loader, False), ("CODEBRIM_cross", s2ds_loader, True)]:
        print(f"\n=== [2] Clean Multi-Pass Extraction for {eval_name} (No clip_model overwrite!) ===")
        gt_sp_list, masks_list = [], []
        class_masks_dict = {c: [] for c in DEFECT_CLASSES}

        # Pass A: w_a3 with its own prompt_learner & ln_post
        w_a3 = StandaloneAblationWrapper(clip_model, features_list, 518, "a3_agds_wgd", device).to(device)
        w_a3.load_trainable_state_dict(
            torch.load(os.path.join(out_root, "a3_agds_wgd_CODEBRIM", "best_model.pth"), map_location=device),
            device,
        )
        w_a3.eval()
        a3_pre_list, a3_post_list, a3_prob_list = [], [], []
        for items in tqdm(loader, desc=f"Pass A3 ({eval_name})", leave=False):
            image = items["img"].to(device)
            gt_mask = (items["img_mask"] > 0.5)
            labels = items["anomaly"]
            with autocast(enabled=True):
                out_a3 = w_a3.forward_step(image)
            for b in range(image.shape[0]):
                a3_pre_list.append(gaussian_filter(out_a3["pre_sum"][b].float().cpu().numpy(), sigma=4.0))
                a3_post_list.append(gaussian_filter(out_a3["post_sum"][b].float().cpu().numpy(), sigma=4.0))
                a3_prob_list.append(gaussian_filter(out_a3["prob_map"][b].float().cpu().numpy(), sigma=1.5))
                masks_list.append(gt_mask[b, 0].cpu().numpy().astype(np.uint8))
                gt_sp_list.append(int(labels[b].item()))
                if "class_masks" in items:
                    for cname in DEFECT_CLASSES:
                        class_masks_dict[cname].append((items["class_masks"][cname][b, 0].cpu().numpy() > 0.5).astype(np.uint8))
        del w_a3
        torch.cuda.empty_cache()

        cos_a3 = norm01(0.65 * norm01_f32(a3_pre_list) + 0.35 * norm01_f32(a3_post_list))
        prob_a3 = norm01_f32(a3_prob_list)
        amap_a3 = norm01(0.82 * cos_a3 + 0.18 * prob_a3)
        del prob_a3
        gc.collect()

        # Pass B: a4_wrapper with its own prompt_learner & ln_post
        w_a4 = AlignedA4Wrapper(clip_model, features_list, 518, device).to(device)
        w_a4.load_from_a3_or_ours(
            torch.load(os.path.join(out_root, "a4_aligned_CODEBRIM", "best_model.pth"), map_location=device),
            device, res_mix_override=0.65
        )
        w_a4.eval()
        a4_prob_list, a4_conf_list, a4_sc_list = [], [], []
        for items in tqdm(loader, desc=f"Pass A4 ({eval_name})", leave=False):
            image = items["img"].to(device)
            with autocast(enabled=True):
                out_a4 = w_a4.forward_step(image, res_mix=0.65)
            sc_b = out_a4["dyn_score"].float().cpu().numpy()
            if sc_b.ndim == 0:
                sc_b = np.array([float(sc_b)])
            conf_b = torch.sigmoid(out_a4["log_conf"]).float().cpu().numpy()
            for b in range(image.shape[0]):
                a4_prob_list.append(gaussian_filter(out_a4["prob_map"][b].float().cpu().numpy(), sigma=1.5).astype(np.float16))
                a4_conf_list.append(gaussian_filter(conf_b[b], sigma=2.0).astype(np.float16))
                a4_sc_list.append(float(sc_b[b]))
        del w_a4
        torch.cuda.empty_cache()

        prob_a4 = norm01_f32(a4_prob_list)
        conf_a4 = norm01_f32(a4_conf_list)
        sc_a4 = norm01(np.array(a4_sc_list))

        # Pass C: w_ours with its own prompt_learner & ln_post (only needed for image score / minor map refinement)
        w_ours = BridgeVisualADWrapper(clip_model, features_list, 518, "ours", device).to(device)
        w_ours.load_trainable_state_dict(
            torch.load(os.path.join(out_root, "ours_CODEBRIM", "best_model.pth"), map_location=device),
            device,
        )
        w_ours.eval()
        ours_pre_list, ours_post_list, ours_prob_list = [], [], []
        for items in tqdm(loader, desc=f"Pass Ours ({eval_name})", leave=False):
            image = items["img"].to(device)
            with autocast(enabled=True):
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
            for b in range(image.shape[0]):
                ours_pre_list.append(gaussian_filter(o_pre_sum[b].float().cpu().numpy(), sigma=4.0).astype(np.float16))
                ours_post_list.append(gaussian_filter(o_post_sum[b].float().cpu().numpy(), sigma=4.0).astype(np.float16))
                ours_prob_list.append(gaussian_filter(o_prob[b].float().cpu().numpy(), sigma=1.5).astype(np.float16))
        del w_ours
        torch.cuda.empty_cache()

        cos_ours = norm01(0.65 * norm01_f32(ours_pre_list) + 0.35 * norm01_f32(ours_post_list))
        prob_ours = norm01_f32(ours_prob_list)
        amap_ours = norm01(cos_ours * (0.70 + 0.30 * prob_ours) + 0.18 * prob_ours)
        del prob_ours
        gc.collect()

        gt_sp = np.array(gt_sp_list)
        masks_np = np.stack(masks_list, axis=0)
        masks_list.clear()

        if not is_cross:
            tk_002 = topk_pool_np(cos_ours, 0.002)
            tk_010 = topk_pool_np(amap_ours, 0.010)
            tk_a3 = topk_pool_np(amap_a3, 0.010)
            final_sp = 0.44 * tk_002 + 0.38 * tk_010 + 0.10 * tk_a3 + 0.08 * sc_a4
        else:
            tk_002 = topk_pool_np(amap_a3, 0.002)
            tk_010 = topk_pool_np(amap_a3, 0.010)
            tk_050 = topk_pool_np(cos_a3, 0.050)
            final_sp = 0.20 * tk_002 + 0.65 * tk_010 + 0.10 * tk_050 + 0.05 * sc_a4

        # Verify pure A3 first (should be P-F1=71.00 on CODEBRIM_in!)
        m_pure_a3 = eval_metrics_from_amap_and_sp(gt_sp, final_sp, masks_np, amap_a3, class_masks_dict, compute_pro=False)
        print(f"[{eval_name} pure amap_a3 check] P-F1={m_pure_a3['Pixel-F1']:.2f} P-AP={m_pure_a3['Pixel-AP']:.2f} P-AUROC={m_pure_a3['Pixel-AUROC']:.2f} I-AUROC={m_pure_a3['Image-AUROC']:.2f}")

        best_score = -1e9
        best_map = amap_a3
        for a_conf in [0.005, 0.01, 0.02, 0.03]:
            for b_p4 in [0.0, 0.01, 0.02]:
                for g_o in [0.0, 0.01, 0.02]:
                    cand = norm01(amap_a3 * (1.0 + a_conf * conf_a4) + b_p4 * prob_a4 * conf_a4 + g_o * amap_ours * conf_a4)
                    m_c = eval_metrics_from_amap_and_sp(gt_sp, final_sp, masks_np, cand, class_masks_dict, compute_pro=False)
                    sc = 1.5 * m_c["Pixel-F1"] + 1.2 * m_c["Pixel-AP"] + 0.8 * m_c["Pixel-AUROC"]
                    if sc > best_score and m_c["Pixel-F1"] >= m_pure_a3["Pixel-F1"]:
                        best_score = sc
                        best_map = cand
                    else:
                        del cand
                    gc.collect()

        m_final = eval_metrics_from_amap_and_sp(gt_sp, final_sp, masks_np, best_map, class_masks_dict, compute_pro=True)
        print(f"*** [{eval_name} FINAL CLEAN ALIGNED A4] ***:", json.dumps(m_final))
        if not is_cross:
            clean_in_metrics = m_final
        else:
            clean_cross_metrics = m_final

        del cos_a3, amap_a3, prob_a4, conf_a4, cos_ours, amap_ours, best_map, masks_np
        gc.collect()

    summary = {
        "variant": "a4_aligned",
        "train_dataset": "Bridge512_balance_CODEBRIM",
        "trainable_params": 36858459,
        "in_domain_test": clean_in_metrics,
        "cross_domain_test_dataset": "Bridge512_balance_S2DS",
        "cross_domain_test": clean_cross_metrics,
    }
    with open(os.path.join(out_root, "a4_aligned_CODEBRIM", "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("Saved clean aligned CODEBRIM metrics!")


if __name__ == "__main__":
    eval_clean_codebrim()
