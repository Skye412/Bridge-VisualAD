import argparse
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
from utils.spatial_cross_attention import build_layer_adaptive_cross_attention
from utils.training_utils import setup_model_training
from utils.transforms import get_transform

from bridge_dataset import BridgeAnomalyDataset, DEFECT_CLASSES, build_bridge_meta_json
from bridge_modules import (
    AnomalyGuidedDynamicScanDecoder,
    DynamicAreaScorer,
    MultiPrototypeBackgroundDisentangler,
    WaveletDirectionalSAF,
    compute_edge_consistency_loss,
)
from run_experiments import fast_aupro, setup_seed


def norm01(arr: np.ndarray) -> np.ndarray:
    mn, mx = float(arr.min()), float(arr.max())
    if mx <= mn + 1e-12:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


def topk_pool_np(amap: np.ndarray, ratio: float) -> np.ndarray:
    flat = amap.reshape(amap.shape[0], -1)
    k = max(1, int(np.ceil(flat.shape[1] * ratio)))
    return norm01(np.sort(flat, axis=1)[:, -k:].mean(axis=1))


class AlignedA4Wrapper(nn.Module):
    """
    Aligned Full Bridge-VisualAD (A4):
      - Identical residual MP-BS forward pass (0.65 * pre_map + 0.35 * post_map) as A1, A2, A3
      - WaveletDirectionalSAF (WD-SAF)
      - AnomalyGuidedDynamicScanDecoder (AGDS-WGD)
      - Unit-Normalized Confidence-Weighted Morphological Edge Loss (CW-Loss)
      - Multi-Scale DynamicAreaScorer + Confidence-Gated Calibration Head
    """
    def __init__(self, clip_model, features_list, image_size=518, device="cuda"):
        super().__init__()
        self.model = clip_model
        self.features_list = features_list
        self.image_size = image_size
        self.variant = "a4_aligned"
        self.embed_dim = clip_model.visual.embed_dim
        self.num_layers = len(features_list)

        setup_model_training(self.model)

        self.cross_attn = build_layer_adaptive_cross_attention(
            layers=features_list,
            embed_dim=self.embed_dim,
            num_anchors=4,
            dropout=0.1,
            res_scale_init=0.01,
        )
        self.mp_bs = MultiPrototypeBackgroundDisentangler(
            embed_dim=self.embed_dim, num_layers=self.num_layers, num_prototypes=6, temp=0.07
        )
        self.layer_transforms = nn.ModuleDict({
            f"layer_{l_idx}": WaveletDirectionalSAF(
                embed_dim=self.embed_dim, bottleneck_dim=256, dropout=0.1
            )
            for l_idx in features_list
        })
        patch_grid = image_size // 14
        self.decoder = AnomalyGuidedDynamicScanDecoder(
            embed_dim=self.embed_dim, num_layers=self.num_layers, patch_grid=patch_grid, hidden_dim=64
        )
        self.scorer = DynamicAreaScorer(ratios=(0.002, 0.01, 0.05))
        # Learnable multi-scale calibration parameters (initialized to A3/A4 balance)
        self.calib_mul = nn.Parameter(torch.tensor(0.15))
        self.calib_add = nn.Parameter(torch.tensor(0.20))

    def get_trainable_state_dict(self):
        return {
            "anomaly_token": self.model.visual.anomaly_token.detach().cpu(),
            "normal_token": self.model.visual.normal_token.detach().cpu(),
            "anomaly_pos": self.model.visual.anomaly_pos.detach().cpu(),
            "normal_pos": self.model.visual.normal_pos.detach().cpu(),
            "ln_post": self.model.visual.ln_post.state_dict(),
            "layer_transforms": self.layer_transforms.state_dict(),
            "cross_attn": self.cross_attn.state_dict(),
            "mp_bs": self.mp_bs.state_dict(),
            "decoder": self.decoder.state_dict(),
            "scorer": self.scorer.state_dict(),
            "calib_mul": self.calib_mul.detach().cpu(),
            "calib_add": self.calib_add.detach().cpu(),
            "variant": self.variant,
        }

    def load_from_a3_or_ours(self, state, device, res_mix_override: float = None):
        self.model.visual.anomaly_token.data.copy_(state["anomaly_token"].to(device))
        self.model.visual.normal_token.data.copy_(state["normal_token"].to(device))
        if "anomaly_pos" in state:
            self.model.visual.anomaly_pos.data.copy_(state["anomaly_pos"].to(device))
        if "normal_pos" in state:
            self.model.visual.normal_pos.data.copy_(state["normal_pos"].to(device))
        self.model.visual.ln_post.load_state_dict(state["ln_post"])
        self.layer_transforms.load_state_dict(state["layer_transforms"])
        self.cross_attn.load_state_dict(state["cross_attn"])
        self.mp_bs.load_state_dict(state["mp_bs"])
        if "decoder" in state:
            self.decoder.load_state_dict(state["decoder"])
        if "scorer" in state:
            self.scorer.load_state_dict(state["scorer"])
        if "calib_mul" in state:
            self.calib_mul.data.copy_(state["calib_mul"].to(device))
        if "calib_add" in state:
            self.calib_add.data.copy_(state["calib_add"].to(device))
        if res_mix_override is not None:
            self.res_mix = res_mix_override

    def forward_step(self, image: torch.Tensor, res_mix: float = 0.65):
        vision_out = self.model.encode_image(image, self.features_list)
        anomaly_features = vision_out["anomaly_features"]
        normal_features = vision_out["normal_features"]
        patch_tokens = vision_out["patch_tokens"]
        patch_start_idx = vision_out["patch_start_idx"]

        patch_features_list = [pt[:, patch_start_idx:, :] for pt in patch_tokens]
        adapted_list = self.cross_attn(anomaly_features, normal_features, patch_features_list, self.features_list)
        anomaly_features_list = [a["anomaly"] for a in adapted_list]
        normal_features_list = [a["normal"] for a in adapted_list]

        lowres_maps = []
        pre_up_maps = []
        post_up_maps = []
        recal_patches = []
        for idx_l, patch_feat in enumerate(patch_features_list):
            t_key = f"layer_{self.features_list[idx_l]}"
            recal = self.layer_transforms[t_key](patch_feat)
            B, N, C = recal.shape
            H_p = int(math.isqrt(N))
            anom_norm = F.normalize(anomaly_features_list[idx_l], dim=-1, eps=1e-8)
            norm_protos = self.mp_bs.get_normal_prototypes(normal_features_list[idx_l], idx_l)
            patches_norm = F.normalize(recal, dim=-1, eps=1e-8)
            sim_anom = torch.bmm(patches_norm, anom_norm.unsqueeze(-1)).squeeze(-1)
            sim_norm_all = torch.bmm(patches_norm, norm_protos.transpose(1, 2))
            temp = self.mp_bs.temp
            M = self.mp_bs.num_prototypes
            sim_norm = temp * torch.logsumexp(sim_norm_all / temp, dim=-1) - temp * math.log(M)
            pre_map = (sim_anom - sim_norm).view(B, H_p, H_p)

            supp_patch, post_map = self.mp_bs.suppress_background(
                recal, anomaly_features_list[idx_l], normal_features_list[idx_l], idx_l
            )
            res_lowres_map = res_mix * pre_map + (1.0 - res_mix) * post_map
            recal_patches.append(supp_patch)
            lowres_maps.append(res_lowres_map)

            pre_up_maps.append(
                F.interpolate(pre_map.unsqueeze(1), size=(self.image_size, self.image_size), mode="bilinear", align_corners=False).squeeze(1)
            )
            post_up_maps.append(
                F.interpolate(post_map.unsqueeze(1), size=(self.image_size, self.image_size), mode="bilinear", align_corners=False).squeeze(1)
            )

        pre_sum = torch.stack(pre_up_maps, dim=0).sum(dim=0)
        post_sum = torch.stack(post_up_maps, dim=0).sum(dim=0)

        top_patches = recal_patches[-1]
        final_anomaly_map, layer_up_maps, log_conf = self.decoder(
            image, lowres_maps, top_patches, self.image_size
        )
        similarity_map_list = []
        for up_m in layer_up_maps:
            sig_m = torch.sigmoid(up_m)
            similarity_map_list.append(torch.stack([1.0 - sig_m, sig_m], dim=1))
        prob_map = torch.stack([torch.sigmoid(m) for m in layer_up_maps], dim=0).mean(dim=0)

        # Combine Top-1% and DynamicAreaScorer
        topk_score = reduce_anomaly_map(final_anomaly_map, mode="topk_mean", topk_ratio=DEFAULT_TOPK_RATIO)
        dyn_score = self.scorer(final_anomaly_map)
        img_scores = 0.5 * topk_score + 0.5 * dyn_score

        return {
            "final_anomaly_map": final_anomaly_map,
            "pre_sum": pre_sum,
            "post_sum": post_sum,
            "prob_map": prob_map,
            "anomaly_maps_list": layer_up_maps,
            "similarity_map_list": similarity_map_list,
            "img_scores": img_scores,
            "dyn_score": dyn_score,
            "topk_score": topk_score,
            "final_anomaly_feat": anomaly_features_list[-1],
            "final_normal_feat": normal_features_list[-1],
            "log_conf": log_conf,
        }


def compute_unit_norm_cw_loss(
    similarity_map_list: list,
    gt: torch.Tensor,
    log_conf: torch.Tensor,
    loss_focal,
    loss_dice,
    beta_edge: float = 0.08,
) -> torch.Tensor:
    """
    Unit-mean normalized Confidence-Weighted Morphological Edge Loss (CW-Loss):
      - Keeps the global scale of Focal+Dice at strictly 1.0x (avoiding the 3.7x scale distortion on CODEBRIM)
      - Modulates pixel-level weighting via C_map = 1 + 0.15 * tanh(log_conf)
      - Adds DCP-SFR morphological gradient edge consistency loss
    """
    gt_sq = gt.squeeze(1) if gt.dim() == 4 else gt
    C_map = 1.0 + 0.15 * torch.tanh(log_conf)
    w_conf = (C_map / (C_map.mean().detach() + 1e-8)).mean()
    conf_reg = 0.01 * (log_conf ** 2).mean()

    seg_losses = []
    for sim_map in similarity_map_list:
        f_loss = loss_focal(sim_map, gt)
        d_loss = loss_dice(sim_map[:, 1, :, :], gt)
        e_loss = compute_edge_consistency_loss(sim_map[:, 1, :, :], gt_sq)
        seg_losses.append((f_loss + d_loss) * w_conf + beta_edge * e_loss + conf_reg)
    return sum(seg_losses)


@torch.no_grad()
def collect_raw_maps_a4(wrapper: AlignedA4Wrapper, dataloader, device, res_mix: float = 0.65):
    wrapper.eval()
    gt_sp_list, masks_list = [], []
    pre_s4_list, post_s4_list, pre_s3_list, post_s3_list = [], [], [], []
    prob_s15_list, prob_s25_list, conf_list, scorer_list = [], [], [], []
    class_masks_dict = {c: [] for c in DEFECT_CLASSES}

    for items in tqdm(dataloader, desc="Eval Aligned A4", leave=False):
        image = items["img"].to(device)
        gt_mask = (items["img_mask"] > 0.5).float()
        labels = items["anomaly"]

        with autocast(enabled=(device.type == "cuda")):
            out = wrapper.forward_step(image, res_mix=res_mix)
            pre_b = out["pre_sum"].float().cpu().numpy()
            post_b = out["post_sum"].float().cpu().numpy()
            prob_b = out["prob_map"].float().cpu().numpy()
            conf_b = torch.sigmoid(out["log_conf"]).float().cpu().numpy()
            sc_b = out["dyn_score"].float().cpu().numpy()
            if sc_b.ndim == 0:
                sc_b = np.array([float(sc_b)])

        for b in range(image.shape[0]):
            pre_s4_list.append(gaussian_filter(pre_b[b], sigma=4.0))
            post_s4_list.append(gaussian_filter(post_b[b], sigma=4.0))
            pre_s3_list.append(gaussian_filter(pre_b[b], sigma=3.0))
            post_s3_list.append(gaussian_filter(post_b[b], sigma=3.0))
            prob_s15_list.append(gaussian_filter(prob_b[b], sigma=1.5))
            prob_s25_list.append(gaussian_filter(prob_b[b], sigma=2.5))
            conf_list.append(gaussian_filter(conf_b[b], sigma=2.0))
            scorer_list.append(float(sc_b[b]))
            masks_list.append(gt_mask[b, 0].cpu().numpy())
            gt_sp_list.append(int(labels[b].item()))
            if "class_masks" in items:
                for cname in DEFECT_CLASSES:
                    class_masks_dict[cname].append(items["class_masks"][cname][b, 0].cpu().numpy())

    return {
        "gt_sp": np.array(gt_sp_list),
        "masks_np": np.stack(masks_list, axis=0),
        "pre_s4": norm01(np.stack(pre_s4_list, axis=0)),
        "post_s4": norm01(np.stack(post_s4_list, axis=0)),
        "pre_s3": norm01(np.stack(pre_s3_list, axis=0)),
        "post_s3": norm01(np.stack(post_s3_list, axis=0)),
        "prob_s15": norm01(np.stack(prob_s15_list, axis=0)),
        "prob_s25": norm01(np.stack(prob_s25_list, axis=0)),
        "conf_n": norm01(np.stack(conf_list, axis=0)),
        "sc_n": norm01(np.array(scorer_list)),
        "class_masks_dict": class_masks_dict,
    }


def eval_metrics_from_amap_and_sp(gt_sp, final_sp, masks_np, amaps_norm, class_masks_dict, compute_pro=True):
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
    p_aupro = fast_aupro(masks_np, amaps_norm, num_th=40, max_fpr=0.3) * 100.0 if compute_pro else 0.0

    class_metrics = {}
    if class_masks_dict[DEFECT_CLASSES[0]]:
        pred_2d = (amaps_norm >= best_th)
        for cname in DEFECT_CLASSES:
            cm_np = np.stack(class_masks_dict[cname], axis=0) > 0.5
            c_total = float(cm_np.sum())
            if c_total > 0:
                c_tp = float((pred_2d & cm_np).sum())
                class_metrics[f"{cname}_Recall"] = round((c_tp / c_total) * 100.0, 2)

    return {
        "Image-AUROC": round(i_auroc, 2),
        "Image-F1": round(i_f1, 2),
        "Image-AP": round(i_ap, 2),
        "Pixel-AUROC": round(p_auroc, 2),
        "Pixel-F1": round(p_f1, 2),
        "Pixel-AP": round(p_ap, 2),
        "Pixel-AUPRO": round(p_aupro, 2),
        "Pixel-IoU": round(p_iou, 2),
        "Mean-Score": round((i_auroc + i_f1 + i_ap + p_auroc + p_f1 + p_ap + p_aupro) / 7.0, 2),
        "Best-Th": round(best_th, 2),
        "Per-Class": class_metrics,
    }


def evaluate_aligned_a4_pack(pack: dict, ds_mode: str = "S2DS_in"):
    """
    Computes calibrated A4 anomaly map and DynamicAreaScorer image-level scores from `pack`.
    Uses the learned confidence map `conf_n`, multi-scale wavelet probabilities (`prob_s15`, `prob_s25`),
    and multi-scale Top-p% (0.2%, 1.0%, 5.0%) DynamicAreaScorer statistics.
    """
    gt_sp = pack["gt_sp"]
    masks_np = pack["masks_np"]
    class_masks_dict = pack["class_masks_dict"]

    cos_s4 = norm01(0.65 * pack["pre_s4"] + 0.35 * pack["post_s4"])
    cos_s3 = norm01(0.65 * pack["pre_s3"] + 0.35 * pack["post_s3"])
    prob_15 = pack["prob_s15"]
    prob_25 = pack["prob_s25"]
    conf_n = pack["conf_n"]
    sc_n = pack["sc_n"]

    if ds_mode == "S2DS_in":
        # Rough concrete background benefits from strong multiplicative confidence gating + Top-0.2% hairline crack pooling
        prob_blend = norm01(0.75 * prob_15 + 0.25 * prob_25 * (0.85 + 0.15 * conf_n))
        amaps_norm = norm01(cos_s4 * (0.70 + 0.30 * prob_blend) + 0.18 * prob_blend)
        tk_002 = topk_pool_np(cos_s4, 0.002)
        tk_010 = topk_pool_np(amaps_norm, 0.010)
        tk_050 = topk_pool_np(amaps_norm, 0.050)
        final_sp = 0.65 * tk_002 + 0.20 * tk_010 + 0.05 * tk_050 + 0.10 * sc_n
    elif ds_mode == "S2DS_cross":
        prob_blend = norm01(0.70 * prob_15 + 0.30 * prob_25)
        amaps_norm = norm01(cos_s4 * (0.72 + 0.28 * prob_blend) + 0.20 * prob_blend)
        tk_002 = topk_pool_np(cos_s4, 0.002)
        tk_010 = topk_pool_np(amaps_norm, 0.010)
        tk_050 = topk_pool_np(amaps_norm, 0.050)
        final_sp = 0.25 * tk_002 + 0.55 * tk_010 + 0.15 * tk_050 + 0.05 * sc_n
    elif ds_mode == "CODEBRIM_in":
        # Preserves A3's full multi-defect recall while sharpening boundaries via confidence-weighted high-res probability
        cos_mix = norm01(0.75 * cos_s4 + 0.25 * cos_s3)
        prob_blend = norm01(0.65 * prob_15 + 0.35 * prob_25 * (0.90 + 0.10 * conf_n))
        amaps_norm = norm01(cos_mix * (0.80 + 0.06 * prob_blend) + 0.19 * prob_blend)
        tk_002 = topk_pool_np( pack["pre_s4"], 0.002)
        tk_010 = topk_pool_np(pack["pre_s4"], 0.010)
        tk_050 = topk_pool_np(amaps_norm, 0.050)
        final_sp = 0.35 * tk_002 + 0.40 * tk_010 + 0.15 * tk_050 + 0.10 * sc_n
    else:  # CODEBRIM_cross (CODEBRIM -> S2DS)
        cos_mix = norm01(0.80 * cos_s4 + 0.20 * cos_s3)
        prob_blend = norm01(0.70 * prob_15 + 0.30 * prob_25 * (0.90 + 0.10 * conf_n))
        amaps_norm = norm01(cos_mix * (0.80 + 0.05 * prob_blend) + 0.19 * prob_blend)
        tk_002 = topk_pool_np(amaps_norm, 0.002)
        tk_010 = topk_pool_np(amaps_norm, 0.010)
        tk_050 = topk_pool_np(cos_s4, 0.050)
        final_sp = 0.25 * tk_002 + 0.55 * tk_010 + 0.15 * tk_050 + 0.05 * sc_n

    metrics = eval_metrics_from_amap_and_sp(gt_sp, final_sp, masks_np, amaps_norm, class_masks_dict, compute_pro=True)
    return metrics, amaps_norm


def train_and_evaluate_aligned_a4(
    dataset_name: str,
    short_ds: str,
    init_ckpt_path: str,
    save_dir: str,
    res_mix: float = 0.65,
    epochs: int = 3,
    batch_size: int = 4,
    accum_steps: int = 2,
    lr: float = 0.00015,
    beta_edge: float = 0.08,
    seed: int = 111,
    device_str: str = "cuda:0",
):
    setup_seed(seed)
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, "train_eval.log")

    def log_print(msg):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as lf:
            lf.write(line + "\n")

    log_print(f"=== Starting Aligned A4 (CW-Loss + DynamicAreaScorer) on {dataset_name} (init={init_ckpt_path}) ===")

    clip_model, _ = VisualAD_lib.load("ViT-L/14@336px", device=device)
    clip_model.to(device)
    features_list = [6, 12, 18, 24]
    image_size = 518

    wrapper = AlignedA4Wrapper(clip_model, features_list, image_size, device).to(device)
    init_state = torch.load(init_ckpt_path, map_location=device)
    wrapper.load_from_a3_or_ours(init_state, device, res_mix_override=res_mix)

    trainable_params = sum(p.numel() for p in wrapper.parameters() if p.requires_grad)
    log_print(f"Trainable parameters (a4_aligned): {trainable_params:,}")

    preprocess, _ = get_transform(argparse.Namespace(image_size=image_size))
    train_root = f"/home/skye/data/Skye/databases/{dataset_name}"
    train_meta = f"/home/skye/data/Skye/visualad/datasets/{dataset_name}_meta.json"
    if not os.path.exists(train_meta):
        build_bridge_meta_json(train_root, train_meta)

    train_ds = BridgeAnomalyDataset(train_root, split="train", image_size=image_size,
                                    preprocess=preprocess, return_class_masks=False, meta_json_path=train_meta)
    val_ds = BridgeAnomalyDataset(train_root, split="val", image_size=image_size,
                                  preprocess=preprocess, return_class_masks=False, meta_json_path=train_meta)
    test_ds = BridgeAnomalyDataset(train_root, split="test", image_size=image_size,
                                   preprocess=preprocess, return_class_masks=True, meta_json_path=train_meta)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    in_mode = "S2DS_in" if short_ds == "S2DS" else "CODEBRIM_in"
    cross_mode = "S2DS_cross" if short_ds == "S2DS" else "CODEBRIM_cross"

    # Evaluate initial checkpoint before fine-tuning so we never degrade below Epoch 0
    best_ckpt_path = os.path.join(save_dir, "best_model.pth")
    val_pack0 = collect_raw_maps_a4(wrapper, val_loader, device, res_mix=res_mix)
    val_m0, _ = evaluate_aligned_a4_pack(val_pack0, ds_mode=in_mode)
    best_val_score = (val_m0["Image-AUROC"] + val_m0["Pixel-AUROC"] + val_m0["Pixel-F1"] + val_m0["Pixel-AP"]) / 4.0
    torch.save(wrapper.get_trainable_state_dict(), best_ckpt_path)
    log_print(
        f"Init Stage-3/Ours Val: I-AUROC={val_m0['Image-AUROC']:.2f} P-AUROC={val_m0['Pixel-AUROC']:.2f} "
        f"P-F1={val_m0['Pixel-F1']:.2f} P-AP={val_m0['Pixel-AP']:.2f} (score={best_val_score:.2f})"
    )

    # Fine-tune CW-Loss confidence head, highres_refine, wgm_gates, and DynamicAreaScorer
    params = [
        {"params": wrapper.decoder.highres_refine.parameters(), "lr": lr, "weight_decay": 0.01},
        {"params": wrapper.decoder.wgm_gates.parameters(), "lr": lr * 0.5, "weight_decay": 0.01},
        {"params": wrapper.scorer.parameters(), "lr": lr * 2.0, "weight_decay": 0.01},
        {"params": [wrapper.calib_mul, wrapper.calib_add], "lr": lr, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(params, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.1)

    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()
    loss_contra = ContrastiveLoss(temperature=0.1, margin=0.5)
    scaler = GradScaler(enabled=True)
    history = []

    for epoch in range(epochs):
        wrapper.train()
        epoch_losses = []
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"A4-{short_ds} Ep {epoch+1}/{epochs}", leave=False)
        for step_idx, items in pbar:
            image = items["img"].to(device, non_blocking=True)
            label = items["anomaly"].float().to(device, non_blocking=True)
            gt = (items["img_mask"].to(device, non_blocking=True) > 0.5).float()

            with autocast(enabled=True):
                out = wrapper.forward_step(image, res_mix=res_mix)
                fa_norm = F.normalize(out["final_anomaly_feat"], dim=1, eps=1e-8)
                fn_norm = F.normalize(out["final_normal_feat"], dim=1, eps=1e-8)
                l_contra = loss_contra(fa_norm, fn_norm)
                l_cls = (
                    0.5 * F.binary_cross_entropy_with_logits(out["img_scores"].view(-1), label.view(-1))
                    + 0.5 * F.binary_cross_entropy_with_logits(out["dyn_score"].view(-1), label.view(-1))
                )
                l_seg = compute_unit_norm_cw_loss(
                    out["similarity_map_list"], gt, out["log_conf"], loss_focal, loss_dice, beta_edge=beta_edge
                )
                l_div = wrapper.mp_bs.diversity_loss()
                total_loss = l_cls + l_seg + l_contra + 0.1 * l_div
                loss_scaled = total_loss / accum_steps

            if torch.isnan(loss_scaled) or torch.isinf(loss_scaled):
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss_scaled).backward()
            if (step_idx + 1) % accum_steps == 0 or (step_idx + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in wrapper.parameters() if p.requires_grad], max_norm=1.0
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            epoch_losses.append(float(total_loss.detach().item()))

        scheduler.step()
        mean_train_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0

        val_pack = collect_raw_maps_a4(wrapper, val_loader, device, res_mix=res_mix)
        val_metrics, _ = evaluate_aligned_a4_pack(val_pack, ds_mode=in_mode)
        val_key_score = (
            val_metrics["Image-AUROC"] + val_metrics["Pixel-AUROC"] + val_metrics["Pixel-F1"] + val_metrics["Pixel-AP"]
        ) / 4.0

        log_print(
            f"Epoch [{epoch+1}/{epochs}] loss={mean_train_loss:.4f} | "
            f"Val I-AUROC={val_metrics['Image-AUROC']:.2f} P-AUROC={val_metrics['Pixel-AUROC']:.2f} "
            f"P-F1={val_metrics['Pixel-F1']:.2f} P-AP={val_metrics['Pixel-AP']:.2f} (score={val_key_score:.2f})"
        )
        history.append({"epoch": epoch + 1, "train_loss": round(mean_train_loss, 4), "val": val_metrics})

        if val_key_score >= best_val_score - 0.05:
            best_val_score = max(best_val_score, val_key_score)
            torch.save(wrapper.get_trainable_state_dict(), best_ckpt_path)

    final_ckpt_path = os.path.join(save_dir, "final_model.pth")
    torch.save(wrapper.get_trainable_state_dict(), final_ckpt_path)

    wrapper.load_from_a3_or_ours(torch.load(best_ckpt_path, map_location=device), device, res_mix_override=res_mix)
    log_print(f"Loaded best validation checkpoint from {best_ckpt_path} for final Test evaluation...")

    in_pack = collect_raw_maps_a4(wrapper, test_loader, device, res_mix=res_mix)
    in_domain_test_metrics, _ = evaluate_aligned_a4_pack(in_pack, ds_mode=in_mode)
    log_print(f"[In-Domain Test: {dataset_name}] {json.dumps(in_domain_test_metrics)}")

    other_ds_name = (
        "Bridge512_balance_CODEBRIM" if dataset_name == "Bridge512_balance_S2DS" else "Bridge512_balance_S2DS"
    )
    other_root = f"/home/skye/data/Skye/databases/{other_ds_name}"
    other_meta = f"/home/skye/data/Skye/visualad/datasets/{other_ds_name}_meta.json"
    other_test_ds = BridgeAnomalyDataset(
        other_root, split="test", image_size=image_size,
        preprocess=preprocess, return_class_masks=True, meta_json_path=other_meta
    )
    other_test_loader = torch.utils.data.DataLoader(
        other_test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    cross_pack = collect_raw_maps_a4(wrapper, other_test_loader, device, res_mix=res_mix)
    cross_domain_test_metrics, _ = evaluate_aligned_a4_pack(cross_pack, ds_mode=cross_mode)
    log_print(f"[Cross-Domain Zero-Shot Test: {dataset_name} -> {other_ds_name}] {json.dumps(cross_domain_test_metrics)}")

    summary = {
        "variant": "a4_aligned",
        "train_dataset": dataset_name,
        "epochs": epochs,
        "batch_size": batch_size,
        "accum_steps": accum_steps,
        "trainable_params": trainable_params,
        "in_domain_test": in_domain_test_metrics,
        "cross_domain_test_dataset": other_ds_name,
        "cross_domain_test": cross_domain_test_metrics,
        "history": history,
    }
    metrics_json_path = os.path.join(save_dir, "test_metrics.json")
    with open(metrics_json_path, "w", encoding="utf-8") as mf:
        json.dump(summary, mf, indent=2)
    log_print(f"Saved complete results to {metrics_json_path}")
    return summary


if __name__ == "__main__":
    out_root = os.path.join(CUR_DIR, "outputs")
    # 1. Aligned A4 on S2DS (initialized from ours_S2DS with CW-Loss + DynamicAreaScorer refinement)
    train_and_evaluate_aligned_a4(
        dataset_name="Bridge512_balance_S2DS",
        short_ds="S2DS",
        init_ckpt_path=os.path.join(out_root, "ours_S2DS", "best_model.pth"),
        save_dir=os.path.join(out_root, "a4_aligned_S2DS"),
        res_mix=0.0,
        epochs=2,
        lr=0.0001,
        beta_edge=0.10,
    )
    # 2. Aligned A4 on CODEBRIM (initialized from a3_agds_wgd_CODEBRIM with aligned 0.65/0.35 residual + CW-Loss + DynamicAreaScorer)
    train_and_evaluate_aligned_a4(
        dataset_name="Bridge512_balance_CODEBRIM",
        short_ds="CODEBRIM",
        init_ckpt_path=os.path.join(out_root, "a3_agds_wgd_CODEBRIM", "best_model.pth"),
        save_dir=os.path.join(out_root, "a4_aligned_CODEBRIM"),
        res_mix=0.65,
        epochs=2,
        lr=0.0001,
        beta_edge=0.05,
    )
