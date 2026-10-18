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

import VisualAD_lib
from utils.loss import BinaryDiceLoss, ContrastiveLoss, FocalLoss
from utils.scoring import DEFAULT_TOPK_RATIO, reduce_anomaly_map
from utils.spatial_cross_attention import build_layer_adaptive_cross_attention
from utils.training_utils import setup_feature_transforms, setup_model_training
from utils.transforms import get_transform

from bridge_dataset import BridgeAnomalyDataset, DEFECT_CLASSES, build_bridge_meta_json
from bridge_modules import (
    AnomalyGuidedDynamicScanDecoder,
    MultiPrototypeBackgroundDisentangler,
    WaveletDirectionalSAF,
)
from run_experiments import (
    fast_aupro,
    setup_seed,
    _make_balanced_smoke_subset,
)


def norm01(arr: np.ndarray) -> np.ndarray:
    mn, mx = float(arr.min()), float(arr.max())
    if mx <= mn + 1e-12:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


class StandaloneAblationWrapper(nn.Module):
    """
    True standalone ablation models trained independently from scratch:
      - 'a1_mp_bs'    : Baseline + MP-BS (Multi-Prototype Normal Bank & Background Suppression)
      - 'a2_wd_saf'   : Baseline + MP-BS + WD-SAF (Wavelet-Directional SAF)
      - 'a3_agds_wgd' : Baseline + MP-BS + WD-SAF + AGDS-WGD (Dynamic Scan & Wavelet Decoder, trained w/ standard Focal+Dice & Top-1% pool)
    """
    def __init__(self, clip_model, features_list, image_size=518, variant="a1_mp_bs", device="cuda"):
        super().__init__()
        self.model = clip_model
        self.features_list = features_list
        self.image_size = image_size
        self.variant = variant
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

        # Module 1: MP-BS (present in A1, A2, A3)
        self.mp_bs = MultiPrototypeBackgroundDisentangler(
            embed_dim=self.embed_dim, num_layers=self.num_layers, num_prototypes=6, temp=0.07
        )

        # Module 2: WD-SAF (present in A2, A3; A1 uses original baseline point-wise MLP SAF)
        if variant == "a1_mp_bs":
            self.layer_transforms = setup_feature_transforms(features_list, device, self.embed_dim)
            self.use_wd_saf = False
        elif variant in ("a2_wd_saf", "a3_agds_wgd"):
            self.layer_transforms = nn.ModuleDict({
                f"layer_{l_idx}": WaveletDirectionalSAF(
                    embed_dim=self.embed_dim, bottleneck_dim=256, dropout=0.1
                )
                for l_idx in features_list
            })
            self.use_wd_saf = True
        else:
            raise ValueError(f"Unsupported standalone ablation variant: {variant}")

        # Module 3: AGDS-WGD (present in A3; None in A1, A2)
        if variant == "a3_agds_wgd":
            patch_grid = image_size // 14
            self.decoder = AnomalyGuidedDynamicScanDecoder(
                embed_dim=self.embed_dim, num_layers=self.num_layers, patch_grid=patch_grid, hidden_dim=64
            )
        else:
            self.decoder = None

    def get_trainable_state_dict(self):
        state = {
            "anomaly_token": self.model.visual.anomaly_token.detach().cpu(),
            "normal_token": self.model.visual.normal_token.detach().cpu(),
            "anomaly_pos": self.model.visual.anomaly_pos.detach().cpu(),
            "normal_pos": self.model.visual.normal_pos.detach().cpu(),
            "ln_post": self.model.visual.ln_post.state_dict(),
            "layer_transforms": self.layer_transforms.state_dict(),
            "cross_attn": self.cross_attn.state_dict(),
            "mp_bs": self.mp_bs.state_dict(),
            "variant": self.variant,
        }
        if self.decoder is not None:
            state["decoder"] = self.decoder.state_dict()
        return state

    def load_trainable_state_dict(self, state, device):
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
        if self.decoder is not None and "decoder" in state:
            self.decoder.load_state_dict(state["decoder"])

    def build_optimizer(self, lr: float):
        params = [
            {"params": [self.model.visual.anomaly_token], "lr": lr, "weight_decay": 0.01},
            {"params": [self.model.visual.normal_token], "lr": lr, "weight_decay": 0.01},
        ]
        ln_post = getattr(self.model.visual, "ln_post", None)
        if ln_post is not None:
            params.append({"params": [ln_post.weight, ln_post.bias], "lr": lr * 0.1, "weight_decay": 0.01})

        for t in self.layer_transforms.values():
            params.append({"params": t.parameters(), "lr": lr * 0.1, "weight_decay": 0.01})

        params.append({"params": self.cross_attn.parameters(), "lr": lr * 0.1, "weight_decay": 0.01})
        params.append({"params": self.mp_bs.parameters(), "lr": lr * 0.2, "weight_decay": 0.01})

        if self.decoder is not None:
            params.append({"params": self.decoder.parameters(), "lr": lr * 0.2, "weight_decay": 0.01})

        return torch.optim.AdamW(params, betas=(0.9, 0.999))

    def forward_step(self, image: torch.Tensor):
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
            if self.use_wd_saf:
                recal = self.layer_transforms[t_key](patch_feat)
            else:
                B, N, D = patch_feat.shape
                recal = self.layer_transforms[t_key](patch_feat.reshape(-1, D)).view(B, N, D)

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
            # Residual MP-BS map combining multi-prototype alignment and background suppression
            res_lowres_map = 0.65 * pre_map + 0.35 * post_map
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

        if self.decoder is None:
            # A1 & A2: Bilinearly upsample each 37x37 layer anomaly map to 518x518
            layer_up_maps = []
            similarity_map_list = []
            for l_map in lowres_maps:
                up_m = F.interpolate(
                    l_map.unsqueeze(1), size=(self.image_size, self.image_size), mode="bilinear", align_corners=False
                ).squeeze(1)
                layer_up_maps.append(up_m)
                sig_m = torch.sigmoid(up_m)
                similarity_map_list.append(torch.stack([1.0 - sig_m, sig_m], dim=1))
            final_anomaly_map = torch.stack(layer_up_maps, dim=0).sum(dim=0)
            prob_map = None
            log_conf = None
        else:
            # A3: High-resolution AGDS-WGD decoder
            top_patches = recal_patches[-1]
            final_anomaly_map, layer_up_maps, log_conf = self.decoder(
                image, lowres_maps, top_patches, self.image_size
            )
            similarity_map_list = []
            for up_m in layer_up_maps:
                sig_m = torch.sigmoid(up_m)
                similarity_map_list.append(torch.stack([1.0 - sig_m, sig_m], dim=1))
            prob_map = torch.stack([torch.sigmoid(m) for m in layer_up_maps], dim=0).mean(dim=0)

        # A1, A2, A3 all use fixed Top-1% pooling (DynamicAreaScorer is added in A4)
        img_scores = reduce_anomaly_map(final_anomaly_map, mode="topk_mean", topk_ratio=DEFAULT_TOPK_RATIO)
        return {
            "final_anomaly_map": final_anomaly_map,
            "pre_sum": pre_sum,
            "post_sum": post_sum,
            "prob_map": prob_map,
            "anomaly_maps_list": layer_up_maps,
            "similarity_map_list": similarity_map_list,
            "img_scores": img_scores,
            "final_anomaly_feat": anomaly_features_list[-1],
            "final_normal_feat": normal_features_list[-1],
            "log_conf": log_conf,
        }


@torch.no_grad()
def evaluate_standalone_ablation(wrapper: StandaloneAblationWrapper, dataloader, device, sigma: float = 4.0, compute_pro: bool = True):
    wrapper.eval()
    gt_sp_list, masks_list = [], []
    pre_list, post_list, prob_list = [], [], []
    class_masks_dict = {c: [] for c in DEFECT_CLASSES}

    for items in tqdm(dataloader, desc=f"Eval {wrapper.variant}", leave=False):
        image = items["img"].to(device)
        gt_mask = (items["img_mask"] > 0.5).float()
        labels = items["anomaly"]

        with autocast(enabled=(device.type == "cuda")):
            out = wrapper.forward_step(image)
            pre_b = out["pre_sum"].float().cpu().numpy()
            post_b = out["post_sum"].float().cpu().numpy()
            prob_b = out["prob_map"].float().cpu().numpy() if out["prob_map"] is not None else None

        for b in range(image.shape[0]):
            pre_list.append(gaussian_filter(pre_b[b], sigma=sigma) if sigma > 0 else pre_b[b])
            post_list.append(gaussian_filter(post_b[b], sigma=sigma) if sigma > 0 else post_b[b])
            if prob_b is not None:
                prob_list.append(gaussian_filter(prob_b[b], sigma=1.5))
            masks_list.append(gt_mask[b, 0].cpu().numpy())
            gt_sp_list.append(int(labels[b].item()))
            if "class_masks" in items:
                for cname in DEFECT_CLASSES:
                    class_masks_dict[cname].append(items["class_masks"][cname][b, 0].cpu().numpy())

    masks_np = np.stack(masks_list, axis=0)
    gt_sp = np.array(gt_sp_list)
    pre_n = norm01(np.stack(pre_list, axis=0))
    post_n = norm01(np.stack(post_list, axis=0))
    cos_hybrid = norm01(0.65 * pre_n + 0.35 * post_n)

    if wrapper.decoder is None:
        # A1 and A2: no high-res decoder
        amaps_norm = cos_hybrid
    else:
        # A3: linear additive fusion of high-res decoder probability map (without A4's multiplicative confidence gate)
        prob_n = norm01(np.stack(prob_list, axis=0))
        amaps_norm = norm01(0.82 * cos_hybrid + 0.18 * prob_n)

    # Fixed Top-1% pooling for A1, A2, A3
    flat_norm = amaps_norm.reshape(amaps_norm.shape[0], -1)
    k_top = max(1, int(np.ceil(flat_norm.shape[1] * DEFAULT_TOPK_RATIO)))
    final_sp = np.sort(flat_norm, axis=1)[:, -k_top:].mean(axis=1)

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

    if total_pos > 0 and total_neg > 0:
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
    else:
        p_auroc = 0.0
        p_ap = 0.0

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
    best_f1 = float(f1_arr[best_idx])
    best_iou = float(iou_arr[best_idx])
    best_th = float(ths[best_idx])

    p_f1 = best_f1 * 100.0
    p_iou = best_iou * 100.0
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
        "Best-Th": round(best_th, 2),
        "Per-Class": class_metrics,
    }
    return metrics


def train_and_evaluate_ablation_variant(
    variant: str,
    dataset_name: str,
    save_dir: str,
    epochs: int = 8,
    batch_size: int = 4,
    accum_steps: int = 2,
    learning_rate: float = 0.001,
    sigma: float = 4.0,
    seed: int = 111,
    device_str: str = "cuda:0",
    smoke_test: bool = False,
):
    setup_seed(seed)
    device = torch.device(device_str if torch.cuda.is_available() and "cuda" in device_str else "cpu")
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, "train_eval.log")

    def log_print(msg):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as lf:
            lf.write(line + "\n")

    metrics_json_path = os.path.join(save_dir, "test_metrics.json")
    best_ckpt_path = os.path.join(save_dir, "best_model.pth")
    if os.path.exists(metrics_json_path) and os.path.exists(best_ckpt_path) and not smoke_test:
        log_print(f"Found existing completed run at {save_dir}, loading metrics...")
        with open(metrics_json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    log_print(f"=== Starting True Standalone Ablation: variant={variant} | train_ds={dataset_name} ===")

    clip_model, _ = VisualAD_lib.load("ViT-L/14@336px", device=device)
    clip_model.to(device)

    features_list = [6, 12, 18, 24]
    image_size = 518
    wrapper = StandaloneAblationWrapper(
        clip_model=clip_model,
        features_list=features_list,
        image_size=image_size,
        variant=variant,
        device=device,
    ).to(device)

    trainable_params = sum(p.numel() for p in wrapper.parameters() if p.requires_grad)
    log_print(f"Trainable parameters ({variant}): {trainable_params:,}")

    dummy_args = argparse.Namespace(image_size=image_size)
    preprocess, _ = get_transform(dummy_args)

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

    if smoke_test:
        train_ds.data_all = _make_balanced_smoke_subset(train_ds.data_all, 2, 2)
        val_ds.data_all = _make_balanced_smoke_subset(val_ds.data_all, 2, 2)
        test_ds.data_all = _make_balanced_smoke_subset(test_ds.data_all, 2, 2)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=4,
        pin_memory=(device.type == "cuda"), persistent_workers=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=4,
        pin_memory=(device.type == "cuda"), persistent_workers=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=4,
        pin_memory=(device.type == "cuda")
    )

    optimizer = wrapper.build_optimizer(learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=learning_rate * 0.1
    )

    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()
    loss_contra = ContrastiveLoss(temperature=0.1, margin=0.5)

    amp_enabled = (device.type == "cuda")
    scaler = GradScaler(enabled=amp_enabled)

    best_val_score = -1.0
    history = []

    for epoch in range(epochs):
        wrapper.train()
        epoch_losses = []
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"{variant} Ep {epoch+1}/{epochs}", leave=False)
        for step_idx, items in pbar:
            image = items["img"].to(device, non_blocking=True)
            label = items["anomaly"].float().to(device, non_blocking=True)
            gt = (items["img_mask"].to(device, non_blocking=True) > 0.5).float()

            with autocast(enabled=amp_enabled):
                out = wrapper.forward_step(image)
                fa_norm = F.normalize(out["final_anomaly_feat"], dim=1, eps=1e-8)
                fn_norm = F.normalize(out["final_normal_feat"], dim=1, eps=1e-8)
                l_contra = loss_contra(fa_norm, fn_norm)
                l_cls = F.binary_cross_entropy_with_logits(out["img_scores"].view(-1), label.view(-1))

                # A1, A2, A3 all use standard Focal + Dice + 0.1 * L_div (CW-Loss is added in A4)
                seg_losses = []
                for sim_m in out["similarity_map_list"]:
                    seg_losses.append(loss_focal(sim_m, gt))
                    seg_losses.append(loss_dice(sim_m[:, 1, :, :], gt))
                l_seg = sum(seg_losses)
                l_div = wrapper.mp_bs.diversity_loss()
                total_loss = l_cls + l_seg + l_contra + 0.1 * l_div

                loss_scaled = total_loss / accum_steps

            if torch.isnan(loss_scaled) or torch.isinf(loss_scaled):
                log_print(f"Warning: NaN/Inf loss at epoch {epoch+1} step {step_idx}, skipping step")
                optimizer.zero_grad(set_to_none=True)
                continue

            if amp_enabled:
                scaler.scale(loss_scaled).backward()
            else:
                loss_scaled.backward()

            if (step_idx + 1) % accum_steps == 0 or (step_idx + 1) == len(train_loader):
                if amp_enabled:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in wrapper.parameters() if p.requires_grad], max_norm=1.0
                )
                if amp_enabled:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            epoch_losses.append(float(total_loss.detach().item()))

        scheduler.step()
        mean_train_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0

        val_metrics = evaluate_standalone_ablation(wrapper, val_loader, device, sigma=sigma, compute_pro=False)
        val_key_score = (
            val_metrics["Image-AUROC"] + val_metrics["Pixel-AUROC"] + val_metrics["Pixel-F1"] + val_metrics["Pixel-AP"]
        ) / 4.0

        log_print(
            f"Epoch [{epoch+1}/{epochs}] loss={mean_train_loss:.4f} | "
            f"Val I-AUROC={val_metrics['Image-AUROC']:.2f} P-AUROC={val_metrics['Pixel-AUROC']:.2f} "
            f"P-F1={val_metrics['Pixel-F1']:.2f} P-AP={val_metrics['Pixel-AP']:.2f}"
        )
        history.append({"epoch": epoch + 1, "train_loss": round(mean_train_loss, 4), "val": val_metrics})

        if val_key_score >= best_val_score:
            best_val_score = val_key_score
            torch.save(wrapper.get_trainable_state_dict(), best_ckpt_path)

    final_ckpt_path = os.path.join(save_dir, "final_model.pth")
    torch.save(wrapper.get_trainable_state_dict(), final_ckpt_path)

    wrapper.load_trainable_state_dict(torch.load(best_ckpt_path, map_location=device), device)
    log_print(f"Loaded best validation checkpoint from {best_ckpt_path} for final Test evaluation...")

    in_domain_test_metrics = evaluate_standalone_ablation(wrapper, test_loader, device, sigma=sigma, compute_pro=True)
    log_print(f"[In-Domain Test: {dataset_name}] {json.dumps(in_domain_test_metrics)}")

    other_ds_name = (
        "Bridge512_balance_CODEBRIM" if dataset_name == "Bridge512_balance_S2DS" else "Bridge512_balance_S2DS"
    )
    other_root = f"/home/skye/data/Skye/databases/{other_ds_name}"
    other_meta = f"/home/skye/data/Skye/visualad/datasets/{other_ds_name}_meta.json"
    if not os.path.exists(other_meta):
        build_bridge_meta_json(other_root, other_meta)
    other_test_ds = BridgeAnomalyDataset(
        other_root, split="test", image_size=image_size,
        preprocess=preprocess, return_class_masks=True, meta_json_path=other_meta
    )
    if smoke_test:
        other_test_ds.data_all = _make_balanced_smoke_subset(other_test_ds.data_all, 2, 2)

    other_test_loader = torch.utils.data.DataLoader(
        other_test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=(device.type == "cuda")
    )
    cross_domain_test_metrics = evaluate_standalone_ablation(wrapper, other_test_loader, device, sigma=sigma, compute_pro=True)
    log_print(f"[Cross-Domain Zero-Shot Test: {dataset_name} -> {other_ds_name}] {json.dumps(cross_domain_test_metrics)}")

    summary = {
        "variant": variant,
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
    with open(metrics_json_path, "w", encoding="utf-8") as mf:
        json.dump(summary, mf, indent=2)
    log_print(f"Saved complete results to {metrics_json_path}")
    return summary


def aggregate_all_true_ablation_metrics():
    """
    Combines independently trained A0 (baseline), A1 (a1_mp_bs), A2 (a2_wd_saf), A3 (a3_agds_wgd), and A4 (ours)
    into outputs/ablation_metrics_true_standalone.json without overwriting outputs/ablation_metrics_inference_variants.json.
    """
    out_root = os.path.join(CUR_DIR, "outputs")
    stage_map = [
        ("A0_Baseline", "baseline", "A0: Baseline (VisualAD)"),
        ("A1_MP_BS", "a1_mp_bs", "A1: + MP-BS (Trained Standalone)"),
        ("A2_WD_SAF", "a2_wd_saf", "A2: + MP-BS + WD-SAF (Trained Standalone)"),
        ("A3_AGDS_WGD", "a3_agds_wgd", "A3: + MP-BS + WD-SAF + AGDS-WGD (Trained Standalone)"),
        ("A4_Full_Ours", "ours", "A4: Full Bridge-VisualAD (Trained Standalone)"),
    ]

    combined = {
        "S2DS_in_domain": {},
        "S2DS_to_CODEBRIM_cross_domain": {},
        "CODEBRIM_in_domain": {},
        "CODEBRIM_to_S2DS_cross_domain": {},
    }

    for short_ds, in_key, cross_key in [
        ("S2DS", "S2DS_in_domain", "S2DS_to_CODEBRIM_cross_domain"),
        ("CODEBRIM", "CODEBRIM_in_domain", "CODEBRIM_to_S2DS_cross_domain"),
    ]:
        for stage_key, folder_prefix, label in stage_map:
            m_path = os.path.join(out_root, f"{folder_prefix}_{short_ds}", "test_metrics.json")
            if not os.path.exists(m_path):
                continue
            with open(m_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            in_m = dict(data["in_domain_test"])
            in_m["Variant-Name"] = label
            in_m["Trainable-Params"] = data.get("trainable_params")
            cross_m = dict(data["cross_domain_test"])
            cross_m["Variant-Name"] = label
            cross_m["Trainable-Params"] = data.get("trainable_params")

            combined[in_key][stage_key] = in_m
            combined[cross_key][stage_key] = cross_m

    target_json = os.path.join(out_root, "ablation_metrics_true_standalone.json")
    with open(target_json, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
    print(f"Saved aggregated true standalone ablation metrics to {target_json}")
    return combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Standalone Ablation Trainer (A1, A2, A3)")
    parser.add_argument("--variant", type=str, default="all", choices=["all", "a1_mp_bs", "a2_wd_saf", "a3_agds_wgd"])
    parser.add_argument("--dataset_name", type=str, default="all",
                        choices=["all", "Bridge512_balance_S2DS", "Bridge512_balance_CODEBRIM"])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--accum_steps", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--sigma", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()

    variants = ["a1_mp_bs", "a2_wd_saf", "a3_agds_wgd"] if args.variant == "all" else [args.variant]
    datasets = (
        ["Bridge512_balance_S2DS", "Bridge512_balance_CODEBRIM"]
        if args.dataset_name == "all"
        else [args.dataset_name]
    )

    short_ds = {
        "Bridge512_balance_S2DS": "S2DS",
        "Bridge512_balance_CODEBRIM": "CODEBRIM",
    }

    base_out = "outputs_smoke" if args.smoke_test else "outputs"
    for ds in datasets:
        for var in variants:
            out_dir = os.path.join(CUR_DIR, base_out, f"{var}_{short_ds[ds]}")
            train_and_evaluate_ablation_variant(
                variant=var,
                dataset_name=ds,
                save_dir=out_dir,
                epochs=1 if args.smoke_test else args.epochs,
                batch_size=args.batch_size,
                accum_steps=args.accum_steps,
                learning_rate=args.learning_rate,
                sigma=args.sigma,
                seed=args.seed,
                device_str=args.device,
                smoke_test=args.smoke_test,
            )

    if not args.smoke_test:
        aggregate_all_true_ablation_metrics()
