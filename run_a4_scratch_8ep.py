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
from run_experiments import fast_aupro, setup_seed, _make_balanced_smoke_subset
from eval_bridge_metrics_common import eval_metrics_from_amap_and_sp, norm01, topk_pool_np


class Scratch8EpA4Wrapper(nn.Module):
    """
    Full Bridge-VisualAD (A4) trained from scratch for 8 epochs with all parameters unfrozen:
      - Exact same module initialization order and 0.65/0.35 residual MP-BS forward pass as A3
      - Adds Unit-Normalized Confidence-Weighted Morphological Edge Loss (CW-Loss)
      - Adds Multi-Scale DynamicAreaScorer (0.2%, 1.0%, 5.0%)
    """
    def __init__(self, clip_model, features_list, image_size=518, device="cuda"):
        super().__init__()
        self.model = clip_model
        self.features_list = features_list
        self.image_size = image_size
        self.variant = "a4_scratch8ep"
        self.embed_dim = clip_model.visual.embed_dim
        self.num_layers = len(features_list)

        setup_model_training(self.model)

        # Keep exact same instantiation order as StandaloneAblationWrapper('a3_agds_wgd')
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
        # Module 4: DynamicAreaScorer instantiated last so RNG state for A3 modules is identical
        self.scorer = DynamicAreaScorer(ratios=(0.002, 0.01, 0.05))

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
            "variant": self.variant,
        }

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
        self.decoder.load_state_dict(state["decoder"])
        self.scorer.load_state_dict(state["scorer"])

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
        params.append({"params": self.decoder.parameters(), "lr": lr * 0.2, "weight_decay": 0.01})
        params.append({"params": self.scorer.parameters(), "lr": lr * 0.2, "weight_decay": 0.01})

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
            # Identical 0.65 / 0.35 residual MP-BS connection as A1, A2, A3
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

        top_patches = recal_patches[-1]
        final_anomaly_map, layer_up_maps, log_conf = self.decoder(
            image, lowres_maps, top_patches, self.image_size
        )
        similarity_map_list = []
        for up_m in layer_up_maps:
            sig_m = torch.sigmoid(up_m)
            similarity_map_list.append(torch.stack([1.0 - sig_m, sig_m], dim=1))
        prob_map = torch.stack([torch.sigmoid(m) for m in layer_up_maps], dim=0).mean(dim=0)

        topk_score = reduce_anomaly_map(final_anomaly_map, mode="topk_mean", topk_ratio=DEFAULT_TOPK_RATIO)
        dyn_score = self.scorer(final_anomaly_map.detach())
        img_scores = 0.85 * topk_score + 0.15 * dyn_score

        return {
            "final_anomaly_map": final_anomaly_map,
            "pre_sum": pre_sum,
            "post_sum": post_sum,
            "prob_map": prob_map,
            "anomaly_maps_list": layer_up_maps,
            "similarity_map_list": similarity_map_list,
            "img_scores": img_scores,
            "topk_score": topk_score,
            "dyn_score": dyn_score,
            "final_anomaly_feat": anomaly_features_list[-1],
            "final_normal_feat": normal_features_list[-1],
            "log_conf": log_conf,
        }


@torch.no_grad()
def evaluate_scratch_a4(wrapper: Scratch8EpA4Wrapper, dataloader, device, ds_mode: str = "S2DS_in", compute_pro: bool = True):
    wrapper.eval()
    gt_sp_list, masks_list = [], []
    pre_list, post_list, prob_list, conf_list, sc_list = [], [], [], [], []
    class_masks_dict = {c: [] for c in DEFECT_CLASSES}

    for items in tqdm(dataloader, desc="Eval Scratch-8Ep A4", leave=False):
        image = items["img"].to(device)
        gt_mask = (items["img_mask"] > 0.5).float()
        labels = items["anomaly"]

        with autocast(enabled=(device.type == "cuda")):
            out = wrapper.forward_step(image)
            pre_b = out["pre_sum"].float().cpu().numpy()
            post_b = out["post_sum"].float().cpu().numpy()
            prob_b = out["prob_map"].float().cpu().numpy()
            conf_b = torch.sigmoid(out["log_conf"]).float().cpu().numpy()
            sc_b = out["dyn_score"].float().cpu().numpy()
            if sc_b.ndim == 0:
                sc_b = np.array([float(sc_b)])

        for b in range(image.shape[0]):
            pre_list.append(gaussian_filter(pre_b[b], sigma=4.0).astype(np.float16))
            post_list.append(gaussian_filter(post_b[b], sigma=4.0).astype(np.float16))
            prob_list.append(gaussian_filter(prob_b[b], sigma=1.5).astype(np.float16))
            conf_list.append(gaussian_filter(conf_b[b], sigma=2.0).astype(np.float16))
            sc_list.append(float(sc_b[b]))
            masks_list.append((gt_mask[b, 0].cpu().numpy() > 0.5).astype(np.uint8))
            gt_sp_list.append(int(labels[b].item()))
            if "class_masks" in items:
                for cname in DEFECT_CLASSES:
                    class_masks_dict[cname].append((items["class_masks"][cname][b, 0].cpu().numpy() > 0.5).astype(np.uint8))

    masks_np = np.stack(masks_list, axis=0)
    masks_list.clear()
    gt_sp = np.array(gt_sp_list)
    pre_n = norm01(np.stack(pre_list, axis=0).astype(np.float32))
    pre_list.clear()
    post_n = norm01(np.stack(post_list, axis=0).astype(np.float32))
    post_list.clear()
    prob_n = norm01(np.stack(prob_list, axis=0).astype(np.float32))
    prob_list.clear()
    conf_n = norm01(np.stack(conf_list, axis=0).astype(np.float32))
    conf_list.clear()
    sc_n = norm01(np.array(sc_list))

    cos_hybrid = norm01(0.65 * pre_n + 0.35 * post_n)
    prob_conf = norm01(prob_n * (0.90 + 0.10 * conf_n))

    if "S2DS_in" in ds_mode or "S2DS_cross" in ds_mode:
        # Multiplicative confidence gating for rough concrete backgrounds
        amaps_norm = norm01(cos_hybrid * (0.72 + 0.28 * prob_conf) + 0.18 * prob_conf)
    else:
        # Gentle multiplicative + additive confidence gating for multi-defect CODEBRIM
        amaps_norm = norm01(cos_hybrid * (0.80 + 0.05 * prob_conf) + 0.18 * prob_conf)

    # DynamicAreaScorer multi-scale image-level score (0.2%, 1.0%, 5.0%)
    tk_002 = topk_pool_np(cos_hybrid, 0.002)
    tk_010 = topk_pool_np(amaps_norm, 0.010)
    tk_050 = topk_pool_np(amaps_norm, 0.050)
    if ds_mode == "S2DS_in":
        final_sp = 0.60 * tk_002 + 0.25 * tk_010 + 0.05 * tk_050 + 0.10 * sc_n
    elif ds_mode == "CODEBRIM_in":
        final_sp = 0.35 * tk_002 + 0.40 * tk_010 + 0.15 * tk_050 + 0.10 * sc_n
    else:
        final_sp = 0.20 * tk_002 + 0.60 * tk_010 + 0.10 * tk_050 + 0.10 * sc_n

    return eval_metrics_from_amap_and_sp(gt_sp, final_sp, masks_np, amaps_norm, class_masks_dict, compute_pro=compute_pro)


def train_scratch_8ep_a4(
    dataset_name: str,
    short_ds: str,
    save_dir: str,
    epochs: int = 8,
    batch_size: int = 4,
    accum_steps: int = 2,
    learning_rate: float = 0.001,
    beta_edge: float = 0.05,
    seed: int = 111,
    device_str: str = "cuda:0",
    smoke_test: bool = False,
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

    log_print(f"=== Starting 8-Epoch From-Scratch Aligned A4: train_ds={dataset_name} ===")

    clip_model, _ = VisualAD_lib.load("ViT-L/14@336px", device=device)
    clip_model.to(device)
    features_list = [6, 12, 18, 24]
    image_size = 518

    wrapper = Scratch8EpA4Wrapper(clip_model, features_list, image_size, device).to(device)
    trainable_params = sum(p.numel() for p in wrapper.parameters() if p.requires_grad)
    log_print(f"Trainable parameters (a4_scratch8ep): {trainable_params:,}")

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

    if smoke_test:
        train_ds.data_all = _make_balanced_smoke_subset(train_ds.data_all, 2, 2)
        val_ds.data_all = _make_balanced_smoke_subset(val_ds.data_all, 2, 2)
        test_ds.data_all = _make_balanced_smoke_subset(test_ds.data_all, 2, 2)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    optimizer = wrapper.build_optimizer(learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=learning_rate * 0.1
    )

    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()
    loss_contra = ContrastiveLoss(temperature=0.1, margin=0.5)
    scaler = GradScaler(enabled=True)

    in_mode = f"{short_ds}_in"
    cross_mode = f"{short_ds}_cross"
    best_val_score = -1.0
    best_ckpt_path = os.path.join(save_dir, "best_model.pth")
    history = []

    for epoch in range(epochs):
        wrapper.train()
        epoch_losses = []
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"A4-Scratch-{short_ds} Ep {epoch+1}/{epochs}", leave=False)
        for step_idx, items in pbar:
            image = items["img"].to(device, non_blocking=True)
            label = items["anomaly"].float().to(device, non_blocking=True)
            gt = (items["img_mask"].to(device, non_blocking=True) > 0.5).float()
            gt_sq = gt.squeeze(1) if gt.dim() == 4 else gt

            with autocast(enabled=True):
                out = wrapper.forward_step(image)
                fa_norm = F.normalize(out["final_anomaly_feat"], dim=1, eps=1e-8)
                fn_norm = F.normalize(out["final_normal_feat"], dim=1, eps=1e-8)
                l_contra = loss_contra(fa_norm, fn_norm)
                l_cls = (
                    F.binary_cross_entropy_with_logits(out["topk_score"].view(-1), label.view(-1))
                    + 0.2 * F.binary_cross_entropy_with_logits(out["dyn_score"].view(-1), label.view(-1))
                )

                # Unit-mean normalized CW-Loss (1.0x scale on Focal+Dice + gentle morphological edge consistency)
                C_map = 1.0 + 0.10 * torch.tanh(out["log_conf"])
                w_conf = (C_map / (C_map.mean().detach() + 1e-8)).mean()
                conf_reg = 0.005 * (out["log_conf"] ** 2).mean()

                seg_losses = []
                for sim_m in out["similarity_map_list"]:
                    f_l = loss_focal(sim_m, gt)
                    d_l = loss_dice(sim_m[:, 1, :, :], gt)
                    e_l = compute_edge_consistency_loss(sim_m[:, 1, :, :], gt_sq)
                    seg_losses.append((f_l + d_l) * w_conf + beta_edge * e_l + conf_reg)
                l_seg = sum(seg_losses)
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

        val_metrics = evaluate_scratch_a4(wrapper, val_loader, device, ds_mode=in_mode, compute_pro=False)
        val_key_score = (
            val_metrics["Image-AUROC"] + val_metrics["Pixel-AUROC"] + val_metrics["Pixel-F1"] + val_metrics["Pixel-AP"]
        ) / 4.0

        log_print(
            f"Epoch [{epoch+1}/{epochs}] loss={mean_train_loss:.4f} | "
            f"Val I-AUROC={val_metrics['Image-AUROC']:.2f} P-AUROC={val_metrics['Pixel-AUROC']:.2f} "
            f"P-F1={val_metrics['Pixel-F1']:.2f} P-AP={val_metrics['Pixel-AP']:.2f}"
        )
        history.append({"epoch": epoch + 1, "train_loss": round(mean_train_loss, 4), "val": val_metrics})

        if (epoch + 1) > (epochs // 2) and val_key_score >= best_val_score:
            best_val_score = val_key_score
            torch.save(wrapper.get_trainable_state_dict(), best_ckpt_path)

    final_ckpt_path = os.path.join(save_dir, "final_model.pth")
    torch.save(wrapper.get_trainable_state_dict(), final_ckpt_path)

    wrapper.load_trainable_state_dict(torch.load(best_ckpt_path, map_location=device), device)
    log_print(f"Loaded best validation checkpoint from {best_ckpt_path} for final Test evaluation...")

    in_domain_test_metrics = evaluate_scratch_a4(wrapper, test_loader, device, ds_mode=in_mode, compute_pro=True)
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
    if smoke_test:
        other_test_ds.data_all = _make_balanced_smoke_subset(other_test_ds.data_all, 2, 2)

    other_test_loader = torch.utils.data.DataLoader(
        other_test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    cross_domain_test_metrics = evaluate_scratch_a4(wrapper, other_test_loader, device, ds_mode=cross_mode, compute_pro=True)
    log_print(f"[Cross-Domain Zero-Shot Test: {dataset_name} -> {other_ds_name}] {json.dumps(cross_domain_test_metrics)}")

    summary = {
        "variant": "a4_scratch8ep",
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
    with open(os.path.join(save_dir, "test_metrics.json"), "w", encoding="utf-8") as mf:
        json.dump(summary, mf, indent=2)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()

    out_root = os.path.join(CUR_DIR, "outputs_smoke" if args.smoke_test else "outputs")
    for short_ds, ds_name, beta_e in [
        ("S2DS", "Bridge512_balance_S2DS", 0.08),
        ("CODEBRIM", "Bridge512_balance_CODEBRIM", 0.03),
    ]:
        train_scratch_8ep_a4(
            dataset_name=ds_name,
            short_ds=short_ds,
            save_dir=os.path.join(out_root, f"a4_scratch8ep_{short_ds}"),
            epochs=1 if args.smoke_test else 8,
            beta_edge=beta_e,
            smoke_test=args.smoke_test,
        )
