import argparse
import json
import os
import sys
import time
import numpy as np
from scipy.ndimage import gaussian_filter
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
    DynamicAreaScorer,
    MultiPrototypeBackgroundDisentangler,
    WaveletDirectionalSAF,
    compute_confidence_weighted_seg_loss,
)
from run_experiments import (
    evaluate_model,
    generate_baseline_anomaly_map,
    setup_seed,
    _make_balanced_smoke_subset,
)


class StandaloneAblationWrapper(nn.Module):
    """
    Supports true standalone training for each ablation stage:
      - 'a1_mp_bs'    : Baseline + MP-BS (Multi-Prototype & Background Suppression)
      - 'a2_wd_saf'   : Baseline + MP-BS + WD-SAF (Wavelet-Directional SAF)
      - 'a3_agds_wgd' : Baseline + MP-BS + WD-SAF + AGDS-WGD (Dynamic Scan & Wavelet Decoder, trained w/ standard Focal+Dice)
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

        self.scorer = None

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
        recal_patches = []
        for idx_l, patch_feat in enumerate(patch_features_list):
            t_key = f"layer_{self.features_list[idx_l]}"
            if self.use_wd_saf:
                recal = self.layer_transforms[t_key](patch_feat)
            else:
                B, N, D = patch_feat.shape
                recal = self.layer_transforms[t_key](patch_feat.reshape(-1, D)).view(B, N, D)
            supp_patch, lowres_map = self.mp_bs.suppress_background(
                recal, anomaly_features_list[idx_l], normal_features_list[idx_l], idx_l
            )
            recal_patches.append(supp_patch)
            lowres_maps.append(lowres_map)

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

        # A1, A2, A3 all use fixed Top-1% pooling (DynamicAreaScorer is added in A4)
        img_scores = reduce_anomaly_map(final_anomaly_map, mode="topk_mean", topk_ratio=DEFAULT_TOPK_RATIO)
        return {
            "final_anomaly_map": final_anomaly_map,
            "anomaly_maps_list": layer_up_maps,
            "similarity_map_list": similarity_map_list,
            "img_scores": img_scores,
            "final_anomaly_feat": anomaly_features_list[-1],
            "final_normal_feat": normal_features_list[-1],
            "log_conf": log_conf,
        }


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

    # Skip if already completed
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

        val_metrics = evaluate_model(wrapper, val_loader, device, sigma=sigma, compute_pro=False)
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

    in_domain_test_metrics = evaluate_model(wrapper, test_loader, device, sigma=sigma, compute_pro=True)
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
    cross_domain_test_metrics = evaluate_model(wrapper, other_test_loader, device, sigma=sigma, compute_pro=True)
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

    for ds in datasets:
        for var in variants:
            out_dir = os.path.join(CUR_DIR, "outputs", f"{var}_{short_ds[ds]}")
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
