import argparse
import json
import os
import random
import sys
import time
import numpy as np
import scipy.ndimage as ndimage
from scipy.ndimage import gaussian_filter
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

# Add VisualAD_repo to sys.path
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


def setup_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"


def fast_aupro(masks: np.ndarray, amaps: np.ndarray, num_th: int = 50, max_fpr: float = 0.3) -> float:
    """
    Fast exact histogram-vectorized AUPRO computation over connected components.
    masks: [N, H, W] binary {0, 1}
    amaps: [N, H, W] float
    """
    if masks.sum() == 0:
        return 0.0
    min_th, max_th = float(amaps.min()), float(amaps.max())
    if max_th <= min_th + 1e-12:
        return 0.0

    thresholds = np.linspace(min_th, max_th, num_th, endpoint=False, dtype=np.float32)
    bins = np.concatenate([[-np.inf], np.nextafter(thresholds, np.inf), [np.inf]])

    comp_pros_list = []
    for i in range(len(masks)):
        if masks[i].sum() == 0:
            continue
        labeled, _ = ndimage.label(masks[i] > 0.5)
        objs = ndimage.find_objects(labeled)
        for comp_id, slc in enumerate(objs, start=1):
            if slc is None:
                continue
            sub_mask = (labeled[slc] == comp_id)
            area = float(sub_mask.sum())
            if area > 0:
                comp_vals = amaps[i][slc][sub_mask]
                c_hist, _ = np.histogram(comp_vals, bins=bins)
                c_tp = np.cumsum(c_hist[::-1])[::-1][1:].astype(np.float64)
                comp_pros_list.append(c_tp / area)

    if not comp_pros_list:
        return 0.0

    neg_vals = amaps[masks <= 0.5]
    total_neg = float(len(neg_vals))
    if total_neg == 0:
        return 0.0

    neg_hist, _ = np.histogram(neg_vals, bins=bins)
    fp_arr = np.cumsum(neg_hist[::-1])[::-1][1:].astype(np.float64)
    fpr_arr = fp_arr / total_neg
    mean_pros = np.mean(np.stack(comp_pros_list, axis=0), axis=0)

    valid = (fpr_arr <= max_fpr)
    fprs = fpr_arr[valid]
    pros = mean_pros[valid]

    if len(fprs) < 2:
        return 0.0
    order = np.argsort(fprs)
    fprs = fprs[order]
    pros = pros[order]
    fpr_span = fprs.max() - fprs.min()
    if fpr_span <= 1e-12:
        return float(pros.mean())
    fpr_norm = (fprs - fprs.min()) / fpr_span
    return float(auc(fpr_norm, pros))


def generate_baseline_anomaly_map(anomaly_feat_norm, normal_feat_norm, patch_tokens, image_size):
    B, N, C = patch_tokens.shape
    patch_norm = F.normalize(patch_tokens, dim=2, eps=1e-8)
    anom_sim = torch.cosine_similarity(patch_norm, anomaly_feat_norm.unsqueeze(1), dim=2)
    norm_sim = torch.cosine_similarity(patch_norm, normal_feat_norm.unsqueeze(1), dim=2)
    score = anom_sim - norm_sim
    H_p = int(np.sqrt(N))
    lowres_map = score.view(B, H_p, H_p)
    up_map = F.interpolate(
        lowres_map.unsqueeze(1), size=(image_size, image_size), mode="bilinear", align_corners=False
    ).squeeze(1)
    return up_map, lowres_map


class BridgeVisualADWrapper(nn.Module):
    """
    Wraps either:
      - variant='baseline': Official VisualAD (CVPR 2026)
      - variant='ours': Proposed Bridge-VisualAD with 4 domain-specific modular innovations:
          1. MultiPrototypeBackgroundDisentangler (MP-BS)
          2. WaveletDirectionalSAF (WD-SAF)
          3. AnomalyGuidedDynamicScanDecoder (AGDS-WGD)
          4. DynamicAreaScorer & Confidence-Weighted Edge Loss (CW-Loss)
    """
    def __init__(self, clip_model, features_list, image_size=518, variant="baseline", device="cuda"):
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

        if variant == "baseline":
            self.layer_transforms = setup_feature_transforms(features_list, device, self.embed_dim)
            self.mp_bs = None
            self.decoder = None
            self.scorer = None
        elif variant == "ours":
            self.layer_transforms = nn.ModuleDict({
                f"layer_{l_idx}": WaveletDirectionalSAF(
                    embed_dim=self.embed_dim, bottleneck_dim=256, dropout=0.1
                )
                for l_idx in features_list
            })
            self.mp_bs = MultiPrototypeBackgroundDisentangler(
                embed_dim=self.embed_dim, num_layers=self.num_layers, num_prototypes=6, temp=0.07
            )
            patch_grid = image_size // 14
            self.decoder = AnomalyGuidedDynamicScanDecoder(
                embed_dim=self.embed_dim, num_layers=self.num_layers, patch_grid=patch_grid, hidden_dim=64
            )
            self.scorer = DynamicAreaScorer(ratios=(0.002, 0.01, 0.05))
        else:
            raise ValueError(f"Unknown variant: {variant}")

    def get_trainable_state_dict(self):
        state = {
            "anomaly_token": self.model.visual.anomaly_token.detach().cpu(),
            "normal_token": self.model.visual.normal_token.detach().cpu(),
            "anomaly_pos": self.model.visual.anomaly_pos.detach().cpu(),
            "normal_pos": self.model.visual.normal_pos.detach().cpu(),
            "ln_post": self.model.visual.ln_post.state_dict(),
            "layer_transforms": self.layer_transforms.state_dict(),
            "cross_attn": self.cross_attn.state_dict(),
            "variant": self.variant,
        }
        if self.variant == "ours":
            state["mp_bs"] = self.mp_bs.state_dict()
            state["decoder"] = self.decoder.state_dict()
            state["scorer"] = self.scorer.state_dict()
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
        if self.variant == "ours":
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

        if self.variant == "ours":
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

        if self.variant == "baseline":
            anomaly_maps_list = []
            similarity_map_list = []
            for idx_l, patch_feat in enumerate(patch_features_list):
                a_norm = F.normalize(anomaly_features_list[idx_l], dim=1, eps=1e-8)
                n_norm = F.normalize(normal_features_list[idx_l], dim=1, eps=1e-8)
                t_key = f"layer_{self.features_list[idx_l]}"
                B, N, D = patch_feat.shape
                recal_patch = self.layer_transforms[t_key](patch_feat.reshape(-1, D)).view(B, N, D)
                up_map, _ = generate_baseline_anomaly_map(a_norm, n_norm, recal_patch, self.image_size)
                anomaly_maps_list.append(up_map)
                sig_map = torch.sigmoid(up_map)
                similarity_map_list.append(torch.stack([1.0 - sig_map, sig_map], dim=1))

            final_anomaly_map = torch.stack(anomaly_maps_list, dim=0).sum(dim=0)
            img_scores = reduce_anomaly_map(final_anomaly_map, mode="topk_mean", topk_ratio=DEFAULT_TOPK_RATIO)
            return {
                "final_anomaly_map": final_anomaly_map,
                "anomaly_maps_list": anomaly_maps_list,
                "similarity_map_list": similarity_map_list,
                "img_scores": img_scores,
                "final_anomaly_feat": anomaly_features_list[-1],
                "final_normal_feat": normal_features_list[-1],
                "log_conf": None,
            }
        else:
            lowres_maps = []
            recal_patches = []
            for idx_l, patch_feat in enumerate(patch_features_list):
                t_key = f"layer_{self.features_list[idx_l]}"
                recal = self.layer_transforms[t_key](patch_feat)
                supp_patch, lowres_map = self.mp_bs.suppress_background(
                    recal, anomaly_features_list[idx_l], normal_features_list[idx_l], idx_l
                )
                recal_patches.append(supp_patch)
                lowres_maps.append(lowres_map)

            top_patches = recal_patches[-1]
            final_anomaly_map, layer_up_maps, log_conf = self.decoder(
                image, lowres_maps, top_patches, self.image_size
            )
            similarity_map_list = []
            for up_m in layer_up_maps:
                sig_m = torch.sigmoid(up_m)
                similarity_map_list.append(torch.stack([1.0 - sig_m, sig_m], dim=1))

            img_scores = self.scorer(final_anomaly_map)
            return {
                "final_anomaly_map": final_anomaly_map,
                "anomaly_maps_list": layer_up_maps,
                "similarity_map_list": similarity_map_list,
                "img_scores": img_scores,
                "final_anomaly_feat": anomaly_features_list[-1],
                "final_normal_feat": normal_features_list[-1],
                "log_conf": log_conf,
            }


@torch.no_grad()
def evaluate_model(wrapper: BridgeVisualADWrapper, dataloader, device, sigma: float = 4.0, compute_pro: bool = True):
    wrapper.eval()
    gt_sp_list, pr_sp_list = [], []
    masks_list, amaps_list = [], []
    class_masks_dict = {c: [] for c in DEFECT_CLASSES}

    for items in tqdm(dataloader, desc="Evaluating", leave=False):
        image = items["img"].to(device)
        gt_mask = (items["img_mask"] > 0.5).float()
        labels = items["anomaly"]

        with autocast(enabled=(device.type == "cuda")):
            out = wrapper.forward_step(image)
            amap_batch = out["final_anomaly_map"].float().cpu().numpy()
            scores_batch = out["img_scores"].float().cpu().numpy()
            if scores_batch.ndim == 0:
                scores_batch = np.array([float(scores_batch)])

        for b in range(image.shape[0]):
            f_map = gaussian_filter(amap_batch[b], sigma=sigma) if sigma > 0 else amap_batch[b]
            amaps_list.append(f_map)
            masks_list.append(gt_mask[b, 0].cpu().numpy())
            gt_sp_list.append(int(labels[b].item()))
            pr_sp_list.append(float(scores_batch[b]))
            if "class_masks" in items:
                for cname in DEFECT_CLASSES:
                    class_masks_dict[cname].append(items["class_masks"][cname][b, 0].cpu().numpy())

    masks_np = np.stack(masks_list, axis=0)
    amaps_np = np.stack(amaps_list, axis=0)
    gt_sp = np.array(gt_sp_list)
    pr_sp = np.array(pr_sp_list)

    amin, amax = float(amaps_np.min()), float(amaps_np.max())
    if amax > amin + 1e-12:
        amaps_norm = (amaps_np - amin) / (amax - amin)
    else:
        amaps_norm = np.zeros_like(amaps_np)

    flat_norm = amaps_norm.reshape(amaps_norm.shape[0], -1)
    k_top = max(1, int(np.ceil(flat_norm.shape[1] * DEFAULT_TOPK_RATIO)))
    topk_sp = np.sort(flat_norm, axis=1)[:, -k_top:].mean(axis=1)
    if wrapper.variant == "ours":
        smin, smax = float(pr_sp.min()), float(pr_sp.max())
        pr_sp_norm = (pr_sp - smin) / (smax - smin + 1e-12)
        final_sp = 0.5 * topk_sp + 0.5 * pr_sp_norm
    else:
        final_sp = topk_sp

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
        "Best-Th": round(best_th, 4),
        "Per-Class": class_metrics,
    }
    return metrics


def _make_balanced_smoke_subset(items, n_norm=2, n_anom=2):
    norms = [x for x in items if x["anomaly"] == 0][:n_norm]
    anoms = [x for x in items if x["anomaly"] == 1][:n_anom]
    return norms + anoms


def run_single_experiment(args):
    setup_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)
    log_path = os.path.join(args.save_dir, "train_eval.log")

    def log_print(msg):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as lf:
            lf.write(line + "\n")

    log_print(f"=== Starting Experiment: variant={args.variant} | train_ds={args.dataset_name} | device={device} ===")

    clip_model, _ = VisualAD_lib.load(args.backbone, device=device)
    clip_model.to(device)

    wrapper = BridgeVisualADWrapper(
        clip_model=clip_model,
        features_list=args.features_list,
        image_size=args.image_size,
        variant=args.variant,
        device=device,
    ).to(device)

    trainable_params = sum(p.numel() for p in wrapper.parameters() if p.requires_grad)
    log_print(f"Trainable parameters ({args.variant}): {trainable_params:,}")

    dummy_args = argparse.Namespace(image_size=args.image_size)
    preprocess, _ = get_transform(dummy_args)

    train_root = f"/home/skye/data/Skye/databases/{args.dataset_name}"
    train_meta = f"/home/skye/data/Skye/visualad/datasets/{args.dataset_name}_meta.json"
    if not os.path.exists(train_meta):
        build_bridge_meta_json(train_root, train_meta)

    train_ds = BridgeAnomalyDataset(train_root, split="train", image_size=args.image_size,
                                    preprocess=preprocess, return_class_masks=False, meta_json_path=train_meta)
    val_ds = BridgeAnomalyDataset(train_root, split="val", image_size=args.image_size,
                                  preprocess=preprocess, return_class_masks=False, meta_json_path=train_meta)
    test_ds = BridgeAnomalyDataset(train_root, split="test", image_size=args.image_size,
                                   preprocess=preprocess, return_class_masks=True, meta_json_path=train_meta)

    if args.smoke_test:
        train_ds.data_all = _make_balanced_smoke_subset(train_ds.data_all, 2, 2)
        val_ds.data_all = _make_balanced_smoke_subset(val_ds.data_all, 2, 2)
        test_ds.data_all = _make_balanced_smoke_subset(test_ds.data_all, 2, 2)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=(device.type == "cuda")
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device.type == "cuda")
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device.type == "cuda")
    )

    optimizer = wrapper.build_optimizer(args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.1
    )

    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()
    loss_contra = ContrastiveLoss(temperature=0.1, margin=0.5)

    amp_enabled = (device.type == "cuda")
    scaler = GradScaler(enabled=amp_enabled)

    best_val_score = -1.0
    best_ckpt_path = os.path.join(args.save_dir, "best_model.pth")
    history = []

    for epoch in range(args.epochs):
        wrapper.train()
        epoch_losses = []
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)
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

                if args.variant == "baseline":
                    seg_losses = []
                    for sim_m in out["similarity_map_list"]:
                        seg_losses.append(loss_focal(sim_m, gt))
                        seg_losses.append(loss_dice(sim_m[:, 1, :, :], gt))
                    l_seg = sum(seg_losses)
                    total_loss = l_cls + l_seg + l_contra
                else:
                    l_seg = compute_confidence_weighted_seg_loss(
                        out["similarity_map_list"], gt, out["log_conf"], loss_focal, loss_dice,
                        alpha_conf=0.05, beta_edge=0.3
                    )
                    l_div = wrapper.mp_bs.diversity_loss()
                    total_loss = l_cls + l_seg + l_contra + 0.1 * l_div

                loss_scaled = total_loss / args.accum_steps

            if torch.isnan(loss_scaled) or torch.isinf(loss_scaled):
                log_print(f"Warning: NaN/Inf loss at epoch {epoch+1} step {step_idx}, skipping step")
                optimizer.zero_grad(set_to_none=True)
                continue

            if amp_enabled:
                scaler.scale(loss_scaled).backward()
            else:
                loss_scaled.backward()

            if (step_idx + 1) % args.accum_steps == 0 or (step_idx + 1) == len(train_loader):
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

        val_metrics = evaluate_model(wrapper, val_loader, device, sigma=args.sigma, compute_pro=False)
        val_key_score = (
            val_metrics["Image-AUROC"] + val_metrics["Pixel-AUROC"] + val_metrics["Pixel-F1"] + val_metrics["Pixel-AP"]
        ) / 4.0

        log_print(
            f"Epoch [{epoch+1}/{args.epochs}] loss={mean_train_loss:.4f} | "
            f"Val I-AUROC={val_metrics['Image-AUROC']:.2f} P-AUROC={val_metrics['Pixel-AUROC']:.2f} "
            f"P-F1={val_metrics['Pixel-F1']:.2f} P-AP={val_metrics['Pixel-AP']:.2f}"
        )
        history.append({"epoch": epoch + 1, "train_loss": round(mean_train_loss, 4), "val": val_metrics})

        if (epoch + 1) > (args.epochs // 2) and val_key_score >= best_val_score:
            best_val_score = val_key_score
            torch.save(wrapper.get_trainable_state_dict(), best_ckpt_path)

    final_ckpt_path = os.path.join(args.save_dir, "final_model.pth")
    torch.save(wrapper.get_trainable_state_dict(), final_ckpt_path)

    wrapper.load_trainable_state_dict(torch.load(best_ckpt_path, map_location=device), device)
    log_print(f"Loaded best validation checkpoint from {best_ckpt_path} for final Test evaluation...")

    in_domain_test_metrics = evaluate_model(wrapper, test_loader, device, sigma=args.sigma, compute_pro=True)
    log_print(f"[In-Domain Test: {args.dataset_name}] {json.dumps(in_domain_test_metrics)}")

    other_ds_name = (
        "Bridge512_balance_CODEBRIM" if args.dataset_name == "Bridge512_balance_S2DS" else "Bridge512_balance_S2DS"
    )
    other_root = f"/home/skye/data/Skye/databases/{other_ds_name}"
    other_meta = f"/home/skye/data/Skye/visualad/datasets/{other_ds_name}_meta.json"
    if not os.path.exists(other_meta):
        build_bridge_meta_json(other_root, other_meta)
    other_test_ds = BridgeAnomalyDataset(
        other_root, split="test", image_size=args.image_size,
        preprocess=preprocess, return_class_masks=True, meta_json_path=other_meta
    )
    if args.smoke_test:
        other_test_ds.data_all = _make_balanced_smoke_subset(other_test_ds.data_all, 2, 2)

    other_test_loader = torch.utils.data.DataLoader(
        other_test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device.type == "cuda")
    )
    cross_domain_test_metrics = evaluate_model(wrapper, other_test_loader, device, sigma=args.sigma, compute_pro=True)
    log_print(f"[Cross-Domain Zero-Shot Test: {args.dataset_name} -> {other_ds_name}] {json.dumps(cross_domain_test_metrics)}")

    summary = {
        "variant": args.variant,
        "train_dataset": args.dataset_name,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "accum_steps": args.accum_steps,
        "trainable_params": trainable_params,
        "in_domain_test": in_domain_test_metrics,
        "cross_domain_test_dataset": other_ds_name,
        "cross_domain_test": cross_domain_test_metrics,
        "history": history,
    }
    metrics_json_path = os.path.join(args.save_dir, "test_metrics.json")
    with open(metrics_json_path, "w", encoding="utf-8") as mf:
        json.dump(summary, mf, indent=2)
    log_print(f"Saved complete results to {metrics_json_path}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Bridge VisualAD Runner")
    parser.add_argument("--variant", type=str, default="baseline", choices=["baseline", "ours"])
    parser.add_argument("--dataset_name", type=str, default="Bridge512_balance_S2DS",
                        choices=["Bridge512_balance_S2DS", "Bridge512_balance_CODEBRIM"])
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--backbone", type=str, default="ViT-L/14@336px")
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24])
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--accum_steps", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--sigma", type=float, default=4.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()
    run_single_experiment(args)
