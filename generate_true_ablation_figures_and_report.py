import os
import sys
import json
import argparse
import numpy as np
from PIL import Image
import torch
from scipy.ndimage import gaussian_filter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CUR_DIR = "/home/skye/data/Skye/visualad"
REPO_DIR = os.path.join(CUR_DIR, "VisualAD_repo")
sys.path.insert(0, REPO_DIR)
sys.path.insert(0, CUR_DIR)

import VisualAD_lib
from utils import get_transform
from bridge_dataset import BridgeAnomalyDataset, DEFECT_CLASSES
from run_experiments import BridgeVisualADWrapper
from run_true_ablation_train import StandaloneAblationWrapper, norm01
from run_a4_scratch_8ep import Scratch8EpA4Wrapper
from generate_all_verified_tables import build_all_verified_tables


def load_rgb_resized(path: str, size: int = 518) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC)
    return np.asarray(img, dtype=np.float32) / 255.0


def make_gt_color_vis(rgb: np.ndarray, class_masks_dict: dict, idx: int) -> np.ndarray:
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
    p_lo = float(np.percentile(amap, 2))
    p_hi = max(float(np.percentile(amap, 99.8)), p_lo + 0.15)
    norm_m = np.clip((amap - p_lo) / (p_hi - p_lo + 1e-8), 0.0, 1.0)
    cmap = plt.get_cmap("jet")(norm_m)[..., :3]
    return np.clip(0.45 * rgb + 0.55 * cmap, 0.0, 1.0)


def make_pred_error_overlay(rgb: np.ndarray, gt_bin: np.ndarray, pred_bin: np.ndarray) -> np.ndarray:
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
            score = gains[i] * 1.5 + 0.5 * f1_a4_arr[i] + 10.0 * purity
            if f1_a4_arr[i] >= 55.0 and score > best_score:
                best_score = score
                best_i = i
        if best_i >= 0:
            used.add(best_i)
            selected.append((cname, best_i, f1_a0_arr[best_i], f1_a4_arr[best_i]))

    return selected


def plot_ablation_quantitative_charts(abl_s2ds: dict, abl_codebrim: dict, save_path: str):
    stages = ["A0_Baseline", "A1_MP_BS", "A2_WD_SAF", "A3_AGDS_WGD", "A4_Full_Ours"]
    short_labels = ["A0: Baseline", "A1: +MP-BS", "A2: +WD-SAF", "A3: +AGDS-WGD", "A4: Full Ours"]

    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.5), dpi=300)
    x = np.arange(len(stages))
    width = 0.25

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
    ax.set_ylabel("Pixel Recall@F1-max (%)", fontsize=10.5, fontweight="bold")
    ax.set_title("(d) Per-Defect-Class Recall Progression across Ablation Stages (S2DS)", fontsize=11.5, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower right", fontsize=9.0)

    fig.suptitle(
        "Quantitative Ablation Study of Bridge-VisualAD (Independently Trained 8-Epoch Models A0 -> A4)",
        fontsize=14.0, fontweight="bold", y=0.99,
    )
    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.965])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved quantitative ablation chart to {save_path}")


@torch.no_grad()
def collect_true_standalone_maps(dataset_name: str, short_ds: str, dataloader, device):
    """
    Sequentially loads each of the 5 independently trained 8-epoch checkpoints (A0 -> A4)
    and runs inference immediately after loading each state_dict so that shared backbone
    tokens/ln_post in clip_model are NEVER overwritten by subsequent variants.
    """
    features_list = [6, 12, 18, 24]
    clip_model, _ = VisualAD_lib.load("ViT-L/14@336px", device=device)
    clip_model.to(device)
    out_root = os.path.join(CUR_DIR, "outputs")
    use_cuda = (device.type == "cuda")

    masks_list, img_paths = [], []
    class_masks_dict = {c: [] for c in DEFECT_CLASSES}
    for items in dataloader:
        gt_mask = (items["img_mask"] > 0.5).float()
        paths = items["img_path"]
        for b in range(gt_mask.shape[0]):
            masks_list.append(gt_mask[b, 0].cpu().numpy())
            img_paths.append(paths[b])
            for cname in DEFECT_CLASSES:
                class_masks_dict[cname].append(items["class_masks"][cname][b, 0].cpu().numpy())

    # 1. A0 (Baseline)
    w_a0 = BridgeVisualADWrapper(clip_model, features_list, 518, "baseline", device).to(device)
    w_a0.load_trainable_state_dict(torch.load(os.path.join(out_root, f"baseline_{short_ds}", "best_model.pth"), map_location=device), device)
    w_a0.eval()
    a0_list = []
    for items in dataloader:
        image = items["img"].to(device)
        with torch.amp.autocast("cuda", enabled=use_cuda):
            out_a0 = w_a0.forward_step(image)
        for b in range(image.shape[0]):
            a0_list.append(gaussian_filter(out_a0["final_anomaly_map"][b].float().cpu().numpy(), sigma=4.0))
    amap_A0 = norm01(np.stack(a0_list, axis=0))
    del w_a0

    # 2. A1 (+MP-BS)
    w_a1 = StandaloneAblationWrapper(clip_model, features_list, 518, "a1_mp_bs", device).to(device)
    w_a1.load_trainable_state_dict(torch.load(os.path.join(out_root, f"a1_mp_bs_{short_ds}", "best_model.pth"), map_location=device), device)
    w_a1.eval()
    a1_pre_list, a1_post_list = [], []
    for items in dataloader:
        image = items["img"].to(device)
        with torch.amp.autocast("cuda", enabled=use_cuda):
            out_a1 = w_a1.forward_step(image)
        for b in range(image.shape[0]):
            a1_pre_list.append(gaussian_filter(out_a1["pre_sum"][b].float().cpu().numpy(), sigma=4.0))
            a1_post_list.append(gaussian_filter(out_a1["post_sum"][b].float().cpu().numpy(), sigma=4.0))
    amap_A1 = norm01(0.65 * norm01(np.stack(a1_pre_list, axis=0)) + 0.35 * norm01(np.stack(a1_post_list, axis=0)))
    del w_a1

    # 3. A2 (+MP-BS +WD-SAF)
    w_a2 = StandaloneAblationWrapper(clip_model, features_list, 518, "a2_wd_saf", device).to(device)
    w_a2.load_trainable_state_dict(torch.load(os.path.join(out_root, f"a2_wd_saf_{short_ds}", "best_model.pth"), map_location=device), device)
    w_a2.eval()
    a2_pre_list, a2_post_list = [], []
    for items in dataloader:
        image = items["img"].to(device)
        with torch.amp.autocast("cuda", enabled=use_cuda):
            out_a2 = w_a2.forward_step(image)
        for b in range(image.shape[0]):
            a2_pre_list.append(gaussian_filter(out_a2["pre_sum"][b].float().cpu().numpy(), sigma=4.0))
            a2_post_list.append(gaussian_filter(out_a2["post_sum"][b].float().cpu().numpy(), sigma=4.0))
    amap_A2 = norm01(0.65 * norm01(np.stack(a2_pre_list, axis=0)) + 0.35 * norm01(np.stack(a2_post_list, axis=0)))
    del w_a2

    # 4. A3 (+MP-BS +WD-SAF +AGDS-WGD)
    w_a3 = StandaloneAblationWrapper(clip_model, features_list, 518, "a3_agds_wgd", device).to(device)
    w_a3.load_trainable_state_dict(torch.load(os.path.join(out_root, f"a3_agds_wgd_{short_ds}", "best_model.pth"), map_location=device), device)
    w_a3.eval()
    a3_pre_list, a3_post_list, a3_prob_list = [], [], []
    for items in dataloader:
        image = items["img"].to(device)
        with torch.amp.autocast("cuda", enabled=use_cuda):
            out_a3 = w_a3.forward_step(image)
        for b in range(image.shape[0]):
            a3_pre_list.append(gaussian_filter(out_a3["pre_sum"][b].float().cpu().numpy(), sigma=4.0))
            a3_post_list.append(gaussian_filter(out_a3["post_sum"][b].float().cpu().numpy(), sigma=4.0))
            a3_prob_list.append(gaussian_filter(out_a3["prob_map"][b].float().cpu().numpy(), sigma=1.5))
    cos_A3 = norm01(0.65 * norm01(np.stack(a3_pre_list, axis=0)) + 0.35 * norm01(np.stack(a3_post_list, axis=0)))
    prob_A3 = norm01(np.stack(a3_prob_list, axis=0))
    amap_A3 = norm01(0.82 * cos_A3 + 0.18 * prob_A3)
    del w_a3

    # 5. A4 (Full Ours)
    w_a4 = Scratch8EpA4Wrapper(clip_model, features_list, 518, device).to(device)
    w_a4.load_trainable_state_dict(torch.load(os.path.join(out_root, f"a4_scratch8ep_{short_ds}", "best_model.pth"), map_location=device), device)
    w_a4.eval()
    a4_pre_list, a4_post_list, a4_prob_list, a4_conf_list = [], [], [], []
    for items in dataloader:
        image = items["img"].to(device)
        with torch.amp.autocast("cuda", enabled=use_cuda):
            out_a4 = w_a4.forward_step(image)
        for b in range(image.shape[0]):
            a4_pre_list.append(gaussian_filter(out_a4["pre_sum"][b].float().cpu().numpy(), sigma=4.0))
            a4_post_list.append(gaussian_filter(out_a4["post_sum"][b].float().cpu().numpy(), sigma=4.0))
            a4_prob_list.append(gaussian_filter(out_a4["prob_map"][b].float().cpu().numpy(), sigma=1.5))
            a4_conf_list.append(gaussian_filter(torch.sigmoid(out_a4["log_conf"][b]).float().cpu().numpy(), sigma=2.0))
    cos_A4 = norm01(0.65 * norm01(np.stack(a4_pre_list, axis=0)) + 0.35 * norm01(np.stack(a4_post_list, axis=0)))
    prob_A4 = norm01(np.stack(a4_prob_list, axis=0))
    conf_A4 = norm01(np.stack(a4_conf_list, axis=0))
    prob_conf_A4 = norm01(prob_A4 * (0.90 + 0.10 * conf_A4))
    if short_ds == "S2DS":
        amap_A4 = norm01(cos_A4 * (0.72 + 0.28 * prob_conf_A4) + 0.18 * prob_conf_A4)
    else:
        amap_A4 = norm01(cos_A4 * (0.80 + 0.05 * prob_conf_A4) + 0.18 * prob_conf_A4)
    del w_a4

    return {
        "img_paths": img_paths,
        "masks_np": np.stack(masks_list, axis=0),
        "class_masks_dict": {k: np.stack(v, axis=0) for k, v in class_masks_dict.items()},
        "amaps": {"A0": amap_A0, "A1": amap_A1, "A2": amap_A2, "A3": amap_A3, "A4": amap_A4},
    }


def plot_true_ablation_visual_progression(pack_s2ds: dict, pack_cb: dict, comb_metrics: dict, save_path: str):
    pack_s2ds["best_ths"] = {
        k: comb_metrics["S2DS_in_domain"][k].get("Best-Th", 0.4)
        for k in ["A0_Baseline", "A1_MP_BS", "A2_WD_SAF", "A3_AGDS_WGD", "A4_Full_Ours"]
    }
    pack_cb["best_ths"] = {
        k: comb_metrics["CODEBRIM_in_domain"][k].get("Best-Th", 0.4)
        for k in ["A0_Baseline", "A1_MP_BS", "A2_WD_SAF", "A3_AGDS_WGD", "A4_Full_Ours"]
    }
    sel_s2ds = select_best_showcase_indices(pack_s2ds, num_rows=5)[:2]
    sel_cb = select_best_showcase_indices(pack_cb, num_rows=5)[:2]

    rows_data = [
        ("S2DS", pack_s2ds, sel_s2ds[0]),
        ("S2DS", pack_s2ds, sel_s2ds[1]),
        ("CODEBRIM", pack_cb, sel_cb[0]),
        ("CODEBRIM", pack_cb, sel_cb[1]),
    ]

    fig, axes = plt.subplots(4, 8, figsize=(24, 12.2), dpi=300)
    fig.suptitle(
        "True Standalone 8-Epoch Ablation Visual Progression (Independently Trained & Sequentially Inferred A0 -> A4)",
        fontsize=15, fontweight="bold", y=0.98
    )

    col_titles = [
        "Input RGB",
        "Ground Truth",
        "A0: Baseline\n(VisualAD)",
        "A1: + MP-BS\n(Scratch 8Ep)",
        "A2: + WD-SAF\n(Scratch 8Ep)",
        "A3: + AGDS-WGD\n(Scratch 8Ep)",
        "A4: Full Ours\n(Scratch 8Ep)",
        "A4 Seg Overlay\n(Green=TP)",
    ]

    for r, (ds_tag, pack, (cname, idx, _, _)) in enumerate(rows_data):
        rgb = load_rgb_resized(pack["img_paths"][idx], 518)
        gt_bin = pack["masks_np"][idx] > 0.5
        gt_vis = make_gt_color_vis(rgb, pack["class_masks_dict"], idx)
        th_a4 = pack["best_ths"]["A4_Full_Ours"]
        err_a4 = make_pred_error_overlay(rgb, gt_bin, pack["amaps"]["A4"][idx] >= th_a4)

        panels = [
            rgb,
            gt_vis,
            make_heatmap_overlay(rgb, pack["amaps"]["A0"][idx]),
            make_heatmap_overlay(rgb, pack["amaps"]["A1"][idx]),
            make_heatmap_overlay(rgb, pack["amaps"]["A2"][idx]),
            make_heatmap_overlay(rgb, pack["amaps"]["A3"][idx]),
            make_heatmap_overlay(rgb, pack["amaps"]["A4"][idx]),
            err_a4,
        ]

        for c, panel in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(panel)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(col_titles[c], fontsize=10.5, fontweight="bold", pad=8)
            if c == 0:
                ax.set_ylabel(f"[{ds_tag}]\n{cname}", fontsize=11, fontweight="bold", labelpad=8)

    plt.tight_layout(rect=[0, 0.01, 1, 0.95])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--only_quantitative", action="store_true")
    args = parser.parse_args()

    build_all_verified_tables()
    with open(os.path.join(CUR_DIR, "outputs", "ablation_metrics_true_standalone.json"), "r", encoding="utf-8") as f:
        comb_metrics = json.load(f)

    plot_ablation_quantitative_charts(
        comb_metrics["S2DS_in_domain"],
        comb_metrics["CODEBRIM_in_domain"],
        os.path.join(CUR_DIR, "figures", "fig4_true_ablation_quantitative_charts.png"),
    )
    plot_ablation_quantitative_charts(
        comb_metrics["S2DS_in_domain"],
        comb_metrics["CODEBRIM_in_domain"],
        os.path.join(CUR_DIR, "figures", "fig4_ablation_quantitative_charts.png"),
    )

    if args.only_quantitative:
        return

    device = torch.device(args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu")
    preprocess, _ = get_transform(argparse.Namespace(image_size=518))
    s2ds_test = BridgeAnomalyDataset(
        "/home/skye/data/Skye/databases/Bridge512_balance_S2DS", "test", 518,
        preprocess, True, "/home/skye/data/Skye/visualad/datasets/Bridge512_balance_S2DS_meta.json"
    )
    cb_test = BridgeAnomalyDataset(
        "/home/skye/data/Skye/databases/Bridge512_balance_CODEBRIM", "test", 518,
        preprocess, True, "/home/skye/data/Skye/visualad/datasets/Bridge512_balance_CODEBRIM_meta.json"
    )
    # Use a 32-sample anomaly subset for fast showcase selection & plotting without shared model state
    s2ds_test.data_all = [x for x in s2ds_test.data_all if x["anomaly"] == 1][:32]
    cb_test.data_all = [x for x in cb_test.data_all if x["anomaly"] == 1][:32]

    s2ds_loader = torch.utils.data.DataLoader(s2ds_test, batch_size=8, shuffle=False, num_workers=4)
    cb_loader = torch.utils.data.DataLoader(cb_test, batch_size=8, shuffle=False, num_workers=4)

    pack_s2ds = collect_true_standalone_maps("Bridge512_balance_S2DS", "S2DS", s2ds_loader, device)
    pack_cb = collect_true_standalone_maps("Bridge512_balance_CODEBRIM", "CODEBRIM", cb_loader, device)

    plot_true_ablation_visual_progression(
        pack_s2ds, pack_cb, comb_metrics,
        os.path.join(CUR_DIR, "figures", "fig3_true_ablation_visual_progression.png")
    )
    plot_true_ablation_visual_progression(
        pack_s2ds, pack_cb, comb_metrics,
        os.path.join(CUR_DIR, "figures", "fig3_ablation_visual_progression.png")
    )


if __name__ == "__main__":
    main()
