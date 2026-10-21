import argparse
import json
import os
import sys
import numpy as np
from scipy.ndimage import gaussian_filter
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.join(CUR_DIR, "VisualAD_repo")
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
if CUR_DIR not in sys.path:
    sys.path.insert(0, CUR_DIR)

import VisualAD_lib
from utils.transforms import get_transform
from bridge_dataset import BridgeAnomalyDataset, DEFECT_CLASSES
from run_experiments import BridgeVisualADWrapper
from run_true_ablation_train import StandaloneAblationWrapper, norm01
from run_a4_scratch_8ep import Scratch8EpA4Wrapper
from run_ablation_and_figures import (
    load_rgb_resized,
    make_gt_color_vis,
    make_heatmap_overlay,
    make_pred_error_overlay,
    plot_ablation_quantitative_charts,
    select_best_showcase_indices,
)
from generate_all_verified_tables import build_all_verified_tables


@torch.no_grad()
def collect_true_standalone_maps(dataset_name: str, short_ds: str, dataloader, device):
    """
    Loads the 5 independently trained 8-epoch checkpoints:
      - A0: baseline_{short_ds}/best_model.pth
      - A1: a1_mp_bs_{short_ds}/best_model.pth
      - A2: a2_wd_saf_{short_ds}/best_model.pth
      - A3: a3_agds_wgd_{short_ds}/best_model.pth
      - A4: a4_scratch8ep_{short_ds}/best_model.pth
    and computes their test-set anomaly maps for visual progression plotting.
    """
    features_list = [6, 12, 18, 24]
    clip_model, _ = VisualAD_lib.load("ViT-L/14@336px", device=device)
    clip_model.to(device)

    out_root = os.path.join(CUR_DIR, "outputs")

    # 1. Load A0 (Baseline)
    w_a0 = BridgeVisualADWrapper(clip_model, features_list, 518, "baseline", device).to(device)
    w_a0.load_trainable_state_dict(torch.load(os.path.join(out_root, f"baseline_{short_ds}", "best_model.pth"), map_location=device), device)
    w_a0.eval()

    # 2. Load A1 (Standalone +MP-BS)
    w_a1 = StandaloneAblationWrapper(clip_model, features_list, 518, "a1_mp_bs", device).to(device)
    w_a1.load_trainable_state_dict(torch.load(os.path.join(out_root, f"a1_mp_bs_{short_ds}", "best_model.pth"), map_location=device), device)
    w_a1.eval()

    # 3. Load A2 (Standalone +MP-BS +WD-SAF)
    w_a2 = StandaloneAblationWrapper(clip_model, features_list, 518, "a2_wd_saf", device).to(device)
    w_a2.load_trainable_state_dict(torch.load(os.path.join(out_root, f"a2_wd_saf_{short_ds}", "best_model.pth"), map_location=device), device)
    w_a2.eval()

    # 4. Load A3 (Standalone +MP-BS +WD-SAF +AGDS-WGD)
    w_a3 = StandaloneAblationWrapper(clip_model, features_list, 518, "a3_agds_wgd", device).to(device)
    w_a3.load_trainable_state_dict(torch.load(os.path.join(out_root, f"a3_agds_wgd_{short_ds}", "best_model.pth"), map_location=device), device)
    w_a3.eval()

    # 5. Load A4 (Standalone Scratch 8Ep Full Ours)
    w_a4 = Scratch8EpA4Wrapper(clip_model, features_list, 518, device).to(device)
    w_a4.load_trainable_state_dict(torch.load(os.path.join(out_root, f"a4_scratch8ep_{short_ds}", "best_model.pth"), map_location=device), device)
    w_a4.eval()

    a0_list = []
    a1_pre_list, a1_post_list = [], []
    a2_pre_list, a2_post_list = [], []
    a3_pre_list, a3_post_list, a3_prob_list = [], [], []
    a4_pre_list, a4_post_list, a4_prob_list, a4_conf_list = [], [], [], []
    masks_list, img_paths = [], []
    class_masks_dict = {c: [] for c in DEFECT_CLASSES}

    use_cuda = (device.type == "cuda")
    for items in dataloader:
        image = items["img"].to(device)
        gt_mask = (items["img_mask"] > 0.5).float()
        paths = items["img_path"]

        with torch.amp.autocast("cuda", enabled=use_cuda):
            out_a0 = w_a0.forward_step(image)
            out_a1 = w_a1.forward_step(image)
            out_a2 = w_a2.forward_step(image)
            out_a3 = w_a3.forward_step(image)
            out_a4 = w_a4.forward_step(image)

        for b in range(image.shape[0]):
            a0_list.append(gaussian_filter(out_a0["final_anomaly_map"][b].float().cpu().numpy(), sigma=4.0))
            a1_pre_list.append(gaussian_filter(out_a1["pre_sum"][b].float().cpu().numpy(), sigma=4.0))
            a1_post_list.append(gaussian_filter(out_a1["post_sum"][b].float().cpu().numpy(), sigma=4.0))
            a2_pre_list.append(gaussian_filter(out_a2["pre_sum"][b].float().cpu().numpy(), sigma=4.0))
            a2_post_list.append(gaussian_filter(out_a2["post_sum"][b].float().cpu().numpy(), sigma=4.0))
            a3_pre_list.append(gaussian_filter(out_a3["pre_sum"][b].float().cpu().numpy(), sigma=4.0))
            a3_post_list.append(gaussian_filter(out_a3["post_sum"][b].float().cpu().numpy(), sigma=4.0))
            a3_prob_list.append(gaussian_filter(out_a3["prob_map"][b].float().cpu().numpy(), sigma=1.5))

            a4_pre_list.append(gaussian_filter(out_a4["pre_sum"][b].float().cpu().numpy(), sigma=4.0))
            a4_post_list.append(gaussian_filter(out_a4["post_sum"][b].float().cpu().numpy(), sigma=4.0))
            a4_prob_list.append(gaussian_filter(out_a4["prob_map"][b].float().cpu().numpy(), sigma=1.5))
            a4_conf_list.append(gaussian_filter(torch.sigmoid(out_a4["log_conf"][b]).float().cpu().numpy(), sigma=2.0))

            masks_list.append(gt_mask[b, 0].cpu().numpy())
            img_paths.append(paths[b])
            for cname in DEFECT_CLASSES:
                class_masks_dict[cname].append(items["class_masks"][cname][b, 0].cpu().numpy())

    amap_A0 = norm01(np.stack(a0_list, axis=0))
    amap_A1 = norm01(0.65 * norm01(np.stack(a1_pre_list, axis=0)) + 0.35 * norm01(np.stack(a1_post_list, axis=0)))
    amap_A2 = norm01(0.65 * norm01(np.stack(a2_pre_list, axis=0)) + 0.35 * norm01(np.stack(a2_post_list, axis=0)))

    cos_A3 = norm01(0.65 * norm01(np.stack(a3_pre_list, axis=0)) + 0.35 * norm01(np.stack(a3_post_list, axis=0)))
    prob_A3 = norm01(np.stack(a3_prob_list, axis=0))
    amap_A3 = norm01(0.82 * cos_A3 + 0.18 * prob_A3)

    cos_A4 = norm01(0.65 * norm01(np.stack(a4_pre_list, axis=0)) + 0.35 * norm01(np.stack(a4_post_list, axis=0)))
    prob_A4 = norm01(np.stack(a4_prob_list, axis=0))
    conf_A4 = norm01(np.stack(a4_conf_list, axis=0))
    prob_conf_A4 = norm01(prob_A4 * (0.90 + 0.10 * conf_A4))
    if short_ds == "S2DS":
        amap_A4 = norm01(cos_A4 * (0.72 + 0.28 * prob_conf_A4) + 0.18 * prob_conf_A4)
    else:
        amap_A4 = norm01(cos_A4 * (0.80 + 0.05 * prob_conf_A4) + 0.18 * prob_conf_A4)

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
        "True Standalone 8-Epoch Ablation Visual Progression (Independently Trained Models A0 -> A4)",
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

    fig_dir = os.path.join(CUR_DIR, "figures_true_ablation")
    os.makedirs(fig_dir, exist_ok=True)

    # Plot quantitative charts from verified standalone 8-epoch models
    plot_ablation_quantitative_charts(
        comb_metrics["S2DS_in_domain"],
        comb_metrics["CODEBRIM_in_domain"],
        os.path.join(fig_dir, "fig4_true_ablation_quantitative_charts.png"),
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
    s2ds_loader = torch.utils.data.DataLoader(s2ds_test, batch_size=4, shuffle=False, num_workers=4)
    cb_loader = torch.utils.data.DataLoader(cb_test, batch_size=4, shuffle=False, num_workers=4)

    pack_s2ds = collect_true_standalone_maps("Bridge512_balance_S2DS", "S2DS", s2ds_loader, device)
    pack_cb = collect_true_standalone_maps("Bridge512_balance_CODEBRIM", "CODEBRIM", cb_loader, device)

    plot_true_ablation_visual_progression(
        pack_s2ds, pack_cb, comb_metrics,
        os.path.join(fig_dir, "fig3_true_ablation_visual_progression.png")
    )
    plot_true_ablation_visual_progression(
        pack_s2ds, pack_cb, comb_metrics,
        os.path.join(CUR_DIR, "figures", "fig3_ablation_visual_progression.png")
    )


if __name__ == "__main__":
    main()
