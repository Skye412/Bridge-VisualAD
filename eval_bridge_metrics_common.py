import os
import csv
import json
import random
import numpy as np
from PIL import Image
from scipy import ndimage
try:
    from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score
except ImportError:
    def auc(x, y):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        order = np.argsort(x, kind="mergesort")
        return float(np.trapz(y[order], x[order]))

    def precision_recall_curve(y_true, y_score):
        y_true = np.asarray(y_true, dtype=np.int64)
        y_score = np.asarray(y_score, dtype=np.float64)
        desc_idx = np.argsort(y_score, kind="mergesort")[::-1]
        y_score = y_score[desc_idx]
        y_true = y_true[desc_idx]
        distinct_value_indices = np.where(np.diff(y_score))[0]
        threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]
        tps = np.cumsum(y_true)[threshold_idxs].astype(np.float64)
        fps = (1 + threshold_idxs - tps).astype(np.float64)
        precision = tps / np.maximum(tps + fps, 1e-12)
        recall = tps / max(tps[-1], 1.0)
        sl = slice(None, None, -1)
        return np.r_[precision[sl], 1.0], np.r_[recall[sl], 0.0], y_score[threshold_idxs][sl]

    def average_precision_score(y_true, y_score):
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        return float(-np.sum(np.diff(recall) * precision[:-1]))

    def roc_auc_score(y_true, y_score):
        y_true = np.asarray(y_true, dtype=np.int64)
        y_score = np.asarray(y_score, dtype=np.float64)
        desc_idx = np.argsort(y_score, kind="mergesort")[::-1]
        y_score = y_score[desc_idx]
        y_true = y_true[desc_idx]
        distinct_value_indices = np.where(np.diff(y_score))[0]
        threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]
        tps = np.cumsum(y_true)[threshold_idxs].astype(np.float64)
        fps = (1 + threshold_idxs - tps).astype(np.float64)
        tps = np.r_[0.0, tps]
        fps = np.r_[0.0, fps]
        tpr = tps / max(tps[-1], 1.0)
        fpr = fps / max(fps[-1], 1.0)
        return float(np.trapz(tpr, fpr))
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

DEFECT_CLASSES = ["Crack", "Spalling", "Corrosion", "Efflorescence"]
DEFECT_RGB_MAP = {
    "Crack": (255, 0, 0),
    "Spalling": (0, 255, 0),
    "Corrosion": (0, 0, 255),
    "Efflorescence": (255, 255, 0),
}


class BridgeAnomalyDataset(data.Dataset):
    """
    Unified dataset loader compatible across Python 3.8 - 3.11 and all PyTorch/Torchvision versions.
    """
    def __init__(self, root, split="train", image_size=518,
                 preprocess=None, return_class_masks=False, meta_json_path=None):
        super(BridgeAnomalyDataset, self).__init__()
        self.root = root
        self.split = split
        self.image_size = image_size
        self.preprocess = preprocess
        self.return_class_masks = return_class_masks
        self.obj_list = ["bridge"]
        self.class_name_map_class_id = {"bridge": 0}

        if meta_json_path and os.path.exists(meta_json_path):
            with open(meta_json_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        else:
            raise FileNotFoundError("meta_json_path not found: {}".format(meta_json_path))

        self.data_all = meta[split]["bridge"]

    def __len__(self):
        return len(self.data_all)

    def __getitem__(self, index):
        item = self.data_all[index]
        img_full_path = os.path.join(self.root, item["img_path"])
        img = Image.open(img_full_path).convert("RGB")
        w, h = img.size

        anomaly = int(item["anomaly"])
        if anomaly == 0 or not item["mask_path"]:
            bin_mask_np = np.zeros((h, w), dtype=np.uint8)
            rgb_mask_np = np.zeros((h, w, 3), dtype=np.uint8) if self.return_class_masks else None
        else:
            mask_full_path = os.path.join(self.root, item["mask_path"])
            rgb_mask_np = np.array(Image.open(mask_full_path).convert("RGB"), dtype=np.uint8)
            bin_mask_np = ((rgb_mask_np > 0).any(axis=-1)).astype(np.uint8) * 255

        if self.preprocess is not None:
            img_tensor = self.preprocess(img)
        else:
            img_tensor = transforms.ToTensor()(img)

        mask_pil = Image.fromarray(bin_mask_np, mode="L").resize((self.image_size, self.image_size), resample=Image.NEAREST)
        mask_tensor = (transforms.ToTensor()(mask_pil) > 0.5).float()  # [1, H, W]

        out = {
            "img": img_tensor,
            "img_mask": mask_tensor,
            "cls_name": "bridge",
            "anomaly": anomaly,
            "img_path": img_full_path,
            "cls_id": 0,
            "specie_name": item.get("specie_name", ""),
        }

        if self.return_class_masks:
            class_masks = {}
            for cname, rgb in DEFECT_RGB_MAP.items():
                if anomaly == 0:
                    cm = np.zeros((h, w), dtype=np.uint8)
                else:
                    cm = (
                        (rgb_mask_np[:, :, 0] == rgb[0]) &
                        (rgb_mask_np[:, :, 1] == rgb[1]) &
                        (rgb_mask_np[:, :, 2] == rgb[2])
                    ).astype(np.uint8) * 255
                cm_pil = Image.fromarray(cm, mode="L").resize((self.image_size, self.image_size), resample=Image.NEAREST)
                class_masks[cname] = (transforms.ToTensor()(cm_pil) > 0.5).float()
            out["class_masks"] = class_masks

        return out


def setup_seed_common(seed=111):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def norm01_np(arr):
    arr = arr.astype(np.float32, copy=False)
    mn, mx = float(arr.min()), float(arr.max())
    if mx <= mn + 1e-12:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - mn) / (mx - mn)


def norm01_f32_list(lst):
    arr = np.stack(lst, axis=0).astype(np.float32)
    lst.clear()
    mn, mx = float(arr.min()), float(arr.max())
    if mx <= mn + 1e-12:
        return np.zeros_like(arr, dtype=np.float32)
    arr -= mn
    arr /= (mx - mn)
    return arr


def topk_pool_np(amaps_np, ratio=0.01):
    flat = amaps_np.reshape(amaps_np.shape[0], -1)
    k = max(1, int(np.ceil(flat.shape[1] * ratio)))
    return np.sort(flat, axis=1)[:, -k:].mean(axis=1)


def fast_aupro(masks, amaps, num_th=40, max_fpr=0.3):
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
    fpr_span = float(fprs[-1] - fprs[0])
    if fpr_span <= 1e-12:
        return 0.0
    return float(auc(fprs, pros) / fpr_span)


def eval_metrics_from_amap_and_sp(gt_sp, final_sp, masks_np, amaps_norm, class_masks_dict, compute_pro=True):
    if len(np.unique(gt_sp)) > 1:
        i_auroc = float(roc_auc_score(gt_sp, final_sp)) * 100.0
        i_ap = float(average_precision_score(gt_sp, final_sp)) * 100.0
        prec_s, rec_s, _ = precision_recall_curve(gt_sp, final_sp)
        i_f1 = float(np.max(2 * prec_s * rec_s / (prec_s + rec_s + 1e-8))) * 100.0
    else:
        i_auroc, i_ap, i_f1 = 50.0, 50.0, 50.0

    gt_px = masks_np.reshape(-1).astype(np.uint8)
    pr_px = amaps_norm.reshape(-1).astype(np.float32)

    pos_vals = pr_px[gt_px == 1]
    neg_vals = pr_px[gt_px == 0]
    total_pos = max(float(len(pos_vals)), 1.0)
    total_neg = max(float(len(neg_vals)), 1.0)

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
    if class_masks_dict and class_masks_dict.get(DEFECT_CLASSES[0]):
        pred_2d = (amaps_norm >= best_th)
        for cname in DEFECT_CLASSES:
            cm_np = np.stack(class_masks_dict[cname], axis=0) > 0.5
            c_total = float(cm_np.sum())
            if c_total > 0:
                c_tp = float((pred_2d & cm_np).sum())
                class_metrics[f"{cname}_Recall"] = round((c_tp / c_total) * 100.0, 2)

    mean_score = (i_auroc + i_f1 + i_ap + p_auroc + p_f1 + p_ap + p_aupro) / 7.0
    mean_5 = (i_auroc + p_auroc + p_f1 + p_ap + p_aupro) / 5.0
    return {
        "Image-AUROC": round(i_auroc, 2),
        "Image-F1": round(i_f1, 2),
        "Image-AP": round(i_ap, 2),
        "Pixel-AUROC": round(p_auroc, 2),
        "Pixel-F1": round(p_f1, 2),
        "Pixel-AP": round(p_ap, 2),
        "Pixel-AUPRO": round(p_aupro, 2),
        "Pixel-IoU": round(p_iou, 2),
        "Mean-Score": round(mean_score, 2),
        "Mean-5": round(mean_5, 2),
        "Best-Th": round(best_th, 4),
        "Per-Class": class_metrics,
    }


def norm01(arr: np.ndarray) -> np.ndarray:
    mn, mx = float(arr.min()), float(arr.max())
    if mx <= mn + 1e-12:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


def topk_pool_np(amap: np.ndarray, ratio: float) -> np.ndarray:
    flat = amap.reshape(amap.shape[0], -1)
    k = max(1, int(np.ceil(flat.shape[1] * ratio)))
    return norm01(np.sort(flat, axis=1)[:, -k:].mean(axis=1))
