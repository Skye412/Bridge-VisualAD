import os
import sys
import gc
import json
import math
import time
import shutil
import argparse
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

BAYESPFL_DIR = "/home/skye/data/Skye/Bayes-PFL-main"
VISUALAD_DIR = "/home/skye/data/Skye/visualad"
os.chdir(BAYESPFL_DIR)
if BAYESPFL_DIR not in sys.path:
    sys.path.insert(0, BAYESPFL_DIR)
if VISUALAD_DIR not in sys.path:
    sys.path.append(VISUALAD_DIR)

from loss import FocalLoss, DiceLoss, Orthogonal_Loss
from models.VPB import Context_Prompting, TextEncoder
from models.model_CLIP import Load_CLIP, tokenize
from eval_bridge_metrics_common import (
    BridgeAnomalyDataset,
    DEFECT_CLASSES,
    eval_metrics_from_amap_and_sp,
    norm01_f32_list,
    setup_seed_common,
)

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def get_bayespfl_transform(n_px=518):
    return Compose([
        Resize((n_px, n_px), interpolation=BICUBIC),
        CenterCrop((n_px, n_px)),
        lambda img: img.convert("RGB"),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


@torch.no_grad()
def extract_bayespfl_cache(model_clip, dataset, device, features_list, cache_dir, prefix, batch_size=16):
    os.makedirs(cache_dir, exist_ok=True)
    n_samples = len(dataset)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    img_mmap_path = os.path.join(cache_dir, f"{prefix}_img_feats.mmap")
    patch_mmap_path = os.path.join(cache_dir, f"{prefix}_patch_feats.mmap")

    img_mmap = None
    patch_mmap = None
    gts_list, labels_list, cls_names_list = [], [], []
    cursor = 0

    model_clip.eval()
    for items in tqdm(loader, desc=f"Pre-extract Bayes-PFL ({prefix})", leave=False):
        image = items["img"].to(device, non_blocking=True)
        bsz = image.shape[0]
        with torch.cuda.amp.autocast():
            image_features, _, patch_tokens = model_clip.encode_image(image, features_list)
            pf_stack = torch.stack(patch_tokens, dim=1)  # [B, 4, L, C]

        if img_mmap is None:
            d_img = image_features.shape[-1]
            n_layers, seq_l, d_p = pf_stack.shape[1], pf_stack.shape[2], pf_stack.shape[3]
            img_mmap = np.memmap(img_mmap_path, dtype=np.float16, mode="w+", shape=(n_samples, d_img))
            patch_mmap = np.memmap(patch_mmap_path, dtype=np.float16, mode="w+", shape=(n_samples, n_layers, seq_l, d_p))

        img_mmap[cursor:cursor + bsz] = image_features.half().cpu().numpy()
        patch_mmap[cursor:cursor + bsz] = pf_stack.half().cpu().numpy()
        gts_list.append((items["img_mask"].squeeze(1).numpy() > 0.5).astype(np.uint8))
        labels_list.append(items["anomaly"].numpy().astype(np.int64))
        cls_names_list.extend(list(items["cls_name"]))
        cursor += bsz

    img_mmap.flush()
    patch_mmap.flush()
    gts_arr = np.concatenate(gts_list, axis=0)
    labels_arr = np.concatenate(labels_list, axis=0)
    return img_mmap, patch_mmap, gts_arr, labels_arr, cls_names_list


@torch.no_grad()
def evaluate_bayespfl_cached(MyModel, PFL_TextEncoder, img_mmap, patch_mmap, gts_arr, labels_arr, cls_names_list, device, args, stage=1, sigma=8.0):
    MyModel.eval()
    MyModel.zk_n = None
    MyModel.zk_a = None

    n_samples = len(labels_arr)
    pr_sp_list, amaps_list = [], []
    R_tot = args.prompt_num * args.sample_num

    for idx in range(n_samples):
        img_feat_b = torch.from_numpy(np.array(img_mmap[idx:idx+1], dtype=np.float32)).to(device)
        pf_b = torch.from_numpy(np.array(patch_mmap[idx:idx+1], dtype=np.float32)).to(device)
        patch_tok_b = [pf_b[:, l_idx].contiguous() for l_idx in range(pf_b.shape[1])]
        cls_b = [cls_names_list[idx]]

        text_embeddings, _ = MyModel.forward_ensemble(PFL_TextEncoder, img_feat_b, patch_tok_b, cls_b, device, tokenize, mode="test")
        pro_img, anomaly_maps_list = MyModel(text_embeddings, img_feat_b, patch_tok_b, stage=stage, mode="test")
        pro_img = pro_img.squeeze(2)

        text_probs = torch.stack([pro_img[:, :R_tot], pro_img[:, R_tot:]], dim=-1).softmax(dim=-1)
        temp_cls = float(text_probs[0, :, 1].mean().item())

        anomaly_maps = []
        for num in range(len(anomaly_maps_list)):
            anomaly_map = anomaly_maps_list[num]
            amap_layer = torch.stack([anomaly_map[:, :R_tot], anomaly_map[:, R_tot:]], dim=2).softmax(dim=2)[:, :, 1].mean(dim=1)
            anomaly_maps.append(amap_layer.float().cpu().numpy()[0])

        amap_2d = np.mean(anomaly_maps, axis=0)
        smooth_m = gaussian_filter(amap_2d, sigma=sigma)

        amaps_list.append(smooth_m.astype(np.float16))
        pr_sp_list.append(float(temp_cls))

    raw_sp = np.array(pr_sp_list, dtype=np.float32)
    amaps_norm = norm01_f32_list(amaps_list)
    sp_norm = (raw_sp - raw_sp.min()) / (raw_sp.max() - raw_sp.min() + 1e-8)
    flat_amaps = amaps_norm.reshape(amaps_norm.shape[0], -1)
    kth = max(-2000, -flat_amaps.shape[1])
    pr_sp_tmp = np.partition(flat_amaps, kth=kth, axis=1)[:, kth:].mean(axis=1)
    pr_sp_tmp = (pr_sp_tmp - pr_sp_tmp.min()) / (pr_sp_tmp.max() - pr_sp_tmp.min() + 1e-8)
    final_sp = 0.5 * sp_norm + 0.5 * pr_sp_tmp

    metrics = eval_metrics_from_amap_and_sp(labels_arr, final_sp, gts_arr, amaps_norm, None, compute_pro=False)
    del amaps_norm
    gc.collect()
    return metrics


@torch.no_grad()
def evaluate_bayespfl(model_clip, MyModel, PFL_TextEncoder, loader, device, args, stage=1, sigma=8.0, compute_pro=True):
    model_clip.eval()
    MyModel.eval()
    MyModel.zk_n = None
    MyModel.zk_a = None

    gt_sp_list, pr_sp_list, masks_list, amaps_list = [], [], [], []
    class_masks_dict = {c: [] for c in DEFECT_CLASSES}

    R_tot = args.prompt_num * args.sample_num

    for items in tqdm(loader, desc="Eval Bayes-PFL", leave=False):
        images = items["img"].to(device)
        cls_names = items["cls_name"]
        gt_mask = (items["img_mask"] > 0.5)
        labels = items["anomaly"]
        bsz = images.shape[0]

        with torch.cuda.amp.autocast():
            image_features_all, _, patch_tokens_all = model_clip.encode_image(images, args.features_list)
        image_features_all = image_features_all.float()
        patch_tokens_all = [pt.float() for pt in patch_tokens_all]

        for b in range(bsz):
            img_feat_b = image_features_all[b:b+1]
            patch_tok_b = [pt[b:b+1] for pt in patch_tokens_all]
            cls_b = [cls_names[b]]

            text_embeddings, _ = MyModel.forward_ensemble(PFL_TextEncoder, img_feat_b, patch_tok_b, cls_b, device, tokenize, mode="test")
            pro_img, anomaly_maps_list = MyModel(text_embeddings, img_feat_b, patch_tok_b, stage=stage, mode="test")
            pro_img = pro_img.squeeze(2)

            text_probs = torch.stack([pro_img[:, :R_tot], pro_img[:, R_tot:]], dim=-1).softmax(dim=-1)
            temp_cls = float(text_probs[0, :, 1].mean().item())

            anomaly_maps = []
            for num in range(len(anomaly_maps_list)):
                anomaly_map = anomaly_maps_list[num]
                amap_layer = torch.stack([anomaly_map[:, :R_tot], anomaly_map[:, R_tot:]], dim=2).softmax(dim=2)[:, :, 1].mean(dim=1)
                anomaly_maps.append(amap_layer.float().cpu().numpy()[0])

            amap_2d = np.mean(anomaly_maps, axis=0)
            smooth_m = gaussian_filter(amap_2d, sigma=sigma)

            amaps_list.append(smooth_m.astype(np.float16))
            pr_sp_list.append(float(temp_cls))
            gt_sp_list.append(int(labels[b].item()))
            masks_list.append(gt_mask[b, 0].cpu().numpy().astype(np.uint8))
            if "class_masks" in items:
                for cname in DEFECT_CLASSES:
                    class_masks_dict[cname].append((items["class_masks"][cname][b, 0].cpu().numpy() > 0.5).astype(np.uint8))

    gt_sp = np.array(gt_sp_list)
    raw_sp = np.array(pr_sp_list, dtype=np.float32)
    masks_np = np.stack(masks_list, axis=0)
    masks_list.clear()
    amaps_norm = norm01_f32_list(amaps_list)
    gc.collect()

    sp_norm = (raw_sp - raw_sp.min()) / (raw_sp.max() - raw_sp.min() + 1e-8)
    flat_amaps = amaps_norm.reshape(amaps_norm.shape[0], -1)
    kth = max(-2000, -flat_amaps.shape[1])
    pr_sp_tmp = np.partition(flat_amaps, kth=kth, axis=1)[:, kth:].mean(axis=1)
    pr_sp_tmp = (pr_sp_tmp - pr_sp_tmp.min()) / (pr_sp_tmp.max() - pr_sp_tmp.min() + 1e-8)
    final_sp = 0.5 * sp_norm + 0.5 * pr_sp_tmp

    metrics = eval_metrics_from_amap_and_sp(gt_sp, final_sp, masks_np, amaps_norm, class_masks_dict, compute_pro=compute_pro)
    del masks_np, amaps_norm
    gc.collect()
    return metrics


def train_and_eval_bayespfl(dataset_name, short_ds, epochs=30, batch_size=16, seed=333, smoke_test=False):
    setup_seed_common(seed)
    os.chdir(BAYESPFL_DIR)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    out_root = os.path.join(VISUALAD_DIR, "outputs_smoke" if smoke_test else "outputs")
    save_dir = os.path.join(out_root, f"comp_bayespfl_{short_ds}")
    cache_dir = os.path.join(out_root, f"_cache_bayespfl_{short_ds}")
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, "train_eval.log")

    def log_print(msg):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as lf:
            lf.write(line + "\n")

    args = argparse.Namespace(
        dataset=short_ds,
        model="ViT-L-14-336",
        pretrained="openai",
        features_list=[6, 12, 18, 24],
        config_path=os.path.join(BAYESPFL_DIR, "open_clip_local/model_configs/ViT-L-14-336.json"),
        pretrained_path=os.path.join(BAYESPFL_DIR, "pretrained_weight/ViT-L-14-336px.pt"),
        num_flows=10,
        prompt_context_len=5,
        prompt_num=3,
        prompt_state_len=5,
        learning_rate=0.0001,
        learning_rate_PFL=0.001,
        learning_rate_prompt=0.001,
        learning_rate_linear=0.0001,
        sample_num=5 if not smoke_test else 2,
        batch_size=batch_size if not smoke_test else 8,
        image_size=518,
        epochs=epochs,
        device_id=0,
        seed=seed,
    )
    with open(args.config_path, "r") as f:
        model_configs = json.load(f)
    args.vision_width = model_configs["vision_cfg"]["width"]
    args.text_width = model_configs["text_cfg"]["width"]
    args.embed_dim = model_configs["embed_dim"]

    model_clip, _, _ = Load_CLIP(args.image_size, args.pretrained_path, device=device)
    model_clip.to(device)
    model_clip.eval()

    PFL_TextEncoder = TextEncoder(model_clip, args)
    MyModel = Context_Prompting(args=args).to(device)

    parameter_prompt_list = []
    parameter_prompt2_list = []
    parameter_linear_list = []
    parameter_linear_list_class = []
    for n, m in MyModel.named_parameters():
        if "prompt" in n:
            parameter_prompt_list.append(m)
        elif "PFL" in n and "encoder" not in n and "decoder" not in n:
            parameter_prompt2_list.append(m)
        elif "temperature" not in n:
            parameter_linear_list.append(m)
        if "class_mapping" in n or "image_mapping" in n or "fuse" in n:
            parameter_linear_list_class.append(m)

    optimizer_stage1 = torch.optim.Adam([
        {"params": parameter_prompt_list, "lr": args.learning_rate_prompt},
        {"params": [MyModel.temperature_pixel, MyModel.temperature_image], "lr": 0.01},
        {"params": parameter_prompt2_list, "lr": args.learning_rate_PFL},
        {"params": parameter_linear_list, "lr": args.learning_rate_linear},
    ], lr=args.learning_rate, betas=(0.5, 0.999))
    optimizer_stage2 = torch.optim.Adam([
        {"params": parameter_linear_list_class, "lr": args.learning_rate_linear}
    ], lr=args.learning_rate, betas=(0.5, 0.999))

    trainable_params = sum(p.numel() for p in MyModel.parameters() if p.requires_grad)
    log_print(f"=== Bayes-PFL ({dataset_name}, Official {epochs} Ep 2-Stage) | Trainable params: {trainable_params} ===")

    preprocess = get_bayespfl_transform(args.image_size)
    train_root = f"/home/skye/data/Skye/databases/{dataset_name}"
    train_meta = f"/home/skye/data/Skye/visualad/datasets/{dataset_name}_meta.json"
    train_ds = BridgeAnomalyDataset(train_root, "train", args.image_size, preprocess, False, train_meta)
    val_ds = BridgeAnomalyDataset(train_root, "val", args.image_size, preprocess, False, train_meta)
    test_ds = BridgeAnomalyDataset(train_root, "test", args.image_size, preprocess, True, train_meta)

    if smoke_test:
        train_ds.data_all = train_ds.data_all[:8]
        val_ds.data_all = val_ds.data_all[:8]
        test_ds.data_all = test_ds.data_all[:8]

    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    loss_dice_fn = DiceLoss()
    loss_focal_fn = FocalLoss()
    loss_ort_fn = Orthogonal_Loss()
    loss_cross = nn.CrossEntropyLoss()

    stage = 1
    optimizer = optimizer_stage1
    patience = 0
    early_stop_patience = 3
    ap_max_pixel = -1.0
    ap_max_image = -1.0

    best_val_score = -1.0
    best_ckpt = os.path.join(save_dir, "best_model.pth")
    best_ckpt_8ep = os.path.join(save_dir, "best_model_8ep.pth")
    stage1_best_ckpt = os.path.join(save_dir, "stage1_best.pth")
    history = []

    try:
        tr_img_m, tr_pf_m, tr_gts, tr_labels, tr_cls = extract_bayespfl_cache(model_clip, train_ds, device, args.features_list, cache_dir, "train")
        va_img_m, va_pf_m, va_gts, va_labels, va_cls = extract_bayespfl_cache(model_clip, val_ds, device, args.features_list, cache_dir, "val")
        n_train = len(tr_labels)

        steps_per_ep = max(n_train // args.batch_size, 1)
        total_steps = max(steps_per_ep * epochs, 1)
        warmup_steps = int(0.1 * total_steps)

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer_stage1, lr_lambda)

        for epoch in range(epochs):
            MyModel.train()
            loss_list = []
            perm = np.random.permutation(n_train)

            for start in range(0, n_train - args.batch_size + 1, args.batch_size):
                batch_idx = perm[start:start + args.batch_size]
                image_features = torch.from_numpy(np.array(tr_img_m[batch_idx], dtype=np.float32)).to(device)
                pf_batch = torch.from_numpy(np.array(tr_pf_m[batch_idx], dtype=np.float32)).to(device)
                patch_tokens = [pf_batch[:, idx].contiguous() for idx in range(pf_batch.shape[1])]
                cls_name = [tr_cls[i] for i in batch_idx]
                anomaly = torch.from_numpy(tr_labels[batch_idx]).long().to(device)
                gt = torch.from_numpy(tr_gts[batch_idx]).float().to(device)

                text_embeddings, loss_reg = MyModel.forward_ensemble(PFL_TextEncoder, image_features, patch_tokens, cls_name, device, tokenize, mode="train")
                pro_img, anomaly_maps_list = MyModel(text_embeddings, image_features, patch_tokens, stage=stage, mode="train")
                pro_img = pro_img.squeeze(2)

                index_prompt = torch.randint(0, args.prompt_num, (1, 1))[0]
                pro_img_sel = torch.cat([pro_img[:, index_prompt], pro_img[:, index_prompt + args.prompt_num]], dim=1)
                loss_class = loss_cross(pro_img_sel, anomaly)
                loss_dist_reg = loss_reg[1] + loss_reg[2] + loss_reg[3]
                loss_text = loss_ort_fn(text_embeddings, args)

                loss_foc = 0.0
                loss_dic = 0.0
                for num in range(len(anomaly_maps_list)):
                    anomaly_map = anomaly_maps_list[num]
                    anomaly_map_select = torch.cat([anomaly_map[:, index_prompt, :, :], anomaly_map[:, index_prompt + args.prompt_num, :, :]], dim=1)
                    anomaly_map_select = torch.softmax(anomaly_map_select, dim=1)
                    loss_foc = loss_foc + loss_focal_fn(anomaly_map_select.clone(), gt.clone())
                    loss_dic = loss_dic + loss_dice_fn(anomaly_map_select, gt)

                loss_seg = loss_foc + loss_dic
                if stage == 1:
                    loss = loss_class + loss_seg + loss_dist_reg + loss_text
                else:
                    loss = loss_class

                if torch.isnan(loss) or torch.isinf(loss):
                    optimizer.zero_grad()
                    continue

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(MyModel.parameters(), max_norm=2.0)
                optimizer.step()
                if stage == 1:
                    scheduler.step()
                loss_list.append(float(loss.detach().item()))

            val_m = evaluate_bayespfl_cached(MyModel, PFL_TextEncoder, va_img_m, va_pf_m, va_gts, va_labels, va_cls, device, args, stage=stage)
            val_key = (val_m["Image-AUROC"] + val_m["Pixel-AUROC"] + val_m["Pixel-F1"] + val_m["Pixel-AP"]) / 4.0
            log_print(f"Epoch [{epoch+1}/{epochs}] Stg{stage} loss={np.mean(loss_list):.4f} | Val I-AUROC={val_m['Image-AUROC']:.2f} P-AUROC={val_m['Pixel-AUROC']:.2f} P-F1={val_m['Pixel-F1']:.2f} P-AP={val_m['Pixel-AP']:.2f}")
            history.append({"epoch": epoch + 1, "stage": stage, "loss": round(float(np.mean(loss_list)), 4), "val": val_m})

            if val_key >= best_val_score:
                best_val_score = val_key
                torch.save({"MyModel": MyModel.state_dict(), "stage": stage}, best_ckpt)

            if (epoch + 1) == 8 or (smoke_test and (epoch + 1) == epochs):
                shutil.copyfile(best_ckpt, best_ckpt_8ep)

            if stage == 1:
                if val_m["Pixel-AP"] >= ap_max_pixel:
                    patience = 0
                    ap_max_pixel = val_m["Pixel-AP"]
                    torch.save({"MyModel": MyModel.state_dict(), "stage": 1}, stage1_best_ckpt)
                else:
                    patience += 1
                    if patience >= early_stop_patience and (epoch + 1) >= 8:
                        log_print("--- Bayes-PFL: Switching from Stage 1 to Stage 2 ---")
                        if os.path.exists(stage1_best_ckpt):
                            s1 = torch.load(stage1_best_ckpt, map_location=device)
                            MyModel.load_state_dict(s1["MyModel"])
                        stage = 2
                        optimizer = optimizer_stage2
                        patience = 0
            else:
                if val_m["Image-AP"] >= ap_max_image:
                    patience = 0
                    ap_max_image = val_m["Image-AP"]
                else:
                    patience += 1
                    if patience >= early_stop_patience and (epoch + 1) >= 12:
                        log_print(f"--- Bayes-PFL: Stage 2 early stop at epoch {epoch+1} ---")
                        break
        del tr_img_m, tr_pf_m, va_img_m, va_pf_m
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)

    other_ds_name = "Bridge512_balance_CODEBRIM" if dataset_name == "Bridge512_balance_S2DS" else "Bridge512_balance_S2DS"
    other_root = f"/home/skye/data/Skye/databases/{other_ds_name}"
    other_meta = f"/home/skye/data/Skye/visualad/datasets/{other_ds_name}_meta.json"
    other_test_ds = BridgeAnomalyDataset(other_root, "test", args.image_size, preprocess, True, other_meta)
    if smoke_test:
        other_test_ds.data_all = other_test_ds.data_all[:8]
    other_loader = torch.utils.data.DataLoader(other_test_ds, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    # Evaluate 8-epoch checkpoint
    in_m_8ep, cross_m_8ep = None, None
    if os.path.exists(best_ckpt_8ep):
        ckpt8 = torch.load(best_ckpt_8ep, map_location=device)
        MyModel.load_state_dict(ckpt8["MyModel"])
        stg8 = ckpt8.get("stage", 1)
        in_m_8ep = evaluate_bayespfl(model_clip, MyModel, PFL_TextEncoder, test_loader, device, args, stage=stg8, compute_pro=True)
        cross_m_8ep = evaluate_bayespfl(model_clip, MyModel, PFL_TextEncoder, other_loader, device, args, stage=stg8, compute_pro=True)
        log_print(f"[Bayes-PFL (8Ep) In-Domain {dataset_name}] {json.dumps(in_m_8ep)}")
        log_print(f"[Bayes-PFL (8Ep) Cross-Domain {dataset_name} -> {other_ds_name}] {json.dumps(cross_m_8ep)}")

    # Evaluate full official checkpoint
    ckpt = torch.load(best_ckpt, map_location=device)
    MyModel.load_state_dict(ckpt["MyModel"])
    best_stg = ckpt.get("stage", 1)

    in_m = evaluate_bayespfl(model_clip, MyModel, PFL_TextEncoder, test_loader, device, args, stage=best_stg, compute_pro=True)
    log_print(f"[Bayes-PFL (Official {epochs}Ep) In-Domain {dataset_name}] {json.dumps(in_m)}")
    cross_m = evaluate_bayespfl(model_clip, MyModel, PFL_TextEncoder, other_loader, device, args, stage=best_stg, compute_pro=True)
    log_print(f"[Bayes-PFL (Official {epochs}Ep) Cross-Domain {dataset_name} -> {other_ds_name}] {json.dumps(cross_m)}")

    summary = {
        "method": "Bayes-PFL (CVPR 2025)",
        "variant": "comp_bayespfl",
        "train_dataset": dataset_name,
        "epochs": epochs,
        "trainable_params": trainable_params,
        "in_domain_test": in_m,
        "cross_domain_test_dataset": other_ds_name,
        "cross_domain_test": cross_m,
        "in_domain_test_8ep": in_m_8ep,
        "cross_domain_test_8ep": cross_m_8ep,
        "history": history,
    }
    with open(os.path.join(save_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()
    ep = 1 if args.smoke_test else args.epochs
    for s_ds, full_ds in [("S2DS", "Bridge512_balance_S2DS"), ("CODEBRIM", "Bridge512_balance_CODEBRIM")]:
        train_and_eval_bayespfl(full_ds, s_ds, epochs=ep, smoke_test=args.smoke_test)
