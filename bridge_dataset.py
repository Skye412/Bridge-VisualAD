import csv
import json
import os
import numpy as np
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode

DEFECT_CLASSES = ["Crack", "Spalling", "Corrosion", "Efflorescence"]
DEFECT_RGB_MAP = {
    "Crack": (255, 0, 0),
    "Spalling": (0, 255, 0),
    "Corrosion": (0, 0, 255),
    "Efflorescence": (255, 255, 0),
}


def build_bridge_meta_json(dataset_root: str, output_json_path: str = None):
    """
    Build AnomalyCLIP/VisualAD-compatible meta.json from train_manifest.csv, val_manifest.csv, test_manifest.csv.
    """
    meta = {"train": {"bridge": []}, "val": {"bridge": []}, "test": {"bridge": []}}
    for split in ["train", "val", "test"]:
        manifest_path = os.path.join(dataset_root, f"{split}_manifest.csv")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        with open(manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                is_anom = 1 if row["category"] == "annotated" else 0
                img_rel = os.path.join(split, row["image_path"])
                mask_rel = os.path.join(split, row["mask_path"]) if (is_anom and row["mask_path"]) else ""
                specie = row["labels"] if row["labels"] else "normal"
                meta[split]["bridge"].append({
                    "img_path": img_rel,
                    "mask_path": mask_rel,
                    "cls_name": "bridge",
                    "specie_name": specie,
                    "anomaly": is_anom,
                    "sample_id": row["sample"],
                    "Crack_pixels": int(row.get("Crack_pixels", 0) or 0),
                    "Spalling_pixels": int(row.get("Spalling_pixels", 0) or 0),
                    "Corrosion_pixels": int(row.get("Corrosion_pixels", 0) or 0),
                    "Efflorescence_pixels": int(row.get("Efflorescence_pixels", 0) or 0),
                })
    if output_json_path:
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
    return meta


class BridgeAnomalyDataset(data.Dataset):
    """
    Dataset loader for Bridge512_balance_S2DS and Bridge512_balance_CODEBRIM.
    Properly handles RGB color-coded defect masks:
      - Crack: (255, 0, 0)
      - Spalling: (0, 255, 0)
      - Corrosion: (0, 0, 255)
      - Efflorescence: (255, 255, 0)
    Any non-zero channel pixel ((arr > 0).any(axis=-1)) is treated as ground-truth anomaly=1.
    Also optionally returns per-class binary masks for fine-grained evaluation.
    """
    def __init__(self, root: str, split: str = "train", image_size: int = 518,
                 preprocess=None, return_class_masks: bool = False, meta_json_path: str = None):
        super().__init__()
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
            meta = build_bridge_meta_json(root)

        self.data_all = meta[split]["bridge"]
        self.mask_resize = transforms.Resize(
            (image_size, image_size), interpolation=InterpolationMode.NEAREST
        )

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

        mask_pil = Image.fromarray(bin_mask_np, mode="L")
        mask_pil = self.mask_resize(mask_pil)
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
                cm_pil = self.mask_resize(Image.fromarray(cm, mode="L"))
                class_masks[cname] = (transforms.ToTensor()(cm_pil) > 0.5).float()
            out["class_masks"] = class_masks

        return out


if __name__ == "__main__":
    for ds_name in ["Bridge512_balance_S2DS", "Bridge512_balance_CODEBRIM"]:
        ds_root = f"/home/skye/data/Skye/databases/{ds_name}"
        out_json = f"/home/skye/data/Skye/visualad/datasets/{ds_name}_meta.json"
        meta = build_bridge_meta_json(ds_root, out_json)
        print(f"Generated {out_json}:")
        for sp in ["train", "val", "test"]:
            items = meta[sp]["bridge"]
            n_anom = sum(x["anomaly"] for x in items)
            n_norm = len(items) - n_anom
            print(f"  {sp}: total={len(items)}, normal={n_norm}, anomaly={n_anom}")
        ds_obj = BridgeAnomalyDataset(ds_root, split="train", image_size=518, return_class_masks=True, meta_json_path=out_json)
        sample = ds_obj[0]
        print(f"  Sample[0]: img={sample['img'].shape}, mask={sample['img_mask'].shape}, anomaly={sample['anomaly']}")
