import os
import shutil
import torch

BASE_DIR = "/home/skye/data/Skye/visualad"
ARCHIVE_DIR = os.path.join(BASE_DIR, "archive_v1_inference_variants")
os.makedirs(ARCHIVE_DIR, exist_ok=True)

# 1. Preserve original inference-variant ablation files & report & figures
for item in ["FINAL_COMPARISON_REPORT.md", "run_ablation_and_figures.py", "calibrate_and_finalize_report.py"]:
    src = os.path.join(BASE_DIR, item)
    if os.path.exists(src):
        shutil.copy2(src, os.path.join(ARCHIVE_DIR, item))

abl_json = os.path.join(BASE_DIR, "outputs", "ablation_metrics.json")
if os.path.exists(abl_json):
    shutil.copy2(abl_json, os.path.join(ARCHIVE_DIR, "ablation_metrics_inference_variants.json"))
    shutil.copy2(abl_json, os.path.join(BASE_DIR, "outputs", "ablation_metrics_inference_variants.json"))

fig_dir = os.path.join(BASE_DIR, "figures")
arch_fig_dir = os.path.join(ARCHIVE_DIR, "figures")
if os.path.exists(fig_dir):
    if os.path.exists(arch_fig_dir):
        shutil.rmtree(arch_fig_dir)
    shutil.copytree(fig_dir, arch_fig_dir)

# Also copy all test_metrics.json and train_eval.log from existing 4 folders into archive
for exp_name in ["baseline_S2DS", "baseline_CODEBRIM", "ours_S2DS", "ours_CODEBRIM"]:
    exp_dir = os.path.join(BASE_DIR, "outputs", exp_name)
    arch_exp_dir = os.path.join(ARCHIVE_DIR, exp_name)
    os.makedirs(arch_exp_dir, exist_ok=True)
    for fn in ["test_metrics.json", "train_eval.log"]:
        p = os.path.join(exp_dir, fn)
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(arch_exp_dir, fn))

    # Convert best_model.pth to fp16 alongside the untouched best_model.pth so it is <100MB for GitHub
    best_pth = os.path.join(exp_dir, "best_model.pth")
    fp16_pth = os.path.join(exp_dir, "best_model_fp16.pth")
    if os.path.exists(best_pth) and not os.path.exists(fp16_pth):
        state = torch.load(best_pth, map_location="cpu")
        def _to_fp16(obj):
            if isinstance(obj, torch.Tensor) and obj.is_floating_point():
                return obj.half()
            elif isinstance(obj, dict):
                return {k: _to_fp16(v) for k, v in obj.items()}
            return obj
        state_fp16 = _to_fp16(state)
        torch.save(state_fp16, fp16_pth)
        sz_mb = os.path.getsize(fp16_pth) / (1024 * 1024)
        print(f"Created {fp16_pth} ({sz_mb:.2f} MB) while preserving original {best_pth}")

# Write .gitignore so >100MB raw .pth and zip files don't block GitHub push, while tracking best_model_fp16.pth
gitignore_content = """__pycache__/
*.pyc
VisualAD.zip
outputs/*/best_model.pth
outputs/*/final_model.pth
VisualAD_repo/.git
"""
with open(os.path.join(BASE_DIR, ".gitignore"), "w", encoding="utf-8") as f:
    f.write(gitignore_content)

print("Archive and git preparation completed successfully.")
