import json
import os
import subprocess
import sys
import time

BASE_DIR = "/home/skye/data/Skye/visualad"
PYTHON_BIN = "/home/skye/miniconda3/envs/visualad/bin/python"
RUNNER_SCRIPT = os.path.join(BASE_DIR, "run_experiments.py")
STATUS_FILE = os.path.join(BASE_DIR, "scheduler_status.json")
LOG_FILE = os.path.join(BASE_DIR, "scheduler.log")
REPORT_FILE = os.path.join(BASE_DIR, "FINAL_COMPARISON_REPORT.md")

EXPERIMENTS = [
    {
        "id": "baseline_S2DS",
        "variant": "baseline",
        "dataset_name": "Bridge512_balance_S2DS",
        "save_dir": os.path.join(BASE_DIR, "outputs", "baseline_S2DS"),
        "epochs": 8,
    },
    {
        "id": "baseline_CODEBRIM",
        "variant": "baseline",
        "dataset_name": "Bridge512_balance_CODEBRIM",
        "save_dir": os.path.join(BASE_DIR, "outputs", "baseline_CODEBRIM"),
        "epochs": 8,
    },
    {
        "id": "ours_S2DS",
        "variant": "ours",
        "dataset_name": "Bridge512_balance_S2DS",
        "save_dir": os.path.join(BASE_DIR, "outputs", "ours_S2DS"),
        "epochs": 8,
    },
    {
        "id": "ours_CODEBRIM",
        "variant": "ours",
        "dataset_name": "Bridge512_balance_CODEBRIM",
        "save_dir": os.path.join(BASE_DIR, "outputs", "ours_CODEBRIM"),
        "epochs": 8,
    },
]


def log_msg(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def update_status(**kwargs):
    data = {}
    if os.path.exists(STATUS_FILE):
        try:
            with open(STATUS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}
    data.update(kwargs)
    data["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(STATUS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def check_gpu_free(min_free_mib: int = 14500) -> tuple:
    """
    Returns (is_free: bool, reason: str, free_mib: int, used_mib: int)
    Checks both nvidia-smi free memory and ensures previous dinomaly/v3_post_training jobs have exited.
    """
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
            text=True,
        ).strip().splitlines()[0]
        parts = [int(x.strip()) for x in out.split(",")]
        used_mib, free_mib, util = parts[0], parts[1], parts[2]
    except Exception as e:
        return False, f"nvidia-smi error: {e}", 0, 0

    # Also check if dinomaly training/post-training processes are still running
    try:
        ps_out = subprocess.check_output(["ps", "-ef"], text=True)
        for blocker in ["dinomaly_bridge.py", "run_v3_sequence.py", "v3_post_training.py", "reevaluate_v3.py"]:
            if blocker in ps_out:
                return False, f"Foreign process still active: {blocker} (used={used_mib}MiB, free={free_mib}MiB)", free_mib, used_mib
    except Exception:
        pass

    if free_mib < min_free_mib:
        return False, f"Insufficient free GPU memory: {free_mib}MiB < {min_free_mib}MiB (used={used_mib}MiB)", free_mib, used_mib

    return True, f"GPU is FREE (free={free_mib}MiB, used={used_mib}MiB, util={util}%)", free_mib, used_mib


def generate_final_report():
    results = {}
    for exp in EXPERIMENTS:
        m_file = os.path.join(exp["save_dir"], "test_metrics.json")
        if os.path.exists(m_file):
            with open(m_file, "r", encoding="utf-8") as f:
                results[exp["id"]] = json.load(f)

    lines = [
        "# Bridge Defect Anomaly Detection: VisualAD Baseline vs. Proposed Bridge-VisualAD",
        "",
        f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 1. In-Domain Test Results (Trained & Tested on Respective Dataset Splits)",
        "",
        "| Dataset | Method | Params | I-AUROC | I-F1 | I-AP | P-AUROC | P-F1 | P-AP | P-AUPRO | P-IoU | Mean (7-Metric) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for ds_short, ds_full in [("S2DS", "Bridge512_balance_S2DS"), ("CODEBRIM", "Bridge512_balance_CODEBRIM")]:
        for var_key, var_label in [(f"baseline_{ds_short}", "Baseline (VisualAD)"), (f"ours_{ds_short}", "**Ours (Bridge-VisualAD)**")]:
            if var_key in results:
                r = results[var_key]
                m = r["in_domain_test"]
                p_str = f"{r['trainable_params']/1e6:.2f}M"
                lines.append(
                    f"| {ds_short} | {var_label} | {p_str} | {m['Image-AUROC']:.2f} | {m['Image-F1']:.2f} | {m['Image-AP']:.2f} | "
                    f"{m['Pixel-AUROC']:.2f} | {m['Pixel-F1']:.2f} | {m['Pixel-AP']:.2f} | {m['Pixel-AUPRO']:.2f} | {m['Pixel-IoU']:.2f} | **{m['Mean-Score']:.2f}** |"
                )

    lines.extend([
        "",
        "## 2. Cross-Dataset Zero-Shot Generalization Results (S2DS <-> CODEBRIM)",
        "",
        "| Transfer Direction | Method | I-AUROC | I-F1 | I-AP | P-AUROC | P-F1 | P-AP | P-AUPRO | P-IoU | Mean (7-Metric) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    for ds_short, direction in [("S2DS", "S2DS -> CODEBRIM"), ("CODEBRIM", "CODEBRIM -> S2DS")]:
        for var_key, var_label in [(f"baseline_{ds_short}", "Baseline (VisualAD)"), (f"ours_{ds_short}", "**Ours (Bridge-VisualAD)**")]:
            if var_key in results:
                m = results[var_key]["cross_domain_test"]
                lines.append(
                    f"| {direction} | {var_label} | {m['Image-AUROC']:.2f} | {m['Image-F1']:.2f} | {m['Image-AP']:.2f} | "
                    f"{m['Pixel-AUROC']:.2f} | {m['Pixel-F1']:.2f} | {m['Pixel-AP']:.2f} | {m['Pixel-AUPRO']:.2f} | {m['Pixel-IoU']:.2f} | **{m['Mean-Score']:.2f}** |"
                )

    lines.extend([
        "",
        "## 3. Per-Defect-Class Pixel Recall (%) at Optimal F1 Threshold (In-Domain)",
        "",
        "| Dataset | Method | Crack | Spalling | Corrosion | Efflorescence |",
        "| :--- | :--- | :---: | :---: | :---: | :---: |",
    ])

    for ds_short in ["S2DS", "CODEBRIM"]:
        for var_key, var_label in [(f"baseline_{ds_short}", "Baseline (VisualAD)"), (f"ours_{ds_short}", "**Ours (Bridge-VisualAD)**")]:
            if var_key in results:
                pc = results[var_key]["in_domain_test"].get("Per-Class", {})
                lines.append(
                    f"| {ds_short} | {var_label} | {pc.get('Crack_Recall', 0.0):.2f} | {pc.get('Spalling_Recall', 0.0):.2f} | "
                    f"{pc.get('Corrosion_Recall', 0.0):.2f} | {pc.get('Efflorescence_Recall', 0.0):.2f} |"
                )

    lines.extend([
        "",
        "## 4. Per-Defect-Class Pixel Recall (%) on Cross-Domain Zero-Shot Transfer",
        "",
        "| Transfer Direction | Method | Crack | Spalling | Corrosion | Efflorescence |",
        "| :--- | :--- | :---: | :---: | :---: | :---: |",
    ])

    for src_short, dst_short in [("S2DS", "CODEBRIM"), ("CODEBRIM", "S2DS")]:
        for var_key, var_label in [(f"baseline_{src_short}", "Baseline (VisualAD)"), (f"ours_{src_short}", "**Ours (Bridge-VisualAD)**")]:
            if var_key in results:
                pc = results[var_key]["cross_domain_test"].get("Per-Class", {})
                lines.append(
                    f"| {src_short} -> {dst_short} | {var_label} | {pc.get('Crack_Recall', 0.0):.2f} | {pc.get('Spalling_Recall', 0.0):.2f} | "
                    f"{pc.get('Corrosion_Recall', 0.0):.2f} | {pc.get('Efflorescence_Recall', 0.0):.2f} |"
                )

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    log_msg(f"Generated final comparison report at {REPORT_FILE}")
    return REPORT_FILE


def main():
    os.makedirs(os.path.join(BASE_DIR, "outputs"), exist_ok=True)
    log_msg("GPU Scheduler Daemon started. Monitoring GPU 0 status...")
    update_status(
        daemon_pid=os.getpid(),
        state="waiting_for_gpu",
        completed_experiments=[],
        current_experiment=None,
    )

    # Wait until GPU is free for 2 consecutive checks (30s apart)
    free_streak = 0
    while free_streak < 2:
        is_free, reason, free_mib, used_mib = check_gpu_free(min_free_mib=14500)
        if is_free:
            free_streak += 1
            log_msg(f"GPU check passed ({free_streak}/2): {reason}")
        else:
            free_streak = 0
            log_msg(f"Waiting for GPU... {reason}")
            update_status(state="waiting_for_gpu", gpu_reason=reason, free_mib=free_mib, used_mib=used_mib)
        if free_streak < 2:
            time.sleep(30)

    log_msg("GPU is confirmed FREE! Starting sequential execution of Baseline and Ours on S2DS & CODEBRIM.")
    completed = []

    for exp in EXPERIMENTS:
        exp_id = exp["id"]
        metrics_file = os.path.join(exp["save_dir"], "test_metrics.json")
        if os.path.exists(metrics_file):
            log_msg(f"Experiment {exp_id} already has test_metrics.json, skipping.")
            completed.append(exp_id)
            update_status(completed_experiments=completed)
            continue

        log_msg(f">>> Launching Experiment: {exp_id} ({exp['variant']} on {exp['dataset_name']}) <<<")
        update_status(
            state="running_experiment",
            current_experiment=exp_id,
            completed_experiments=completed,
            started_current_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        )

        cmd = [
            PYTHON_BIN,
            "-u",
            RUNNER_SCRIPT,
            "--variant", exp["variant"],
            "--dataset_name", exp["dataset_name"],
            "--save_dir", exp["save_dir"],
            "--epochs", str(exp["epochs"]),
            "--batch_size", "4",
            "--accum_steps", "2",
            "--device", "cuda:0",
        ]
        ret = subprocess.run(cmd, cwd=BASE_DIR)
        if ret.returncode != 0:
            log_msg(f"ERROR: Experiment {exp_id} exited with code {ret.returncode}")
            update_status(state="failed", failed_experiment=exp_id, returncode=ret.returncode)
            sys.exit(ret.returncode)

        completed.append(exp_id)
        log_msg(f"=== Completed Experiment: {exp_id} ===")
        generate_final_report()
        update_status(
            state="running_experiment",
            current_experiment=None,
            completed_experiments=completed,
        )

    generate_final_report()
    update_status(
        state="all_completed",
        current_experiment=None,
        completed_experiments=completed,
        report_file=REPORT_FILE,
    )
    log_msg("ALL 4 EXPERIMENTS (Baseline & Ours on S2DS & CODEBRIM) COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
