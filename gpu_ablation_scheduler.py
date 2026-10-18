import json
import os
import subprocess
import sys
import time

BASE_DIR = "/home/skye/data/Skye/visualad"
PYTHON_BIN = "/home/skye/miniconda3/envs/visualad/bin/python"
LOG_FILE = os.path.join(BASE_DIR, "ablation_scheduler.log")
STATUS_FILE = os.path.join(BASE_DIR, "ablation_scheduler_status.json")

EXPERIMENTS = [
    ("a1_mp_bs", "Bridge512_balance_S2DS", "a1_mp_bs_S2DS"),
    ("a2_wd_saf", "Bridge512_balance_S2DS", "a2_wd_saf_S2DS"),
    ("a3_agds_wgd", "Bridge512_balance_S2DS", "a3_agds_wgd_S2DS"),
    ("a1_mp_bs", "Bridge512_balance_CODEBRIM", "a1_mp_bs_CODEBRIM"),
    ("a2_wd_saf", "Bridge512_balance_CODEBRIM", "a2_wd_saf_CODEBRIM"),
    ("a3_agds_wgd", "Bridge512_balance_CODEBRIM", "a3_agds_wgd_CODEBRIM"),
]


def log_msg(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def update_status(state: str, current_job: str = None, completed: list = None, gpu_info: dict = None):
    payload = {
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "state": state,
        "current_job": current_job,
        "completed_jobs": completed or [],
        "gpu_info": gpu_info or {},
    }
    with open(STATUS_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def get_gpu_status():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
            universal_newlines=True,
        ).strip()
        line = out.splitlines()[0]
        parts = [int(x.strip()) for x in line.split(",")]
        return {"mem_used_mib": parts[0], "mem_total_mib": parts[1], "util_pct": parts[2]}
    except Exception as e:
        log_msg(f"Error querying nvidia-smi: {e}")
        return {"mem_used_mib": 99999, "mem_total_mib": 16384, "util_pct": 100}


def wait_until_gpu_idle(mem_threshold_mib: int = 1500, util_threshold_pct: int = 10, poll_sec: int = 120, completed: list = None):
    consecutive_idle = 0
    while True:
        info = get_gpu_status()
        is_free = (info["mem_used_mib"] < mem_threshold_mib) and (info["util_pct"] <= util_threshold_pct)
        if is_free:
            consecutive_idle += 1
            log_msg(f"GPU idle check {consecutive_idle}/2 passed: {info}")
            if consecutive_idle >= 2:
                return info
            time.sleep(30)
        else:
            consecutive_idle = 0
            log_msg(f"GPU currently in use ({info['mem_used_mib']} MiB / {info['util_pct']}% util). Waiting {poll_sec}s...")
            update_status("waiting_for_gpu", current_job=None, completed=completed, gpu_info=info)
            time.sleep(poll_sec)


def is_exp_done(exp_folder: str) -> bool:
    out_dir = os.path.join(BASE_DIR, "outputs", exp_folder)
    return (
        os.path.exists(os.path.join(out_dir, "best_model.pth"))
        and os.path.exists(os.path.join(out_dir, "test_metrics.json"))
    )


def main():
    log_msg("=== True Standalone Ablation GPU Scheduler Started ===")
    completed = [folder for _, _, folder in EXPERIMENTS if is_exp_done(folder)]

    for variant, ds_name, folder in EXPERIMENTS:
        if is_exp_done(folder):
            log_msg(f"Experiment {folder} already completed, skipping.")
            if folder not in completed:
                completed.append(folder)
            continue

        gpu_info = wait_until_gpu_idle(completed=completed)
        log_msg(f"Starting standalone ablation training: {folder} (variant={variant}, dataset={ds_name})")
        update_status("running_ablation", current_job=folder, completed=completed, gpu_info=gpu_info)

        cmd = [
            PYTHON_BIN,
            os.path.join(BASE_DIR, "run_true_ablation_train.py"),
            "--variant", variant,
            "--dataset_name", ds_name,
            "--epochs", "8",
            "--batch_size", "4",
            "--accum_steps", "2",
        ]
        ret = subprocess.call(cmd, cwd=BASE_DIR)
        if ret == 0 and is_exp_done(folder):
            completed.append(folder)
            log_msg(f"Successfully finished {folder}!")
        else:
            log_msg(f"Warning: {folder} exited with code {ret}. Will retry after GPU check.")

    log_msg("All 6 standalone ablation trainings completed! Generating aggregated metrics and figures...")
    update_status("generating_figures", current_job="generate_true_ablation_figures_and_report.py", completed=completed)
    subprocess.call([PYTHON_BIN, os.path.join(BASE_DIR, "generate_true_ablation_figures_and_report.py")], cwd=BASE_DIR)

    # Commit and push updated metrics & figures to GitHub
    subprocess.call(
        "git add outputs/ablation_metrics_true_standalone.json outputs/a1_* outputs/a2_* outputs/a3_* figures_true_ablation/ && "
        "git commit -m 'Add true standalone trained ablation results (A1, A2, A3) and comparison figures' && "
        "git push origin main",
        shell=True,
        cwd=BASE_DIR,
    )
    update_status("all_done", current_job=None, completed=completed, gpu_info=get_gpu_status())
    log_msg("=== All Standalone Ablation Experiments & GitHub Sync Completed! ===")


if __name__ == "__main__":
    main()
