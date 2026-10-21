import os, sys, subprocess, time
CUR = "/home/skye/data/Skye/visualad"
PY = "/home/skye/miniconda3/envs/visualad/bin/python"
OUT = os.path.join(CUR, "outputs", "seeds")
os.makedirs(OUT, exist_ok=True)
SEEDS = [222, 333]   # fixed before any run; original runs used seed 111
DS = [("S2DS", "Bridge512_balance_S2DS"), ("CODEBRIM", "Bridge512_balance_CODEBRIM")]
log = open(os.path.join(OUT, "seed_runs_master.log"), "a")
def L(m):
    line = time.strftime("[%Y-%m-%d %H:%M:%S] ") + m
    print(line, flush=True); log.write(line + "\n"); log.flush()
for seed in SEEDS:
    for short, full in DS:
        d0 = os.path.join(OUT, f"baseline_{short}_seed{seed}")
        if not os.path.exists(os.path.join(d0, "test_metrics.json")):
            L(f"START A0 {short} seed={seed}")
            r = subprocess.run([PY, os.path.join(CUR, "run_experiments.py"), "--variant", "baseline",
                                "--dataset_name", full, "--save_dir", d0, "--seed", str(seed)])
            L(f"END A0 {short} seed={seed} rc={r.returncode}")
        d3 = os.path.join(OUT, f"a3_agds_wgd_{short}_seed{seed}")
        if not os.path.exists(os.path.join(d3, "test_metrics.json")):
            L(f"START A3 {short} seed={seed}")
            code = ("import sys; sys.path.insert(0, %r); "
                    "from run_true_ablation_train import train_and_evaluate_ablation_variant as T; "
                    "T(variant='a3_agds_wgd', dataset_name=%r, save_dir=%r, seed=%d)") % (CUR, full, d3, seed)
            r = subprocess.run([PY, "-c", code], cwd=CUR)
            L(f"END A3 {short} seed={seed} rc={r.returncode}")
L("ALL 8 SEED RUNS FINISHED")
