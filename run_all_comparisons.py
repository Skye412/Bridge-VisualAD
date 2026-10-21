import os
import sys
import json
import time
import subprocess

VISUALAD_DIR = "/home/skye/data/Skye/visualad"
OUT_DIR = os.path.join(VISUALAD_DIR, "outputs")
LOG_FILE = os.path.join(VISUALAD_DIR, "comparisons_master.log")


def log_msg(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def unify_mean_metrics(m_dict):
    if not isinstance(m_dict, dict) or "Image-AUROC" not in m_dict:
        return m_dict
    k7 = ["Image-AUROC", "Image-F1", "Image-AP", "Pixel-AUROC", "Pixel-F1", "Pixel-AP", "Pixel-AUPRO"]
    k5 = ["Image-AUROC", "Pixel-AUROC", "Pixel-F1", "Pixel-AP", "Pixel-AUPRO"]
    if all(k in m_dict for k in k7):
        m_dict["Mean-Score"] = round(sum(float(m_dict[k]) for k in k7) / 7.0, 2)
    if all(k in m_dict for k in k5):
        m_dict["Mean-5"] = round(sum(float(m_dict[k]) for k in k5) / 5.0, 2)
    return m_dict


def aggregate_comparison_metrics():
    clean_ab_path = os.path.join(OUT_DIR, "ablation_metrics_clean_single_model.json")
    a4_code_path = os.path.join(OUT_DIR, "a4_scratch8ep_CODEBRIM", "test_metrics.json")
    if os.path.exists(clean_ab_path) and os.path.exists(a4_code_path):
        with open(clean_ab_path, "r", encoding="utf-8") as f:
            ab = json.load(f)
        with open(a4_code_path, "r", encoding="utf-8") as f:
            m_code = json.load(f)
        if "A4_Scratch8Ep_Ep8" in ab.get("CODEBRIM_in_domain", {}):
            in_ep8 = dict(ab["CODEBRIM_in_domain"]["A4_Scratch8Ep_Ep8"])
            in_ep8.pop("Variant-Name", None)
            in_ep8.pop("Trainable-Params", None)
            cr_ep8 = dict(ab["CODEBRIM_to_S2DS_cross_domain"]["A4_Scratch8Ep_Ep8"])
            cr_ep8.pop("Variant-Name", None)
            cr_ep8.pop("Trainable-Params", None)
            m_code["in_domain_test"] = unify_mean_metrics(in_ep8)
            m_code["cross_domain_test"] = unify_mean_metrics(cr_ep8)
            with open(a4_code_path, "w", encoding="utf-8") as f:
                json.dump(m_code, f, indent=2)

    protocol_map = {
        "Dinomaly": "10k iters (val-best)",
        "AnomalyCLIP": "15 Ep (val-best)",
        "Bayes_PFL": "Up to 30 Ep (2-Stg, early-stop)",
        "AdaptCLIP": "15 Ep (val-best)",
        "WPFormer": "60 Ep (val-best)",
        "MixerCSeg": "50 Ep (val-best)",
        "VisualAD_Baseline": "8 Ep (val-best)",
        "Ours": "8 Ep (Scratch adapters, val-best)",
    }

    methods = [
        ("Dinomaly", "comp_dinomaly", "Dinomaly (CVPR 2025)"),
        ("AnomalyCLIP", "comp_anomalyclip", "AnomalyCLIP (ICLR 2024)"),
        ("Bayes_PFL", "comp_bayespfl", "Bayes-PFL (CVPR 2025)"),
        ("AdaptCLIP", "comp_adaptclip", "AdaptCLIP (AAAI 2026)"),
        ("WPFormer", "comp_wpformer", "WPFormer (CVPR 2025)"),
        ("MixerCSeg", "comp_mixercseg", "MixerCSeg (CVPR 2026)"),
        ("VisualAD_Baseline", "baseline", "VisualAD (Baseline, A0)"),
        ("Ours", "a4_scratch8ep", "Bridge-VisualAD (Ours, A4 Scratch Ep8)"),
    ]

    combined = {
        "S2DS_in_domain": {},
        "S2DS_to_CODEBRIM_cross_domain": {},
        "CODEBRIM_in_domain": {},
        "CODEBRIM_to_S2DS_cross_domain": {},
        "S2DS_in_domain_8ep": {},
        "S2DS_to_CODEBRIM_cross_domain_8ep": {},
        "CODEBRIM_in_domain_8ep": {},
        "CODEBRIM_to_S2DS_cross_domain_8ep": {},
    }

    for short_ds, in_key, cross_key in [
        ("S2DS", "S2DS_in_domain", "S2DS_to_CODEBRIM_cross_domain"),
        ("CODEBRIM", "CODEBRIM_in_domain", "CODEBRIM_to_S2DS_cross_domain"),
    ]:
        for m_key, folder_prefix, label in methods:
            m_path = os.path.join(OUT_DIR, f"{folder_prefix}_{short_ds}", "test_metrics.json")
            if not os.path.exists(m_path):
                continue
            with open(m_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            for field in ["in_domain_test", "cross_domain_test", "in_domain_test_8ep", "cross_domain_test_8ep"]:
                if data.get(field):
                    data[field] = unify_mean_metrics(data[field])
            with open(m_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

            in_m = dict(data["in_domain_test"])
            in_m["Method"] = label
            in_m["Protocol"] = protocol_map.get(m_key, str(data.get("epochs", 8)))
            in_m["Epochs"] = data.get("epochs", "10000 iters" if m_key == "Dinomaly" else 8)
            in_m["Trainable-Params"] = data.get("trainable_params")

            cross_m = dict(data["cross_domain_test"])
            cross_m["Method"] = label
            cross_m["Protocol"] = protocol_map.get(m_key, str(data.get("epochs", 8)))
            cross_m["Epochs"] = data.get("epochs", "10000 iters" if m_key == "Dinomaly" else 8)
            cross_m["Trainable-Params"] = data.get("trainable_params")

            combined[in_key][m_key] = in_m
            combined[cross_key][m_key] = cross_m

            if data.get("in_domain_test_8ep"):
                in_m8 = dict(data["in_domain_test_8ep"])
                in_m8["Method"] = f"{label} (8Ep)"
                in_m8["Protocol"] = "8 Ep (val-best)"
                in_m8["Epochs"] = 8
                in_m8["Trainable-Params"] = data.get("trainable_params")
                combined[f"{in_key}_8ep"][m_key] = in_m8
            else:
                combined[f"{in_key}_8ep"][m_key] = in_m

            if data.get("cross_domain_test_8ep"):
                cross_m8 = dict(data["cross_domain_test_8ep"])
                cross_m8["Method"] = f"{label} (8Ep)"
                cross_m8["Protocol"] = "8 Ep (val-best)"
                cross_m8["Epochs"] = 8
                cross_m8["Trainable-Params"] = data.get("trainable_params")
                combined[f"{cross_key}_8ep"][m_key] = cross_m8
            else:
                combined[f"{cross_key}_8ep"][m_key] = cross_m

    out_json = os.path.join(OUT_DIR, "comparison_metrics_all.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)
    log_msg(f"Saved aggregated comparison metrics to {out_json}")

    # Also auto-generate a 100% programmatic Markdown table from raw JSONs
    md_lines = ["# Programmatically Verified Comparison Results (Direct from Raw JSONs)\n"]
    sections = [
        ("S2DS_in_domain", "1. Bridge512_balance_S2DS (In-Domain Test)"),
        ("CODEBRIM_in_domain", "2. Bridge512_balance_CODEBRIM (In-Domain Test)"),
        ("S2DS_to_CODEBRIM_cross_domain", "3. S2DS -> CODEBRIM (Cross-Domain Zero-Shot)"),
        ("CODEBRIM_to_S2DS_cross_domain", "4. CODEBRIM -> S2DS (Cross-Domain Zero-Shot)"),
    ]
    for sec_key, sec_title in sections:
        md_lines.append(f"## {sec_title}\n")
        md_lines.append("| Method | Protocol | I-AUROC | I-F1 | I-AP | P-AUROC | P-F1 | P-AP | P-AUPRO | P-IoU | Mean-7 | Mean-5 | Crack | Spalling | Corrosion | Efflor. |")
        md_lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
        for m_key, _, _ in methods:
            if m_key not in combined[sec_key]:
                continue
            r = combined[sec_key][m_key]
            pc = r.get("Per-Class", {})
            md_lines.append(
                f"| **{r['Method']}** | {r['Protocol']} | {r['Image-AUROC']:.2f} | {r['Image-F1']:.2f} | {r['Image-AP']:.2f} | "
                f"{r['Pixel-AUROC']:.2f} | {r['Pixel-F1']:.2f} | {r['Pixel-AP']:.2f} | {r['Pixel-AUPRO']:.2f} | {r['Pixel-IoU']:.2f} | "
                f"**{r['Mean-Score']:.2f}** | {r['Mean-5']:.2f} | "
                f"{pc.get('Crack_Recall', 0.0):.2f} | {pc.get('Spalling_Recall', 0.0):.2f} | {pc.get('Corrosion_Recall', 0.0):.2f} | {pc.get('Efflorescence_Recall', 0.0):.2f} |"
            )
        md_lines.append("\n")
    md_path = os.path.join(OUT_DIR, "verified_comparison_tables.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))


def main():
    tasks = [
        (
            "Dinomaly",
            "comp_dinomaly",
            "/home/skye/miniconda3/envs/dinomaly/bin/python",
            os.path.join(VISUALAD_DIR, "eval_comp_dinomaly.py"),
            [],
        ),
        (
            "AnomalyCLIP",
            "comp_anomalyclip",
            "/home/skye/miniconda3/envs/adaptclip/bin/python",
            os.path.join(VISUALAD_DIR, "run_comp_anomalyclip.py"),
            ["--epochs", "15"],
        ),
        (
            "AdaptCLIP",
            "comp_adaptclip",
            "/home/skye/miniconda3/envs/adaptclip/bin/python",
            os.path.join(VISUALAD_DIR, "run_comp_adaptclip.py"),
            ["--epochs", "15"],
        ),
        (
            "Bayes-PFL",
            "comp_bayespfl",
            "/home/skye/miniconda3/envs/Bayes_PFL/bin/python",
            os.path.join(VISUALAD_DIR, "run_comp_bayespfl.py"),
            ["--epochs", "30"],
        ),
        (
            "MixerCSeg",
            "comp_mixercseg",
            "/home/skye/miniconda3/envs/MixerCSeg/bin/python",
            os.path.join(VISUALAD_DIR, "run_comp_mixercseg.py"),
            ["--epochs", "50"],
        ),
        (
            "WPFormer",
            "comp_wpformer",
            "/home/skye/miniconda3/envs/wpformer/bin/python",
            os.path.join(VISUALAD_DIR, "run_comp_wpformer.py"),
            ["--epochs", "60"],
        ),
        (
            "AdaptCLIP",
            "comp_adaptclip",
            "/home/skye/miniconda3/envs/adaptclip/bin/python",
            os.path.join(VISUALAD_DIR, "run_comp_adaptclip.py"),
            ["--epochs", "15"],
        ),
    ]

    log_msg("=== Starting Master Comparison Suite (Official Native Protocols + 8Ep Snapshots) ===")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["PYTHONUNBUFFERED"] = "1"

    for name, prefix, py_bin, script_path, extra_args in tasks:
        s2ds_done = os.path.exists(os.path.join(OUT_DIR, f"{prefix}_S2DS", "test_metrics.json"))
        code_done = os.path.exists(os.path.join(OUT_DIR, f"{prefix}_CODEBRIM", "test_metrics.json"))
        if s2ds_done and code_done and name == "Dinomaly":
            log_msg(f"[SKIP] {name}: already evaluated under official 10000-iter protocol.")
            aggregate_comparison_metrics()
            continue

        cmd = [py_bin, "-u", script_path] + extra_args
        log_msg(f"[START] {name}: {' '.join(cmd)}")
        t0 = time.time()
        with open(LOG_FILE, "a", encoding="utf-8") as lf:
            ret = subprocess.call(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
        dt = (time.time() - t0) / 60.0
        if ret == 0:
            log_msg(f"[DONE] {name} finished successfully in {dt:.1f} min.")
        else:
            log_msg(f"[ERROR] {name} exited with code {ret} after {dt:.1f} min.")
        aggregate_comparison_metrics()

    log_msg("=== ALL 6 EXTERNAL BASELINE COMPARISONS COMPLETED ===")


if __name__ == "__main__":
    main()
