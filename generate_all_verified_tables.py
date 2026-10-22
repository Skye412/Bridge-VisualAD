import json
import os

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(CUR_DIR, "outputs")


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def calc_mean7(m):
    keys = ["Image-AUROC", "Image-F1", "Image-AP", "Pixel-AUROC", "Pixel-F1", "Pixel-AP", "Pixel-AUPRO"]
    return round(sum(float(m[k]) for k in keys) / 7.0, 2)


def calc_mean5(m):
    keys = ["Image-AUROC", "Pixel-AUROC", "Pixel-F1", "Pixel-AP", "Pixel-AUPRO"]
    return round(sum(float(m[k]) for k in keys) / 5.0, 2)


def format_row(name, proto, m):
    pc = m.get("Per-Class", {})
    m7 = m.get("Mean-Score", calc_mean7(m))
    m5 = m.get("Mean-5", calc_mean5(m))
    return (
        f"| **{name}** | {proto} | "
        f"{m['Image-AUROC']:.2f} | {m['Image-F1']:.2f} | {m['Image-AP']:.2f} | "
        f"{m['Pixel-AUROC']:.2f} | {m['Pixel-F1']:.2f} | {m['Pixel-AP']:.2f} | "
        f"{m['Pixel-AUPRO']:.2f} | {m['Pixel-IoU']:.2f} | "
        f"**{m7:.2f}** | {m5:.2f} | "
        f"{pc.get('Crack_Recall', 0.0):.2f} | {pc.get('Spalling_Recall', 0.0):.2f} | "
        f"{pc.get('Corrosion_Recall', 0.0):.2f} | {pc.get('Efflorescence_Recall', 0.0):.2f} |"
    )


def build_all_verified_tables():
    stage_specs = [
        ("A0_Baseline", "baseline", "A0 (Baseline: VisualAD)", "8 Ep (Scratch, val-best)"),
        ("A1_MP_BS", "a1_mp_bs", "A1 (+MP-BS)", "8 Ep (Scratch, val-best)"),
        ("A2_WD_SAF", "a2_wd_saf", "A2 (+MP-BS +WD-SAF)", "8 Ep (Scratch, val-best)"),
        ("A3_AGDS_WGD", "a3_agds_wgd", "A3 (+MP-BS +WD-SAF +AGDS-WGD)", "8 Ep (Scratch, val-best)"),
        ("A3_plus_CW", "a3_plus_cw_8ep", "A3 + CW (Boundary Loss & Gating)", "8 Ep (Scratch, val-best)"),
        ("A3_plus_Scorer", "a3_plus_scorer_8ep", "A3 + Scorer (Dynamic Area Scorer)", "8 Ep (Scratch, val-best)"),
        ("A4_Full_Ours", "a4_scratch8ep", "A4 (Ours Full: +CW +Scorer)", "8 Ep (Scratch, val-best)"),
    ]

    ablation_all = {
        "S2DS_in_domain": {},
        "CODEBRIM_in_domain": {},
        "S2DS_to_CODEBRIM_cross_domain": {},
        "CODEBRIM_to_S2DS_cross_domain": {},
    }

    for short_ds, in_sc, cross_sc in [
        ("S2DS", "S2DS_in_domain", "S2DS_to_CODEBRIM_cross_domain"),
        ("CODEBRIM", "CODEBRIM_in_domain", "CODEBRIM_to_S2DS_cross_domain"),
    ]:
        for key, folder, label, proto in stage_specs:
            d = load_json(os.path.join(OUT_DIR, f"{folder}_{short_ds}", "test_metrics.json"))
            if d is None:
                continue
            for sc, d_field in [(in_sc, "in_domain_test"), (cross_sc, "cross_domain_test")]:
                m = dict(d[d_field])
                m["Mean-Score"] = calc_mean7(m)
                m["Mean-5"] = calc_mean5(m)
                m["Variant-Name"] = label
                m["Protocol"] = proto
                m["Trainable-Params"] = d.get("trainable_params")
                ablation_all[sc][key] = m

            if "in_domain_test_linear_fusion" in d:
                for sc, d_field in [(in_sc, "in_domain_test_linear_fusion"), (cross_sc, "cross_domain_test_linear_fusion")]:
                    m_lin = dict(d[d_field])
                    m_lin["Mean-Score"] = calc_mean7(m_lin)
                    m_lin["Mean-5"] = calc_mean5(m_lin)
                    m_lin["Variant-Name"] = f"{label} [Linear Fusion]"
                    m_lin["Protocol"] = "8 Ep (Linear Fusion Eval)"
                    m_lin["Trainable-Params"] = d.get("trainable_params")
                    ablation_all[sc][f"{key}_LinearFusion"] = m_lin

    with open(os.path.join(OUT_DIR, "ablation_metrics_true_standalone.json"), "w", encoding="utf-8") as f:
        json.dump(ablation_all, f, indent=2, ensure_ascii=False)
    with open(os.path.join(OUT_DIR, "ablation_metrics_clean_single_model.json"), "w", encoding="utf-8") as f:
        json.dump(ablation_all, f, indent=2, ensure_ascii=False)

    comp_specs = [
        ("Dinomaly (CVPR 2025)", "comp_dinomaly", "10k iters (val-best)"),
        ("AnomalyCLIP (ICLR 2024)", "comp_anomalyclip", "15 Ep (val-best)"),
        ("Bayes-PFL (CVPR 2025)", "comp_bayespfl", "Up to 30 Ep (2-Stg, early-stop)"),
        ("AdaptCLIP (AAAI 2026)", "comp_adaptclip", "15 Ep (val-best)"),
        ("WPFormer (CVPR 2025)", "comp_wpformer", "60 Ep (val-best)"),
        ("MixerCSeg (CVPR 2026)", "comp_mixercseg", "50 Ep (S2DS) / 8 Ep (CODEBRIM)"),
        ("VisualAD (Baseline, A0)", "baseline", "8 Ep (Scratch, val-best)"),
        ("Bridge-VisualAD (Ours, A4)", "a4_scratch8ep", "8 Ep (Scratch, val-best)"),
    ]

    scenarios = [
        ("1. Bridge512_balance_S2DS (In-Domain Test)", "S2DS", "in_domain_test", "S2DS_in_domain"),
        ("2. Bridge512_balance_CODEBRIM (In-Domain Test)", "CODEBRIM", "in_domain_test", "CODEBRIM_in_domain"),
        ("3. S2DS -> CODEBRIM (Cross-Domain Zero-Shot)", "S2DS", "cross_domain_test", "S2DS_to_CODEBRIM_cross_domain"),
        ("4. CODEBRIM -> S2DS (Cross-Domain Zero-Shot)", "CODEBRIM", "cross_domain_test", "CODEBRIM_to_S2DS_cross_domain"),
    ]

    header = (
        "| Method / Variant | Protocol | I-AUROC | I-F1 | I-AP | P-AUROC | P-F1 | P-AP | P-AUPRO | P-IoU | "
        "Mean-7 | Mean-5 | Crack | Spalling | Corrosion | Efflor. |\n"
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | "
        ":---: | :---: | :---: | :---: | :---: | :---: |"
    )

    md_lines = [
        "# Programmatically Verified Master Tables (100% Direct from Raw `test_metrics.json`)",
        "",
        "## Part I: Main SOTA Comparison Tables",
        "",
    ]

    for title, short_ds, field, _ in scenarios:
        md_lines.append(f"### {title}")
        md_lines.append("")
        md_lines.append(header)
        for name, folder, proto in comp_specs:
            d = load_json(os.path.join(OUT_DIR, f"{folder}_{short_ds}", "test_metrics.json"))
            if d is None or field not in d:
                continue
            ep_proto = f"{d.get('epochs')} Ep (val-best)" if folder == "comp_mixercseg" else proto
            md_lines.append(format_row(name, ep_proto, d[field]))
        md_lines.append("")

    md_lines.append("## Part II: Progressive Module Ablation (`A0 -> A1 -> A2 -> A3 -> A4`, Standalone 8 Epochs)")
    md_lines.append("")
    prog_keys = ["A0_Baseline", "A1_MP_BS", "A2_WD_SAF", "A3_AGDS_WGD", "A4_Full_Ours"]
    for title, _, _, sc_key in scenarios:
        md_lines.append(f"### {title}")
        md_lines.append("")
        md_lines.append(header)
        for k in prog_keys:
            if k in ablation_all[sc_key]:
                m = ablation_all[sc_key][k]
                md_lines.append(format_row(m["Variant-Name"], m["Protocol"], m))
        md_lines.append("")

    md_lines.append("## Part III: 8-Epoch Component Ablation (`A3` vs `A3+CW` vs `A3+Scorer` vs `A4`)")
    md_lines.append("")
    comp_abl_keys = ["A3_AGDS_WGD", "A3_plus_CW", "A3_plus_CW_LinearFusion", "A3_plus_Scorer", "A4_Full_Ours", "A4_Full_Ours_LinearFusion"]
    for title, _, _, sc_key in scenarios:
        md_lines.append(f"### {title}")
        md_lines.append("")
        md_lines.append(header)
        for k in comp_abl_keys:
            if k in ablation_all[sc_key]:
                m = ablation_all[sc_key][k]
                md_lines.append(format_row(m["Variant-Name"], m["Protocol"], m))
        md_lines.append("")

    md_lines.append("## Part IV: 4-Scenario Macro-Average Summary (Direct Arithmetic Mean of the 4 Scenarios)")
    md_lines.append("")
    md_lines.append(
        "| Variant (Standalone 8 Ep) | S2DS In | CODEBRIM In | S2DS->CODEBRIM | CODEBRIM->S2DS | "
        "**Macro Mean-7** | **Macro Mean-5** | **Macro Crack Recall** |"
    )
    md_lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    all_keys = ["A0_Baseline", "A1_MP_BS", "A2_WD_SAF", "A3_AGDS_WGD", "A3_plus_CW", "A3_plus_Scorer", "A4_Full_Ours"]
    sc_order = ["S2DS_in_domain", "CODEBRIM_in_domain", "S2DS_to_CODEBRIM_cross_domain", "CODEBRIM_to_S2DS_cross_domain"]
    for k in all_keys:
        if all(k in ablation_all[sc] for sc in sc_order):
            m7_list = [ablation_all[sc][k]["Mean-Score"] for sc in sc_order]
            m5_list = [ablation_all[sc][k]["Mean-5"] for sc in sc_order]
            cr_list = [ablation_all[sc][k]["Per-Class"]["Crack_Recall"] for sc in sc_order]
            name = ablation_all["S2DS_in_domain"][k]["Variant-Name"]
            md_lines.append(
                f"| **{name}** | {m7_list[0]:.2f} | {m7_list[1]:.2f} | {m7_list[2]:.2f} | {m7_list[3]:.2f} | "
                f"**{sum(m7_list)/4.0:.2f}** | **{sum(m5_list)/4.0:.2f}** | **{sum(cr_list)/4.0:.2f}** |"
            )
    md_lines.append("")

    dacl_data = load_json(os.path.join(OUT_DIR, "dacl10k_v3_zero_shot_results.json"))
    if dacl_data is not None:
        md_lines.append("## Part V: Third-Domain Zero-Shot Transfer (`DACL10k-v3 validation`, 975 Images, All 13 Bridge Damages)")
        md_lines.append("")
        dacl_order = [
            ("comp_dinomaly", "Dinomaly (CVPR 2025)"),
            ("comp_anomalyclip", "AnomalyCLIP (ICLR 2024)"),
            ("comp_bayespfl", "Bayes-PFL (CVPR 2025)"),
            ("comp_adaptclip", "AdaptCLIP (AAAI 2026)"),
            ("comp_wpformer", "WPFormer (CVPR 2025)"),
            ("comp_mixercseg", "MixerCSeg (CVPR 2026)"),
            ("baseline", "A0 (VisualAD Baseline)"),
            ("a1_mp_bs", "A1 (+MP-BS)"),
            ("a2_wd_saf", "A2 (+MP-BS +WD-SAF)"),
            ("a3_agds_wgd", "A3 (+MP-BS +WD-SAF +AGDS-WGD)"),
            ("a3_plus_cw_8ep", "A3 + CW-Loss"),
            ("a3_plus_scorer_8ep", "A3 + Scorer"),
            ("a4_scratch8ep", "A4 (Bridge-VisualAD Full)"),
        ]
        for dir_key, d_title in [
            ("S2DS_to_DACL10k", "1. S2DS -> DACL10k-v3 (Zero-Shot)"),
            ("CODEBRIM_to_DACL10k", "2. CODEBRIM -> DACL10k-v3 (Zero-Shot)"),
        ]:
            md_lines.append(f"### {d_title}")
            md_lines.append("")
            md_lines.append(
                "| Model / Method | P-AUROC | P-F1 | P-AP | P-AUPRO | P-AUPRO(Std03) | P-IoU | **Pixel-Mean-4** | P-F1(SrcTh) | Crack | Spall | Rust | Effl | **Shared4-Rec** |"
            )
            md_lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
            for mk, ml in dacl_order:
                if mk not in dacl_data.get(dir_key, {}):
                    continue
                m = dacl_data[dir_key][mk]["all_13_damages"]
                s4 = m["Per-Class-Shared4"]
                md_lines.append(
                    f"| **{ml}** | {m['Pixel-AUROC']:.2f} | {m['Pixel-F1']:.2f} | {m['Pixel-AP']:.2f} | "
                    f"{m['Pixel-AUPRO']:.2f} | {m['Pixel-AUPRO-Std03']:.2f} | {m['Pixel-IoU']:.2f} | "
                    f"**{m['Pixel-Mean-4']:.2f}** | {m['Pixel-F1-at-Source-Th']:.2f} | "
                    f"{s4['Crack_Recall']:.2f} | {s4['Spalling_Recall']:.2f} | {s4['Corrosion_Recall']:.2f} | "
                    f"{s4['Efflorescence_Recall']:.2f} | **{s4['Shared4_Macro_Recall']:.2f}** |"
                )
            md_lines.append("")

    out_md = os.path.join(OUT_DIR, "verified_all_paper_tables.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"Successfully generated {out_md}")


if __name__ == "__main__":
    build_all_verified_tables()
