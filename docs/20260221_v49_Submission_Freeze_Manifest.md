# AERIS Sensors v49 Submission Freeze Manifest

> Freeze date: 2026-02-22
> Git HEAD: `b6b2e5e5e9cc2295975dca594eff6b29ba066bbf`
> Bundle path: `for_submission/submission_bundle_v49_20260221/`

---

## Review Chain Summary

| Version | Verdict | P0 | P1 | Key Action |
|---|---|---|---|---|
| v42 | Minor Revision | 0 | 5 | Full 5-role review |
| v44 | Accept (conditional) | 0 | 0 | 5 P1 CLOSED |
| v47 | Accept (conditional) | 0 | 1 | +4 figures; xref 3/24 |
| v48 | Accept (conditional) | 0 | 1 | xref 24/24; L354 misplaced |
| v49 | **Accept** | 0 | 0 | L354 fixed; all gates PASS |

---

## Bundle Contents (20 files)

### Main Files

| File | SHA256 |
|---|---|
| v49.tex | `1aa4f2f2263bfab1afbc01960dbc7b8b485605efd5a2faf6791585f9e92841a0` |
| v49.pdf | `7a084b94f3dc7ecd61ed46d66c59f3749452a8bf0e3c4a212e8ec9907ffbd525` |
| bibliography.bib | `339211b527c7f9a3095fc95747fd6aef8cd5119591f5365497228ef1458521b2` |
| README.txt | `a557f6e31392f5d4286772535281a092ab673fb66306f2e13b930187dc3a424c` |

### Definitions/

| File | SHA256 |
|---|---|
| journalnames.tex | `67e4b0534bee9c3c176ffaa6ef88fbc3e06054f9ad64eb24073b78e92b8cfa2d` |
| mdpi.bst | `ac520a91ae4a4ac22da4d75ad25417522cbc3adaca4fb77826d67d05d1593f3d` |
| mdpi.cls | `d5ef51fe52c34100ab67f7002824c1dd48d7720cabcdd8ef9490a7ec9c4d4d2d` |

### figures/ (12 files)

| File | SHA256 |
|---|---|
| fig0_aeris_workflow_20260221_s45.pdf | `57ab188b296b4d097d755ee40aa03c2ef891a85bf80d44af160ea0c1eb95e2bf` |
| fig1_env_pdr_panel_20260221_s45.pdf | `744166de2a68aa58a84e1d04e97689bb4de5b1bc23c945e62b254a59cae851f6` |
| fig2_ablation_panel_20260221_s45.pdf | `2af902c0adc494a0f49f62580efa963f23b96536e0a6ec5bcf04dd5e95003d28` |
| fig3_scalability_panel_20260221_s45.pdf | `e6dc9635e0a7b5e38c0717ea0899db1297c73b6a3898a7e16736307278ae0779` |
| fig4_tradeoff_panel_20260221_s45.pdf | `115584fb354ece93a9919598d50553312cc80ccc0e64156b0c1bce434fbd272b` |
| fig5_s11_patch_control_delta_20260221_s45.pdf | `6e4bd0569f10a5a7ec1b92ec2ff4f830f02e2561e8cbff754aac89d77b2d3a04` |
| fig6_s10_power_sensitivity_20260221_s45.pdf | `d62e04587e04ced451e31b9b7936114ec46007691224d1cc33c017a1f6218f46` |
| fig7_ns3_trend_panel_20260221_s45.pdf | `754cbabe00e396caba892ee8014ef59fc36573950f348f4ede845c262c3f4632` |
| fig8_s8_significance_heatmap_20260221_s45.pdf | `35e50dcfcb0baffe134d14b41143131fc786a87de47d47d74a370b7d68c3c0d7` |
| fig9_s9_s11_consistency_20260221_s45.pdf | `88184eda8ffb55630817c190c1030165c9bb78dddcf99ff5eaae1395ee8aaf29` |
| fig10_s10_absolute_profiles_20260221_s45.pdf | `4493fc884373b74832b437901b24f7fe2c824883f17a9c6ccf148347a98b4161` |
| fig11_s11_significance_panel_20260221_s45.pdf | `1646b8965ec2f0aee6d7e80e8b559cb28fcfcd78be8efe7d57b457699449ffc5` |

---

## Final Gate Status

- P0: 0 | P1: 0 | P2: optional only
- xref: 24/24 | forbidden: 0 | bib: 17/17 | abstract: 165 words
- Data cross-validation: S8 20/20, NS3 8/8

## Freeze Policy

- v49 is the submission-locked version
- Any further changes require v50+ on a new branch
- This manifest is the immutable record of the submitted bundle
