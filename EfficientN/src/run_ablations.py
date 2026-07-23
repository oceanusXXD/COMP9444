"""Run the required ablations for three seeds, then create summary tables."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SUITES = {
    "report10": [
        ("efficientnet_b0_10e", "efficientnet_b0", "cnn_baseline"),
        ("stage4_residual_10e", "rmof_efficientnet", "safe_deep_residual"),
        ("multiscale_regions_10e", "rmof_efficientnet", "regions"),
        ("colour_stats_10e", "rmof_efficientnet", "color_stats"),
        ("colour_texture_10e", "rmof_efficientnet", "color_texture"),
        ("scalar_gate_10e", "rmof_efficientnet", "fusion_gate"),
        ("full_rmof_10e", "rmof_efficientnet", "ordinal_supervision"),
    ],
    "baseline": [
        ("simple_cnn", "simple_cnn", "cnn_baseline"),
        ("efficientnet_b0", "efficientnet_b0", "cnn_baseline"),
        ("resnet18", "resnet18", "cnn_baseline"),
        ("deit_tiny", "deit_tiny", "cnn_baseline"),
    ],
    "main": [
        ("cnn_baseline", "efficientnet_b0", "cnn_baseline"),
        ("multiscale", "rmof_efficientnet", "multiscale"),
        ("regions", "rmof_efficientnet", "regions"),
        ("color_stats", "rmof_efficientnet", "color_stats"),
        ("color_texture", "rmof_efficientnet", "color_texture"),
        ("region_cross_attention", "rmof_efficientnet", "region_cross_attention"),
        ("ordinal_supervision", "rmof_efficientnet", "ordinal_supervision"),
    ],
    "fusion": [
        ("fusion_concat", "rmof_efficientnet", "fusion_concat"),
        ("fusion_gate", "rmof_efficientnet", "fusion_gate"),
        ("fusion_cross_attention", "rmof_efficientnet", "fusion_cross_attention"),
        ("fusion_region_attention", "rmof_efficientnet", "fusion_region_attention"),
    ],
    "loss": [
        ("loss_ce", "rmof_efficientnet", "loss_ce"),
        ("loss_ce_emd", "rmof_efficientnet", "loss_ce_emd"),
        ("loss_ce_score", "rmof_efficientnet", "loss_ce_score"),
        ("loss_ce_emd_score", "rmof_efficientnet", "loss_ce_emd_score"),
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=["report10", "baseline", "main", "fusion", "loss", "all"], default="all")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 52, 62])
    parser.add_argument("--csv", default="split_40_10_50.csv")
    parser.add_argument("--data-root", default=".")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--augmentation", choices=["mild", "leaf", "strong"], default="leaf")
    parser.add_argument("--train-fraction", type=float, default=1.0)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cache-images", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_suites = [name for name in SUITES if name != "report10"] if args.suite == "all" else [args.suite]
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    def project_path(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else project_root / path

    train_script = script_dir / "rmof_efficientnet.py"
    csv_path = project_path(args.csv)
    data_root = project_path(args.data_root)
    output_dir = project_path(args.output_dir)
    jobs = [job for suite in selected_suites for job in SUITES[suite]]
    for experiment, model, preset in jobs:
        for seed in args.seeds:
            command = [
                sys.executable,
                str(train_script),
                "--preset",
                preset,
                "--model",
                model,
                "--experiment-name",
                experiment,
                "--seed",
                str(seed),
                "--csv",
                str(csv_path),
                "--data-root",
                str(data_root),
                "--output-dir",
                str(output_dir),
                "--epochs",
                str(args.epochs),
                "--patience",
                str(args.patience),
                "--batch-size",
                str(args.batch_size),
                "--num-workers",
                str(args.num_workers),
                "--augmentation",
                args.augmentation,
                "--train-fraction",
                str(args.train_fraction),
            ]
            if args.suite == "report10":
                command.extend((
                    "--weight-decay", "1e-2",
                    "--label-smoothing", "0.1",
                    "--dropout", "0.4",
                ))
                if experiment == "full_rmof_10e":
                    command.extend(("--emd-weight", "0.25", "--score-weight", "0.25"))
                else:
                    command.extend(("--emd-weight", "0", "--score-weight", "0"))
                if model == "rmof_efficientnet":
                    command.extend(("--learning-rate", "1e-3", "--backbone-lr-scale", "0.3"))
                else:
                    command.extend(("--learning-rate", "3e-4"))
            if args.pretrained:
                command.append("--pretrained")
            if args.amp:
                command.append("--amp")
            else:
                command.append("--no-amp")
            if args.cache_images:
                command.append("--cache-images")
            else:
                command.append("--no-cache-images")
            if args.device:
                command.extend(("--device", args.device))
            print("running:", " ".join(command))
            subprocess.run(command, check=True, cwd=project_root)
    subprocess.run(
        [sys.executable, str(script_dir / "summarize_results.py"), "--source", str(output_dir)],
        check=True,
        cwd=project_root,
    )


if __name__ == "__main__":
    main()
