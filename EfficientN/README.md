# EfficientNet Colour-Fusion Ablations

`rmof_efficientnet.py` implements EfficientNet-B0 and a safely initialized
deep-feature residual. The active leakage-free manifest is
`split_40_10_50.csv`, which expects `Images/` beside it; that directory is
populated from the official Mendeley archive.

## Active split

The current protocol uses a deterministic, plot-disjoint 40/10/50 split:
480 training images, 120 validation images, and 600 test images (160/40/200
images per class). Regenerate it with:

```bash
python3 src/make_plot_split.py
```

## Model switches

The recommended `safe_deep_residual` preset keeps the native pretrained
EfficientNet classifier and adds a zero-initialized Stage-4 logit correction.
At initialization its logits are exactly equal to the baseline, so the added
branch never corrupts the pretrained representation before it learns.

The experiment code also supports independent `--use-stage3` / `--no-use-stage3`,
`--use-stage4`, `--use-region-tokens`, `--use-color-stats`,
`--use-color-texture`, and `--fusion` switches. Fusion modes are `concat`,
`gate`, `cross_attention`, and `region_attention`.

The colour branch derives a soft ExG foreground mask, mask-weighted HSV, Lab,
ExG, ExGR and NGRDI mean/std tokens, and shallow-CNN colour texture tokens.
The aligned attention mode adds a learnable global/2x2 region-pair bias before
attention, then uses residual connections and LayerNorm.

`--emd-weight` and `--score-weight` enable cumulative EMD ordinal loss and a
Smooth L1 score loss. CE is always present. `resnet18`, `efficientnet_b0`,
and `deit_tiny` remain available through `--model` as requested baselines.

## Checks and training

Run the module-level forward/backward check without accessing images:

```bash
python src/rmof_efficientnet.py --smoke-check
```

Run the short 5-epoch baseline:

```bash
python src/rmof_efficientnet.py --model efficientnet_b0 --preset cnn_baseline \
  --pretrained --epochs 5 --learning-rate 3e-4 --weight-decay 1e-2 \
  --label-smoothing 0.1 --dropout 0.4 --experiment-name baseline_5e \
  --csv split_40_10_50.csv
```

Run the recommended residual model. Its backbone LR is also `3e-4`, while the
new residual head uses `1e-3` so it can learn within five epochs:

```bash
python src/rmof_efficientnet.py --preset safe_deep_residual --pretrained --epochs 5 \
  --learning-rate 1e-3 --backbone-lr-scale 0.3 --weight-decay 1e-2 \
  --label-smoothing 0.1 --dropout 0.4 --experiment-name residual_deep_5e \
  --csv split_40_10_50.csv
```

The previous full RMOF curve peaks on validation at epoch 7 and then overfits.
Use this capped run for the optimized algorithm: it retains all 480 training
images, performs exactly seven epochs, and selects the epoch-7-or-earlier
checkpoint using validation data only. The schedule horizon preserves the
learning-rate trajectory of the validated epoch-7 checkpoint while avoiding
the remaining training epochs.

```bash
python3 src/rmof_efficientnet.py --preset ordinal_supervision --base-checkpoint \
  ../three_baselines_outputs/B0_Base/seed_42/best_model.pt --epochs 7 \
  --lr-schedule-horizon 25 --patience 7 --learning-rate 5e-4 \
  --weight-decay 1e-2 --label-smoothing 0.05 --dropout 0.2 \
  --augmentation mild --train-fraction 1.0 --batch-size 16 --validation-only \
  --experiment-name full_rmof_7e_mild_lr5e4_h25 \
  --output-dir optimized_algorithm_outputs_v3 \
  --csv split_40_10_50.csv
```

The `low_resource` preset remains available for separate reduced-label
experiments, but it is not part of this full-data short-epoch run.

On the current seed-42, plot-disjoint 40/10/50 split, the 10-epoch
EfficientNet-B0 baseline obtains test Macro-F1 `0.6594`. The tested residual
variants do not exceed that score; use the ablation table in
`../report/main.pdf` rather than the retired short-run results for comparison.

The complete ablation runner now defaults to five epochs. Use three seeds when
you need a report rather than a quick development comparison:

```bash
python3 src/run_ablations.py --suite report10 --pretrained --epochs 10 \
  --csv split_40_10_50.csv --output-dir report10_40_10_50_outputs
```

For a smaller, plot-balanced training subset with geometry-only leaf
augmentation, use `--train-fraction 0.7 --augmentation leaf`. Colour jitter is
deliberately omitted because leaf colour is a target signal in this task.

For a quicker smoke run, lower `--epochs` and select one suite. Outputs are
written as `outputs/<experiment>/seed_<seed>/`; the final CSV tables and
`confusion_matrix.png` / `module_increment_curve.png` are in
`outputs/summary/`.

The dataset is Galic et al., *Nitrogen deficiency in maize: annotated image
classification dataset*, Mendeley Data v1, DOI `10.17632/g7xnn2bm4g.1`
(CC BY 4.0).

## Code layout

Install the experiment dependencies into a virtual environment before running
any of the commands above:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

The current pipeline is deliberately split by responsibility:

| File | Responsibility |
|---|---|
| `src/rmof_efficientnet.py` | RMOF main branch, isolated baseline branches, training loop, checkpointing, and training CLI |
| `src/rmof_data.py` | CSV validation, plot-aware sampling, image transforms, and DataLoaders |
| `src/rmof_metrics.py` | Classification/ordinal metrics and confusion-matrix rendering |
| `src/run_ablations.py` | Reproducible multi-seed experiment suites |
| `src/summarize_results.py` | Aggregated tables, plots, and ablation interpretation |
| `src/make_plot_split.py` | Deterministic plot-disjoint split generation |
| `src/prepare_report_artifacts.py`, `src/export_report_tables.py` | Report-specific CSV and figure inputs |
| `src/probe_n75_calibration.py` | Validation-selected N75 logit-bias diagnostic |

`EfficientNet-B0.py` is the original standalone baseline and is retained only
as a historical reference. The active experiments and report outputs use the
modular RMOF-Net pipeline above.
