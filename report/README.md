# RMOF-Net report source

This folder contains the final report, standalone LaTeX source, figure-generation script, and every figure in both EPS and PDF form.

## Compile

From this folder:

```bash
pdflatex RMOF_Net_Report.tex
pdflatex RMOF_Net_Report.tex
```

The LaTeX source includes the references directly, so BibTeX is not required. The PDF versions in `figures/` were converted from their corresponding EPS files and are the files included by LaTeX.

To regenerate the plots, run `python3 make_figures.py`, then convert each `figures/*.eps` file with `ps2pdf -dEPSCrop`. The supplied architecture and case-study assets are in `assets/`.

## Layout

The report has four body pages followed by a references page. Only the architecture overview and the main comparison table use the full two-column width. Figures 2--5 are independent single-column figures, each placed beside the corresponding result discussion.

The revised method section documents the complete RMOF-Net forward pass: aligned global/$2\times2$ deep tokens, explicit ExG/ExGR/NGRDI definitions, soft leaf masking, regional colour-statistic and texture tokens, region-biased cross-attention, the final global/region aggregation, dual prediction heads, and the fully defined cross-entropy, cumulative EMD, and SmoothL1 objective.

Figures 2--5 are true vector EPS files with Times-Roman/Times-Bold PostScript fonts and are then converted to PDF for inclusion. Figure 1 is also delivered as EPS, but its supplied source is a PNG architecture diagram; its EPS necessarily carries that supplied raster artwork rather than recreating unavailable vector source text.

## Metric reconciliation and Figure 3

The Base and Full rows in Table I are now computed from the same two 200-image confusion matrices that appear in Figure 3. This makes accuracy, binary deficiency F1, all class-wise F1 values, Macro-F1, ordinal MAE, endpoint-reversal counts, and residual histograms mutually consistent.

- The Full RMOF-Net `N0 -> N75` count is 20, rather than 21, so its matrix totals 200.
- EfficientNet-B0 residual counts are `12, 31, 117, 28, 12` for residuals `-2, -1, 0, +1, +2`.
- Full RMOF-Net residual counts are `2, 22, 143, 30, 3`.

The hard-label ordinal MAE is calculated as the mean absolute class residual. Accordingly, Base has Def.-F1/Acc./Macro-F1/Ord.-MAE of `0.788/0.585/0.585/0.535`, and Full has `0.890/0.715/0.715/0.310`. The remaining point-estimate component rows retain their supplied class-wise F1 values, while their Macro-F1 values are recomputed as the arithmetic mean of the three displayed class-wise F1 values. Figures 2 and 5 use these reconciled Table-I values.
