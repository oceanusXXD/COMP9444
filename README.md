# RMOF-Net Report

The IEEE double-column paper is in [`report/`](report/).

To compile the report:

```bash
cd report
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

The compiled paper is [`report/main.pdf`](report/main.pdf). The supplied architecture image is kept unchanged at [`report/figures/rmof_overview.png`](report/figures/rmof_overview.png).
