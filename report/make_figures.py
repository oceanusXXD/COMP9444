from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image


ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update(
    {
        # Use the same Times-compatible family as the LaTeX document.  All
        # labels below are plain text rather than Matplotlib/STIX mathtext so
        # the EPS files contain one consistent vector font family.
        "font.family": "Times",
        "font.serif": ["Times"],
        "font.size": 8.2,
        "axes.titlesize": 8.7,
        "axes.labelsize": 8.2,
        "xtick.labelsize": 7.3,
        "ytick.labelsize": 7.3,
        "legend.fontsize": 7.2,
        "axes.linewidth": 0.65,
        "xtick.major.width": 0.55,
        "ytick.major.width": 0.55,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.025,
        "ps.useafm": True,
        "ps.fonttype": 42,
        "pdf.fonttype": 42,
    }
)

BLUE = "#2F67B1"
GREEN = "#3F8A54"
AMBER = "#C77C19"
RED = "#B84A4A"
GRAY = "#646B73"
LIGHT_GRID = "#D8DDE3"


def save_eps(fig, name):
    fig.savefig(FIG / f"{name}.eps", format="eps", dpi=600, facecolor="white")
    plt.close(fig)


def figure2():
    labels = ["Base", "+MS", "+ECP", "+LCT", "+Gate", "Full"]
    x = np.arange(len(labels))
    # Macro-F1 is the arithmetic mean of the three class-wise F1 values in
    # Table I. The Base and Full rows are additionally derived from the
    # 200-image confusion matrices used in Figure 3.
    macro = np.array([0.585, 0.606, 0.594, 0.660, 0.680, 0.715])
    n75 = np.array([0.563, 0.410, 0.395, 0.485, 0.510, 0.618])
    mae = np.array([0.535, 0.420, 0.435, 0.380, 0.365, 0.310])

    # The two panels are vertically stacked so this figure remains readable
    # when placed in one paper column.
    fig, axes = plt.subplots(2, 1, figsize=(3.20, 2.30), gridspec_kw={"hspace": 0.44})
    ax = axes[0]
    ax.plot(x, macro, color=BLUE, marker="o", lw=1.55, ms=4.2, label="Macro-F1")
    ax.plot(x, n75, color=GREEN, marker="s", lw=1.55, ms=4.0, label="F1-N75")
    for xi, yi, text, color, offset in [
        (0, macro[0], "0.585", BLUE, 0.013),
        (5, macro[-1], "0.715", BLUE, 0.013),
        (0, n75[0], "0.563", GREEN, -0.030),
        (5, n75[-1], "0.618", GREEN, -0.030),
    ]:
        ax.text(xi, yi + offset, text, ha="center", va="center", color=color, fontsize=6.5)
    ax.set_ylim(0.35, 0.76)
    ax.set_ylabel("F1 score")
    ax.set_xticks(x, labels)
    ax.grid(axis="y", color=LIGHT_GRID, lw=0.55)
    ax.legend(loc="upper left", frameon=False, ncol=2, handlelength=1.45, columnspacing=0.9)
    ax.set_title("(a) Classification quality")

    ax = axes[1]
    ax.plot(x, mae, color=AMBER, marker="D", lw=1.65, ms=4.0)
    ax.text(0, mae[0] + 0.012, "0.535", ha="center", color=AMBER, fontsize=6.5)
    ax.text(5, mae[-1] + 0.012, "0.310", ha="center", color=AMBER, fontsize=6.5)
    ax.set_ylim(0.28, 0.57)
    ax.set_ylabel("Ordinal MAE")
    ax.set_xticks(x, labels)
    ax.grid(axis="y", color=LIGHT_GRID, lw=0.55)
    ax.set_title("(b) Ordered-treatment error (lower is better)")
    save_eps(fig, "figure2_component_ablation")


def _heatmap(ax, matrix, title):
    cmap = LinearSegmentedColormap.from_list("paper_blues", ["#F5F8FC", "#A8C5E8", BLUE])
    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=55, interpolation="nearest", aspect="equal")
    for i in range(3):
        for j in range(3):
            value = int(matrix[i, j])
            color = "white" if value >= 35 else "#222222"
            ax.text(j, i, str(value), ha="center", va="center", fontsize=8.0, fontweight="bold" if i == j else "normal", color=color)
    ax.set_xticks(range(3), ["N0", "N75", "NFull"])
    ax.set_yticks(range(3), ["N0", "N75", "NFull"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    ax.tick_params(length=0)
    return im


def figure3():
    # The paired Base/Full Table-I rows are computed from these 200-image
    # confusion matrices; every residual count is derived from its matrix.
    base = np.array([[42, 16, 12], [10, 38, 12], [12, 21, 37]])
    full = np.array([[47, 20, 3], [8, 42, 10], [2, 14, 54]])
    residuals = np.arange(-2, 3)
    base_res = np.array([12, 31, 117, 28, 12])
    full_res = np.array([2, 22, 143, 30, 3])

    # Compact one-column diagnostic: the matrices retain directional detail
    # while the residual panel summarises ordered-error severity.
    fig = plt.figure(figsize=(3.20, 2.95))
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.28, 0.72],
        hspace=0.76,
        wspace=0.48,
    )
    _heatmap(fig.add_subplot(gs[0, 0]), base, "(a) Base")
    _heatmap(fig.add_subplot(gs[0, 1]), full, "(b) Full")

    ax = fig.add_subplot(gs[1, :])
    width = 0.34
    ax.bar(residuals - width / 2, base_res, width, color="#AAB2BC", edgecolor="#606873", lw=0.45, label="Base")
    ax.bar(residuals + width / 2, full_res, width, color=GREEN, edgecolor="#27643A", lw=0.45, label="Full")
    for xs, ys in [(residuals - width / 2, base_res), (residuals + width / 2, full_res)]:
        for xi, yi in zip(xs, ys):
            ax.text(xi, yi + 3.2, str(int(yi)), ha="center", va="bottom", fontsize=5.8)
    ax.set_xticks(residuals, ["-2", "-1", "0", "+1", "+2"])
    ax.set_ylim(0, 160)
    ax.set_xlabel("Class residual (prediction - target)")
    ax.set_ylabel("Images")
    ax.grid(axis="y", color=LIGHT_GRID, lw=0.55)
    ax.legend(frameon=False, loc="upper left", ncol=2, handlelength=1.2, columnspacing=0.9)
    ax.set_title("(c) Hard-label residuals")
    save_eps(fig, "figure3_error_analysis")


def figure4():
    img = Image.open(ROOT / "assets" / "classic_case.png").convert("RGB")
    fig = plt.figure(figsize=(3.20, 1.62))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.78, 1.22], wspace=0.12)
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(img)
    ax.axis("off")
    ax = fig.add_subplot(gs[0, 1])
    ax.axis("off")
    rows = [
        ("Ground truth", "N75", "#222222"),
        ("EfficientNet-B0", "NFull", RED),
        ("Full RMOF-Net", "N75", GREEN),
    ]
    ypos = [0.76, 0.50, 0.24]
    for (model, pred, color), y in zip(rows, ypos):
        ax.text(0.03, y + 0.065, model, ha="left", va="center", fontsize=7.5, color="#333333")
        ax.text(0.97, y - 0.045, pred, ha="right", va="center", fontsize=10.0, color=color, fontweight="bold")
        ax.plot([0.03, 0.97], [y - 0.115, y - 0.115], color=LIGHT_GRID, lw=0.65)
    save_eps(fig, "figure4_qualitative_case")


def figure5():
    methods = ["ResNet18", "DeiT-T", "B0", "+MS", "+ECP", "+LCT", "+Gate", "Full"]
    macro = np.array([0.588, 0.435, 0.585, 0.606, 0.594, 0.660, 0.680, 0.715])
    params = np.array([11.178, 5.525, 4.011, 4.069, 4.073, 4.182, 4.214, 4.314])
    latency = np.array([0.690, 0.795, 0.555, 0.565, 0.994, 1.137, 1.137, 1.143])
    norm = mpl.colors.Normalize(vmin=0.50, vmax=1.20)
    cmap = LinearSegmentedColormap.from_list("latency", [BLUE, "#E0AD4B", RED])
    sizes = 28 + 42 * (latency - latency.min()) / (latency.max() - latency.min())

    fig, ax = plt.subplots(figsize=(3.20, 2.20))
    ax.scatter(params, macro, c=latency, s=sizes, cmap=cmap, norm=norm, edgecolor="white", linewidth=0.55, zorder=3)
    for i in [0, 1]:
        offset = {0: (-34, 5), 1: (4, -11)}[i]
        ax.annotate(methods[i], (params[i], macro[i]), xytext=offset, textcoords="offset points", fontsize=6.4, color="#30343A")
    ax.set_xlim(3.55, 11.65)
    ax.set_ylim(0.40, 0.75)
    ax.set_xlabel("Parameters (M)")
    ax.set_ylabel("Macro-F1")
    ax.grid(color=LIGHT_GRID, lw=0.55)

    inset = ax.inset_axes([0.10, 0.43, 0.52, 0.48])
    ids = np.arange(2, 8)
    inset.scatter(params[ids], macro[ids], c=latency[ids], s=sizes[ids] * 0.78, cmap=cmap, norm=norm, edgecolor="white", linewidth=0.45, zorder=3)
    inset.set_xlim(3.99, 4.34)
    inset.set_ylim(0.570, 0.730)
    offsets = [(2, -10), (-12, 5), (2, -11), (2, -10), (-14, 5), (-18, -10)]
    for i, off in zip(ids, offsets):
        inset.annotate(methods[i], (params[i], macro[i]), xytext=off, textcoords="offset points", fontsize=5.6)
    inset.grid(color=LIGHT_GRID, lw=0.42)
    inset.tick_params(labelsize=5.5, length=2)

    cax = ax.inset_axes([0.73, 0.10, 0.055, 0.33])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.set_label("Latency (ms)", fontsize=6.0, labelpad=1.0)
    cb.ax.tick_params(labelsize=5.3, length=2)
    save_eps(fig, "figure5_performance_cost")


def raster_to_eps():
    overview = Image.open(ROOT / "assets" / "rmof_overview.png").convert("RGB")
    overview.save(FIG / "figure1_rmof_overview.eps", format="EPS")


if __name__ == "__main__":
    raster_to_eps()
    figure2()
    figure3()
    figure4()
    figure5()
