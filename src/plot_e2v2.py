"""E2 v2 figure (Sycamore r05, multi-sample training pool) from cached results.

    ../../.venv/bin/python -B src/plot_e2v2.py

Reads data/results/e2v2_datasize.csv, e2v2_transfer.csv, e2_baselines.csv.
Writes figures/e2_sycamore_pool.png. Style constants match the notebook.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "data" / "results", ROOT / "figures"
DOUBLE_COL, SIGNAL, CONTRAST = 183 / 25.4, "#B64342", "#0F4D92"
PM_COLOR, REF_COLOR = "0.18", "0.55"
NATURE_RC = {"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
             "font.size": 7, "axes.labelsize": 7.5, "axes.titlesize": 8, "legend.fontsize": 6.5,
             "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
             "legend.frameon": False, "figure.dpi": 200, "savefig.dpi": 600, "pdf.fonttype": 42}
NN_STYLE = {"transformer": (SIGNAL, "o", "Transformer"), "mlp": (CONTRAST, "^", "MLP")}
BAR = dict(elinewidth=0.6, capsize=1.5, capthick=0.6, mew=0.7)
METHODS = {  # (method key in e2v2_transfer.csv, tick label)
    "legacy_pre0_s0": "Simulated\n(simple.dem)",
    "legacy_ft_s0": "Simulated\n+ fine-tune",
    "scratch_45k_sample14": "Scratch,\n45k real",
    "pretrain_zero_shot": "Other 52 runs\n(3.9M real)",
    "finetune_45k": "Other 52 runs\n+ fine-tune",
}


def panel_label(ax, letter, title=None):
    ax.annotate(letter, (0, 1), xycoords="axes fraction", xytext=(0, 8), textcoords="offset points",
                fontweight="bold", fontsize=9)
    if title:
        ax.set_title(title, pad=8)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="0.90", lw=0.4)


def ci(r):
    return [r.ratio - r.ratio_ci_low, r.ratio_ci_high - r.ratio]


def main():
    size = pd.read_csv(RES / "e2v2_datasize.csv")
    transfer = pd.read_csv(RES / "e2v2_transfer.csv")
    base = pd.read_csv(RES / "e2_baselines.csv").set_index("name")
    pm = float(base.loc["PM-corr+simple", "ler_test"])
    google = float(base.loc["Google harmony+correlations", "ler_test"]) / pm  # validation-selected reference

    with plt.rc_context(NATURE_RC):
        fig, (a, b) = plt.subplots(1, 2, figsize=(DOUBLE_COL, 3.0), gridspec_kw={"width_ratios": [1, 1.35]})
        fig.subplots_adjust(left=0.075, right=0.99, bottom=0.27, top=0.88, wspace=0.22)

        for ax in (a, b):
            ax.axhline(1, color=PM_COLOR, lw=0.8, label="PM-correlated + simple.dem")
            ax.axhline(google, color=REF_COLOR, lw=0.8, ls=(0, (1, 1.5)),
                       label="Reference: Google harmony, DEM fitted on these shots")

        for j, (kind, (col, mk, lab)) in enumerate(NN_STYLE.items()):
            r = size[size.model == kind].sort_values("n_train_unique")
            a.errorbar(r.n_train_unique * (1 + 0.04 * (j - 0.5)), r.ratio, yerr=ci(r), color=col, marker=mk,
                       ms=3.6, lw=0.9, label=lab, **BAR)
        a.set_xscale("log")
        a.set_xticks([4.5e4, 1.5e5, 5e5, 1e6, 2e6, 3.945e6], ["45k", "150k", "500k", "1M", "2M", "3.9M"])
        a.minorticks_off()
        a.set_xlabel("Real training shots (nested, 53 runs pooled)")
        a.set_ylabel(r"$P_L$ ratio to PM-correlated (paired 95% CI)")
        panel_label(a, "a", "Training-set size, scratch")

        x = np.arange(len(METHODS))
        for j, (kind, (col, mk, lab)) in enumerate(NN_STYLE.items()):
            r = transfer[transfer.model == kind].set_index("method").loc[list(METHODS)]
            b.errorbar(x + (j - 0.5) * 0.24, r.ratio, yerr=ci(r), color=col, marker=mk, ms=3.8, lw=0,
                       label=lab, **BAR)
        b.axvspan(-0.5, 1.5, color="0.95", zorder=0)
        b.text(0.5, 1.15, "Pretraining on\nsimulated shots", ha="center", va="top", fontsize=6.2, color="0.35")
        b.text(3.5, 1.15, "Pretraining on\nreal shots", ha="center", va="top", fontsize=6.2, color="0.35")
        b.set_xlim(-0.5, len(METHODS) - 0.5)
        b.set_xticks(x, list(METHODS.values()), fontsize=6.4)
        b.set_ylim(top=1.48)
        panel_label(b, "b", "Pretraining prior and fine-tuning (test: run 14)")

        handles, labels = a.get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=4)
        fig.savefig(FIG / "e2_sycamore_pool.png")
        plt.close(fig)
    print("wrote figures/e2_sycamore_pool.png; google ref ratio", round(google, 4))


if __name__ == "__main__":
    main()
