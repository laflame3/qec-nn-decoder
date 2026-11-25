"""E3 v2 figures (IBM idle-delay knob) from cached results. Reads only data/results/e3v2_*.csv.

    ../../.venv/bin/python -B src/plot_e3v2.py

Writes figures/e3_ibm_tau.png (4 panels) and figures/e3_ibm_transfer.png.
Style constants copied from the notebook so the figures match the others.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "data" / "results", ROOT / "figures"
SINGLE_COL, DOUBLE_COL, SIGNAL, CONTRAST = 89 / 25.4, 183 / 25.4, "#B64342", "#0F4D92"
PM_COLOR, UNIFORM_COLOR = "0.18", "0.62"
NATURE_RC = {"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
             "font.size": 7, "axes.labelsize": 7.5, "axes.titlesize": 8, "legend.fontsize": 6.5,
             "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
             "legend.frameon": False, "figure.dpi": 200, "savefig.dpi": 600, "pdf.fonttype": 42}
NN_STYLE = {"transformer": (SIGNAL, "o", "Transformer"), "mlp": (CONTRAST, "^", "MLP")}
BAR = dict(elinewidth=0.6, capsize=1.5, capthick=0.6, mew=0.7)
TAUS = [0, 5, 15]
TAU_SHADE = {0: "#E3A29F", 5: "#C9625F", 15: "#7E2726"}  # one hue, light -> dark = larger tau


def panel_label(ax, letter, title=None):
    ax.annotate(letter, (0, 1), xycoords="axes fraction", xytext=(0, 8), textcoords="offset points",
                fontweight="bold", fontsize=9)
    if title:
        ax.set_title(title, pad=8)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="0.90", lw=0.4)


def ci(r):
    return [r.ratio - r.ratio_ci_low, r.ratio_ci_high - r.ratio]


def pl_se(p, n):
    return np.sqrt(p * (1 - p) / n)


def main_figure(main, graph, size):
    with plt.rc_context(NATURE_RC):
        fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL, 5.0))
        fig.subplots_adjust(left=0.085, right=0.985, bottom=0.10, top=0.93, wspace=0.28, hspace=0.50)
        (a, b), (c, d) = axes
        x = np.array(TAUS, float)

        # a, b: P_L per initial state on the held-out time block
        for ax, subset, letter, title in ((a, "L1", "a", r"Prepared $|111\rangle$"),
                                          (b, "L0", "b", r"Prepared $|000\rangle$")):
            ref = main[(main.model == "transformer") & (main.subset == subset)].set_index("tau").loc[TAUS]
            n = ref.n_test.to_numpy()
            ax.errorbar(x, ref.pm_uniform_pm_pl, yerr=pl_se(ref.pm_uniform_pm_pl, n), color=UNIFORM_COLOR,
                        marker="s", ms=3.2, lw=0.9, ls=(0, (3, 2)), label="PM-uniform (unfitted)", **BAR)
            ax.errorbar(x, ref.pm_pl, yerr=pl_se(ref.pm_pl, n), color=PM_COLOR, marker="s", ms=3.2, lw=0.9,
                        label="PM-fitted (training blocks)", **BAR)
            for j, (kind, (col, mk, lab)) in enumerate(NN_STYLE.items()):
                r = main[(main.model == kind) & (main.subset == subset)].set_index("tau").loc[TAUS]
                ax.errorbar(x + (j - 0.5) * 0.35, r.pl, yerr=r.pl_se, color=col, marker=mk, ms=3.6, lw=0.9,
                            label=lab, **BAR)
            ax.set_xticks(TAUS)
            ax.set_xlabel(r"Idle delay per round $\tau$ ($\mu$s)")
            ax.set_ylabel(r"Logical failure probability $P_L$")
            ax.set_ylim(bottom=0)
            panel_label(ax, letter, title)

        # c: paired ratio to PM-fitted, real device vs graph-like simulation
        for j, (kind, (col, mk, lab)) in enumerate(NN_STYLE.items()):
            r = main[(main.model == kind) & (main.subset == "all")].set_index("tau").loc[TAUS]
            g = graph[graph.model == kind].set_index("tau").loc[TAUS]
            off = (j - 0.5) * 0.45
            c.errorbar(x + off - 0.25, r.ratio, yerr=ci(r), color=col, marker=mk, ms=3.8, lw=0.9,
                       label=f"{lab}, IBM data", **BAR)
            c.errorbar(x + off + 0.25, g.ratio, yerr=ci(g), color=col, marker=mk, ms=3.8, lw=0.9, ls=(0, (3, 2)),
                       mfc="white", label=f"{lab}, graph-like simulation", **BAR)
        c.axhline(1, color=PM_COLOR, lw=0.8)
        c.set_xticks(TAUS)
        c.set_xlabel(r"Idle delay per round $\tau$ ($\mu$s)")
        c.set_ylabel(r"$P_L$ ratio to PM-fitted (paired 95% CI)")
        panel_label(c, "c", "NN vs PM-fitted, both initial states")

        # d: learning curve, Transformer, ratio to PM-fitted
        for tau in TAUS:
            r = size[(size.model == "transformer") & (size.tau == tau)].sort_values("n_train")
            m = size[(size.model == "mlp") & (size.tau == tau)].sort_values("n_train")
            d.errorbar(r.n_train, r.ratio, yerr=ci(r), color=TAU_SHADE[tau], marker="o", ms=3.6, lw=0.9,
                       label=rf"$\tau$ = {tau} $\mu$s", **BAR)
            d.plot(m.n_train, m.ratio, color=TAU_SHADE[tau], marker="^", ms=3.2, lw=0.7, ls=(0, (3, 2)),
                   mfc="white")
        d.axhline(1, color=PM_COLOR, lw=0.8)
        d.set_xscale("log")
        d.set_xticks([6000, 20000, 60000], ["6,000", "20,000", "60,000"])
        d.minorticks_off()
        d.set_xlabel("Real training shots (nested subsets)")
        d.set_ylabel(r"$P_L$ ratio to PM-fitted")
        panel_label(d, "d", "Training-set size (Transformer; MLP dashed)")
        d.legend(loc="upper right")

        h, l = a.get_legend_handles_labels()
        a.legend(h, l, loc="upper left")
        c.legend(loc="upper right", ncol=1, fontsize=6)
        fig.savefig(FIG / "e3_ibm_tau.png")
        plt.close(fig)


def transfer_figure(transfer):
    modes = {"zero_shot": "Simulation\npretraining only", "scratch_60k": "Scratch,\n60,000 real shots",
             "finetune_60k": "Pretrained +\nfine-tuned"}
    with plt.rc_context(NATURE_RC):
        fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.4), sharey=True)
        fig.subplots_adjust(left=0.08, right=0.985, bottom=0.17, top=0.88, wspace=0.10)
        x = np.arange(len(modes))
        for ax, (kind, (col, mk, lab)), letter in zip(axes, NN_STYLE.items(), "ab"):
            for k, tau in enumerate(TAUS):
                r = transfer[(transfer.model == kind) & (transfer.tau == tau) & (transfer.test_domain == "real")]
                r = r.set_index("mode").loc[list(modes)]
                ax.errorbar(x + (k - 1) * 0.22, r.ratio, yerr=ci(r), color=TAU_SHADE[tau], marker=mk, ms=3.8,
                            lw=0, label=rf"$\tau$ = {tau} $\mu$s", **BAR)
            ax.axhline(1, color=PM_COLOR, lw=0.8)
            ax.set_xticks(x, list(modes.values()), fontsize=6.4)
            panel_label(ax, letter, lab)
        axes[0].set_ylabel(r"$P_L$ ratio to PM-fitted (IBM test block)")
        axes[1].legend(loc="upper right")
        fig.savefig(FIG / "e3_ibm_transfer.png")
        plt.close(fig)


if __name__ == "__main__":
    main = pd.read_csv(RES / "e3v2_main.csv")
    graph = pd.read_csv(RES / "e3v2_graph.csv")
    size = pd.read_csv(RES / "e3v2_datasize.csv")
    transfer = pd.read_csv(RES / "e3v2_transfer.csv")
    assert set(main.test_domain) == {"real"} and set(graph.test_domain) == {"graph"}
    main_figure(main, graph, size)
    transfer_figure(transfer)
    print("wrote figures/e3_ibm_tau.png, figures/e3_ibm_transfer.png")
