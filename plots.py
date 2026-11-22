"""Figures from saved validation outputs (no model or images needed).

    python plots.py exits --series CNN=results/cnn_val_outputs.npz [--series MLP=...] [--logistic ...]
        Validation macro-F1 and accuracy at every exit (the step-8 gate figure).
    python plots.py confidence results/cnn_val_baselines.json
        Macro-F1 against compute for the fixed exits and confidence-only stopping.

Each figure is saved as a PNG with a CSV of the plotted numbers next to it.
Outputs from the test split are refused unless --final-test is given.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from exit_outputs import exit_metrics, load_exit_outputs  # noqa: E402

# Reference palette (light mode): categorical slots 1-2, ink and chrome.
SERIES = ["#2a78d6", "#eb6834"]
SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BUDGETS = (0.4, 0.6, 0.8)


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelcolor=INK_2, labelsize=9, length=0)


def new_figure(width, height):
    plt.rcParams.update({"font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]})
    fig = plt.figure(figsize=(width, height), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    return fig


def line(ax, x, y, color, label):
    ax.plot(x, y, color=color, linewidth=2, label=label, solid_capstyle="round", zorder=3)
    ax.plot(x, y, linestyle="none", marker="o", markersize=5.5, markerfacecolor=color,
            markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=4)


def load_validation(path: Path, final_test: bool):
    outputs = load_exit_outputs(path)
    if outputs.metadata.get("split") != "val" and not final_test:
        raise SystemExit(f"{path} is not from the validation split; add --final-test only for the final evaluation")
    return outputs


def plot_exits(args):
    series = []
    for item in args.series:
        name, _, path = item.partition("=")
        if not path:
            raise SystemExit("--series must look like NAME=path/to/outputs.npz")
        outputs = load_validation(Path(path), args.final_test)
        series.append((name, outputs, exit_metrics(outputs.logits, outputs.labels)))
    if len(series) > len(SERIES):
        raise SystemExit(f"at most {len(SERIES)} series")
    logistic = None
    if args.logistic:
        lr = load_validation(args.logistic, args.final_test)
        logistic = exit_metrics(lr.logits, lr.labels)

    fig = new_figure(9.2, 3.8)
    axes = fig.subplots(1, 2, sharey=True)
    values = [v for _, _, m in series for key in ("exit_macro_f1", "exit_accuracy") for v in m[key]]
    if logistic:
        values += logistic["exit_macro_f1"] + logistic["exit_accuracy"]
    low, high = min(values), max(values)
    pad = (high - low) * 0.08
    for ax, key, title in zip(axes, ("exit_macro_f1", "exit_accuracy"), ("Macro-F1 by exit", "Accuracy by exit")):
        style_axes(ax)
        for (name, outputs, metrics), color in zip(series, SERIES):
            exits = range(1, outputs.num_exits + 1)
            line(ax, list(exits), metrics[key], color, name)
            ax.annotate(name, (outputs.num_exits, metrics[key][-1]), xytext=(8, 0), textcoords="offset points",
                        va="center", fontsize=9, color=INK_2)
        if logistic:
            value = logistic[key][0]
            ax.axhline(value, color=MUTED, linewidth=1, zorder=2)
            ax.annotate(f"Logistic regression {value:.3f}", (10, value), xytext=(0, -4), textcoords="offset points",
                        ha="right", va="top", fontsize=8.5, color=MUTED)
        ax.set_title(title, loc="left", fontsize=11, color=INK, pad=8)
        ax.set_xlabel("Exit", fontsize=9, color=INK_2)
        ax.set_xticks(range(1, 11))
        ax.set_xlim(0.6, 11.6)
        ax.set_ylim(low - pad, high + pad)
    axes[0].set_ylabel("Validation score", fontsize=9, color=INK_2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=len(series), frameon=False, fontsize=9,
               labelcolor=INK_2, bbox_to_anchor=(0.99, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    out = args.out or Path("results/figures/exits_val.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    with out.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["series", "exit", "compute_share", "macro_f1", "accuracy"])
        for name, outputs, metrics in series:
            shares = outputs.compute_profile().fractions(all_heads=False)
            for k in range(outputs.num_exits):
                writer.writerow([name, k + 1, f"{shares[k]:.4f}", f"{metrics['exit_macro_f1'][k]:.4f}",
                                 f"{metrics['exit_accuracy'][k]:.4f}"])
        if logistic:
            writer.writerow(["logistic regression", 1, "", f"{logistic['exit_macro_f1'][0]:.4f}",
                             f"{logistic['exit_accuracy'][0]:.4f}"])
    print("Saved:", out, "and", out.with_suffix(".csv"))


def plot_confidence(args):
    result = json.loads(args.baselines.read_text())
    if result["outputs_metadata"].get("split") != "val" and not args.final_test:
        raise SystemExit("these results are not from the validation split; add --final-test only for the final evaluation")
    fixed, sweep = result["fixed_exits"], result["confidence_sweep"]
    fig = new_figure(6.4, 4.2)
    ax = fig.subplots()
    style_axes(ax)
    for budget in BUDGETS:
        ax.axvline(100 * budget, color=AXIS, linewidth=0.8, zorder=1)
        ax.annotate(f"{budget:.0%} budget", (100 * budget, 0), xycoords=("data", "axes fraction"),
                    xytext=(3, 4), textcoords="offset points", va="bottom", fontsize=8, color=MUTED)
    line(ax, [100 * r["compute"] for r in fixed], [r["macro_f1"] for r in fixed], SERIES[0], "Fixed exit")
    for r in fixed:
        if r["exit"] in (1, 2, 4, 10):
            ax.annotate(f"exit {r['exit']}", (100 * r["compute"], r["macro_f1"]), xytext=(6, -10),
                        textcoords="offset points", fontsize=8, color=INK_2)
    ax.plot([100 * r["compute"] for r in sweep], [r["macro_f1"] for r in sweep], color=SERIES[1], linewidth=2,
            label="Confidence only (from exit 2)", zorder=3)
    if "logistic_regression" in result:
        lr = result["logistic_regression"]
        ax.plot([100 * lr["compute"]], [lr["macro_f1"]], marker="o", markersize=5.5, color=MUTED,
                markeredgecolor=SURFACE, markeredgewidth=1.5, linestyle="none", zorder=4)
        ax.annotate(f"logistic regression {lr['macro_f1']:.3f}", (100 * lr["compute"], lr["macro_f1"]),
                    xytext=(6, 4), textcoords="offset points", fontsize=8, color=MUTED)
    ax.set_xlim(0, 103)
    ax.set_xlabel("Compute (% of the full network's multiply-adds)", fontsize=9, color=INK_2)
    ax.set_ylabel("Validation macro-F1", fontsize=9, color=INK_2)
    ax.set_title("Macro-F1 against compute", loc="left", fontsize=11, color=INK, pad=8)
    legend = ax.legend(loc="center right", frameon=True, fontsize=9, labelcolor=INK_2,
                       facecolor=SURFACE, edgecolor="none", framealpha=1)
    legend.set_zorder(5)
    fig.tight_layout()

    out = args.out or args.baselines.with_name(args.baselines.stem + "_confidence.png")
    if args.out is None:
        out = Path("results/figures") / out.name
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    with out.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rule", "setting", "compute_share", "macro_f1", "accuracy"])
        for r in fixed:
            writer.writerow(["fixed exit", r["exit"], f"{r['compute']:.4f}", f"{r['macro_f1']:.4f}", f"{r['accuracy']:.4f}"])
        for r in sweep:
            writer.writerow(["confidence only", r["threshold"], f"{r['compute']:.4f}", f"{r['macro_f1']:.4f}",
                             f"{r['accuracy']:.4f}"])
    print("Saved:", out, "and", out.with_suffix(".csv"))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = p.add_subparsers(dest="command", required=True)
    exits = commands.add_parser("exits", help="Validation macro-F1 and accuracy at every exit")
    exits.add_argument("--series", action="append", required=True, help="NAME=outputs.npz (up to two)")
    exits.add_argument("--logistic", type=Path, help="Logistic-regression outputs, drawn as a reference line")
    exits.add_argument("--out", type=Path)
    exits.add_argument("--final-test", action="store_true")
    confidence = commands.add_parser("confidence", help="Macro-F1 against compute from sweep.py results")
    confidence.add_argument("baselines", type=Path, help="*_baselines.json from sweep.py")
    confidence.add_argument("--out", type=Path)
    confidence.add_argument("--final-test", action="store_true")
    args = p.parse_args(argv)
    plot_exits(args) if args.command == "exits" else plot_confidence(args)


if __name__ == "__main__":
    main()
