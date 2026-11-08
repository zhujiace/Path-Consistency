"""从已有窗口统计重绘正文 token 曲线，不重新计算或修改实验数据。"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[1]


def main():
    data = json.loads((ROOT / "data/window_snapshot.json").read_text())
    rows = data["windows"]
    x = [row["window"] for row in rows]
    assert x == list(range(1, 9)) and data["n"] == 1319
    assert rows[0]["sc_tokens"] == rows[0]["pc_tokens"]
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Liberation Serif", "DejaVu Serif"],
        "font.size": 11,
        "axes.labelsize": 12,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })
    fig, ax = plt.subplots(figsize=(4, 4))
    fig.subplots_adjust(left=0.18, right=0.82, bottom=0.17, top=0.96)
    ax.set_box_aspect(1)
    ax.set_axisbelow(True)
    ax.grid(True, color="#bcbcbc", linestyle="--", linewidth=0.65)
    ax.plot(x, [r["sc_tokens"] for r in rows], color="black", marker="s",
            markersize=4.3, markerfacecolor="white", linewidth=1.5, label="SC")
    ax.plot(x, [r["pc_tokens"] for r in rows], color="#1f77b4", marker="o",
            markersize=4.5, linewidth=1.7, label="PC")
    ax.set(xlim=(0.7, 8.3), ylim=(0, 1200), xlabel="Sampling window",
           ylabel="Generated tokens per question", xticks=x,
           yticks=list(range(0, 1201, 300)))
    ax.tick_params(axis="both", direction="out", length=4, width=1.1, pad=3)
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.4)
    # 各窗口相对同窗口 SC 的节省率，不是累计节省率或逐题比例的均值。
    saving = [100 * (1 - r["pc_tokens"] / r["sc_tokens"]) for r in rows]
    right = ax.twinx()
    right.set_box_aspect(1)
    right.plot(x, saving, color="#d47700", linestyle="--", marker="^",
               markersize=4.5, linewidth=1.5, label="Token saving")
    right.set(ylim=(0, 100), yticks=list(range(0, 101, 20)),
              ylabel="Token saving relative to SC")
    right.set_ylabel("Token saving relative to SC", rotation=270, labelpad=18)
    right.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    right.tick_params(axis="y", direction="out", length=4, width=1.1, pad=3)
    right.grid(False)
    for spine in right.spines.values():
        spine.set_visible(False)
    lines, labels = ax.get_legend_handles_labels()
    extra_lines, extra_labels = right.get_legend_handles_labels()
    legend = right.legend(lines + extra_lines, labels + extra_labels,
                       loc="lower right", frameon=True, facecolor="white",
                       edgecolor="black", framealpha=1, fancybox=False,
                       fontsize=10.5, borderpad=0.4, handlelength=1.8)
    legend.get_frame().set_linewidth(0.9)
    for extension in ("pdf", "svg", "png"):
        path = ROOT / "figs" / f"window_tokens.{extension}"
        fig.savefig(path, dpi=220, facecolor="white", bbox_inches="tight", pad_inches=0.01)
        print(path)
    print("Window token saving (%):", [round(value, 2) for value in saving])
    fig.canvas.draw()
    bounds = ax.get_window_extent()
    assert abs(bounds.width - bounds.height) < 0.01, "绘图区须为正方形"
    plt.close(fig)


if __name__ == "__main__":
    main()
