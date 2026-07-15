"""
Figure 1 — Drift characterization
三个数据集的漂移速率 σ(t) + KLL checkpoint 位置标记

布局：1 行 × 3 列，每列一个数据集
输出：fig1_drift.pdf  (双栏宽 7.0in × 1.6in)
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from fig_utils import (load_dataset, set_paper_style,
                        COLORS, SINGLE_COL, DOUBLE_COL)

# ─────────────────────────────────────────────────────────────
# 1. 核心计算：滑动窗口分位数漂移速率
# ─────────────────────────────────────────────────────────────
def compute_drift(data: np.ndarray,
                  phi: float = 0.5,
                  window_frac: float = 0.005,   # 窗口 = 0.5% of N
                  step_frac: float  = 0.001,    # 步长 = 0.1% of N
                  normalize: bool   = True) -> tuple:
    """
    返回 (t_array, sigma_array)
    
    sigma(t) = |q_phi(data[t..t+w]) - q_phi(data[t-w..t])| / w
    
    normalize=True 时除以数据范围，使三个数据集可以叠在同一 y 轴比较。
    实际论文里三子图各自独立 y 轴时设 normalize=False。
    """
    N = len(data)
    w = max(int(window_frac * N), 500)
    step = max(int(step_frac * N), 100)
    data_range = np.percentile(data, 99) - np.percentile(data, 1) + 1e-9

    ts, sigmas = [], []
    for t in range(w, N - w, step):
        q_before = np.quantile(data[t - w: t],     phi)
        q_after  = np.quantile(data[t:     t + w], phi)
        sigma = abs(q_after - q_before) / w
        if normalize:
            sigma /= data_range
        ts.append(t)
        sigmas.append(sigma)
    return np.array(ts), np.array(sigmas)

def kll_structural_cps(N: int, k: int = 256, n_min: int = 1000) -> list:
    """
    几何序列近似 KLL 层跃升触发点
    与 scheduler.py 的 generate_kll_cps 结果一致
    """
    cps = []
    level = 1
    while True:
        n = int(k * (2 ** level))
        if n > N:
            break
        if n >= n_min:
            cps.append(n)
        level += 1
    if not cps or cps[-1] != N:
        cps.append(N)
    return sorted(set(cps))

# ─────────────────────────────────────────────────────────────
# 2. 画图
# ─────────────────────────────────────────────────────────────
def plot_fig1(out_path: str = 'fig1_drift.pdf'):
    set_paper_style()

    datasets = [
        ('mimic', 'MIMIC-IV HR',        'bpm',   200_000),
        ('power', 'UCI Household Power', 'kW',    200_000),
        ('taxi',  'NYC Taxi',            'miles', 200_000),
    ]
    phis  = [0.25, 0.50, 0.90]            # 三条漂移曲线
    alpha = [0.55, 1.0, 0.55]             # 中位数最突出
    lws   = [0.9,  1.5, 0.9]
    labels= ['φ=0.25', 'φ=0.50', 'φ=0.90']

    fig, axes = plt.subplots(
        1, 3,
        figsize=(DOUBLE_COL, 1.65),
        sharey=False,
    )
    fig.subplots_adjust(wspace=0.38)

    phi_colors = ['#4dac26', '#2166ac', '#d6604d']

    for ax, (dname, title, unit, max_n) in zip(axes, datasets):
        data = load_dataset(dname, max_n=max_n)
        N    = len(data)
        cps  = kll_structural_cps(N, k=256)

        # ── 漂移曲线（三个分位数） ────────────────
        for phi, col, alp, lw, lab in zip(phis, phi_colors, alpha, lws, labels):
            ts, sigmas = compute_drift(data, phi=phi, normalize=False)
            # 平滑：rolling mean，窗口 = 5 个点
            pad = 2
            smooth = np.convolve(sigmas,
                                 np.ones(2*pad+1)/(2*pad+1),
                                 mode='same')
            ax.plot(ts / N, smooth,
                    color=col, alpha=alp, linewidth=lw, label=lab)

        # ── KLL checkpoint 标记（底部竖线） ───────
        for cp in cps[:-1]:   # 去掉终点 N，避免右边缘拥挤
            ax.axvline(cp / N,
                       ymin=0, ymax=0.12,      # 只画底部 12% 高度
                       color='#555555',
                       linewidth=0.7,
                       alpha=0.7)

        # ── 轴标签和标题 ──────────────────────────
        ax.set_xlabel('Stream position (fraction of $n$)', labelpad=3)
        if ax is axes[0]:
            ax.set_ylabel(r'$\sigma(t)$ — drift rate', labelpad=3)
        ax.set_title(f'({chr(ord("a") + datasets.index((dname, title, unit, max_n)))})'
                     f'  {title}', loc='left', pad=3)
        ax.set_xlim(0, 1)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(4, prune='upper'))
        ax.grid(axis='y')

        # ── 右上角标注数据集统计 ──────────────────
        n_label = f'$n={max_n//1000}$K'
        ax.text(0.97, 0.95, n_label,
                transform=ax.transAxes,
                ha='right', va='top',
                fontsize=7, color='#555555')

    # ── 全局图例（φ 三条线 + KLL checkpoint 符号） ───
    leg_lines = [
        Line2D([0],[0], color=c, linewidth=1.2, label=l)
        for c, l in zip(phi_colors, labels)
    ]
    leg_lines.append(
        Line2D([0],[0], color='#555555', linewidth=0.8,
               linestyle='-', label='KLL checkpoints')
    )
    axes[1].legend(handles=leg_lines,
                   ncol=4, loc='upper center',
                   bbox_to_anchor=(0.5, -0.38),
                   frameon=True, columnspacing=1.0,
                   handlelength=1.5)

    plt.savefig(out_path)
    print(f'Saved: {out_path}')

if __name__ == '__main__':
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else 'fig1_drift.pdf'
    plot_fig1(out)
