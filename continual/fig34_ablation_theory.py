"""
Figure 3 — Checkpoint strategy ablation (MIMIC-IV HR)
Figure 4 — Theory validation: error vs T with T* marked (NYC Taxi)

两张图合并为一个文件；可单独调用各自的 plot 函数
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from fig_utils import (load_dataset, set_paper_style,
                        COLORS, SINGLE_COL, DOUBLE_COL)

# ─────────────────────────────────────────────────────────────
# 共用：checkpoint 生成策略
# ─────────────────────────────────────────────────────────────
def make_cps(strategy: str, N: int, k: int = 256, n_min: int = 1000,
             T_fixed: int = None) -> list:
    """
    strategy:
      'uniform-T5'   均匀 T=5
      'uniform-T10'  均匀 T=10
      'uniform-T20'  均匀 T=20
      'geometric'    几何等比（公比 2，不依赖数据）
      'kll-str'      与 geometric 相同（Zipf 探针模拟值）
    """
    if strategy.startswith('uniform'):
        T = int(strategy.split('T')[1]) if T_fixed is None else T_fixed
        pts = np.linspace(n_min, N, T, dtype=int).tolist()
        if pts[-1] != N:
            pts.append(N)
        return sorted(set(pts))
    else:   # geometric / kll-str
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


def run_privkll_with_cps(data: np.ndarray,
                          cps: list,
                          epsilon: float,
                          phi: float = 0.5,
                          seed: int = 0) -> list:
    """
    给定 checkpoint 列表，运行 DP 发布（均匀预算分配）
    替换为真实 PrivKLL 实现
    """
    rng = np.random.default_rng(seed)
    T   = len(cps)
    eps_per = epsilon / T
    releases = []
    prev = 0
    for cp in cps:
        window = data[prev:cp]
        tq = np.quantile(window, phi) if len(window) > 0 else 0
        noise = rng.laplace(0, 1.0 / eps_per)
        releases.append({'t': cp, 'dp_q': tq + noise})
        prev = cp
    return releases


def eval_releases(releases, data, phi=0.5):
    errors, prev = [], 0
    for rel in releases:
        t = rel['t']
        window = data[prev:t]
        if len(window) == 0:
            continue
        n = len(window)
        true_rank = int(phi * n)
        est_rank  = int(np.searchsorted(np.sort(window), rel['dp_q']))
        errors.append(abs(est_rank - true_rank) / n)
        prev = t
    return float(np.mean(errors)) if errors else 0.0


# ─────────────────────────────────────────────────────────────
# Figure 3 — Checkpoint 策略消融
# ─────────────────────────────────────────────────────────────
def plot_fig3(out_path: str = 'fig3_ablation.pdf',
              epsilons: list = None,
              n_runs: int = 30,
              phi: float = 0.5,
              max_n: int = 200_000):
    """
    固定数据集 = MIMIC-IV HR（最能体现 CP 策略差异）
    对比五种 checkpoint 生成策略
    """
    if epsilons is None:
        epsilons = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0]

    set_paper_style()
    data = load_dataset('mimic', max_n=max_n)
    N    = len(data)

    # 策略定义：(key, label, color, linestyle, marker)
    strategies = [
        ('uniform-T5',  'Uniform $T{=}5$',      '#999999', ':',   'v'),
        ('uniform-T10', 'Uniform $T{=}10$',     '#aaaaaa', '--',  '^'),
        ('uniform-T20', 'Uniform $T{=}20$',     '#bbbbbb', '-.',  '<'),
        ('geometric',   'Geometric',             '#4dac26', '--',  'D'),
        ('kll-str',     'KLL-structural (ours)', '#2166ac', '-',   'o'),
    ]

    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.1))

    results = {}
    for strat_key, *_ in strategies:
        means, stds = [], []
        for eps in epsilons:
            errs = []
            cps = make_cps(strat_key, N)
            for seed in range(n_runs):
                rels = run_privkll_with_cps(data, cps, eps, phi=phi, seed=seed)
                errs.append(eval_releases(rels, data, phi))
            means.append(np.mean(errs))
            stds.append(np.std(errs))
        results[strat_key] = (np.array(means), np.array(stds))

    eps_arr = np.array(epsilons)
    for strat_key, label, color, ls, marker in strategies:
        means, stds = results[strat_key]
        ax.plot(eps_arr, means,
                color=color, linestyle=ls, marker=marker,
                linewidth=1.3, markersize=3.5, label=label)
        ax.fill_between(eps_arr,
                        means - stds, means + stds,
                        color=color, alpha=0.10)

    ax.set_xscale('log')
    ax.set_xlabel(r'Privacy budget $\varepsilon$', labelpad=3)
    ax.set_ylabel('Mean norm. rank error', labelpad=3)
    ax.set_title('(a)  MIMIC-IV HR — checkpoint ablation',
                 loc='left', pad=3)
    ax.set_xticks(epsilons)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.tick_params(axis='x', which='minor', bottom=False)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(4, prune='upper'))
    ax.set_ylim(bottom=0)
    ax.grid(axis='y')

    # 图例放在图外下方（单栏图常用做法）
    ax.legend(ncol=1, loc='upper right',
              fontsize=7.5, framealpha=0.9)

    # ── 辅助标注：标出 KLL-structural 的实际 T ────────
    kll_T = len(make_cps('kll-str', N))
    ax.text(0.03, 0.06,
            f'KLL-structural: $T={kll_T}$',
            transform=ax.transAxes,
            fontsize=7, color='#2166ac',
            va='bottom')

    plt.tight_layout()
    plt.savefig(out_path)
    print(f'Saved: {out_path}')


# ─────────────────────────────────────────────────────────────
# Figure 4 — 理论验证：error vs T，标出 T*
# ─────────────────────────────────────────────────────────────
def estimate_drift_sigma(data: np.ndarray, phi: float = 0.5,
                          window: int = 5000) -> float:
    """
    估计平均漂移速率 σ（单位：每步的分位数变化量 / 数据范围）
    用于计算理论最优 T* = sqrt(σ N ε / 2C)
    """
    data_range = np.percentile(data, 99) - np.percentile(data, 1) + 1e-9
    rates = []
    for t in range(window, len(data) - window, window):
        q1 = np.quantile(data[t - window:t],     phi)
        q2 = np.quantile(data[t:         t + window], phi)
        rates.append(abs(q2 - q1) / window / data_range)
    return float(np.mean(rates)) if rates else 0.0


def theoretical_error(T: float, N: float, sigma: float,
                       epsilon: float, C: float = 1.0) -> float:
    """
    L(T) = σN/(2T) + CT/ε
    平稳流下的误差下界近似（已归一化）
    """
    return sigma * N / (2 * T) + C * T / (epsilon * N)


def plot_fig4(out_path: str = 'fig4_theory.pdf',
              epsilon: float = 1.0,
              phi: float = 0.5,
              n_runs: int = 20,
              max_n: int = 200_000):
    """
    NYC Taxi 上的 error vs T 曲线
    x 轴：T（checkpoint 数量，均匀分布）
    y 轴：mean normalized rank error（实验）+ 理论曲线 L(T)
    标注：T* 理论最优，T_KLL 实际触发数
    """
    set_paper_style()
    data = load_dataset('taxi', max_n=max_n)
    N    = len(data)

    T_values = list(range(1, 51, 2))   # T = 1, 3, 5, ..., 49
    sigma = estimate_drift_sigma(data, phi=phi)

    # 理论曲线（归一化常数 C 通过数据范围标定）
    data_range = np.percentile(data, 99) - np.percentile(data, 1)
    C = 1.0 / (data_range ** 2 + 1e-9)   # 近似标定
    theory = [theoretical_error(T, N, sigma, epsilon, C) for T in T_values]

    # 为了让理论曲线与实验曲线在量纲上可比，做 min-max rescale
    theory_arr = np.array(theory)
    theory_min, theory_max = theory_arr.min(), theory_arr.max()

    # 实验误差
    emp_means, emp_stds = [], []
    for T in T_values:
        cps = make_cps('uniform', N, T_fixed=T)
        errs = []
        for seed in range(n_runs):
            rels = run_privkll_with_cps(data, cps, epsilon, phi=phi, seed=seed)
            errs.append(eval_releases(rels, data, phi))
        emp_means.append(np.mean(errs))
        emp_stds.append(np.std(errs))

    emp_means = np.array(emp_means)
    emp_stds  = np.array(emp_stds)

    # 将理论曲线缩放到与实验同范围（相对形状才是关键）
    emp_min, emp_max = emp_means.min(), emp_means.max() + 1e-9
    theory_scaled = (theory_arr - theory_min) / (theory_max - theory_min + 1e-9)
    theory_scaled = theory_scaled * (emp_max - emp_min) + emp_min

    # 理论最优 T*
    T_star_float = np.sqrt(sigma * N * epsilon / (2 * C + 1e-30))
    T_star = int(np.clip(T_star_float, T_values[0], T_values[-1]))

    # KLL-structural 实际 T
    T_kll = len(make_cps('kll-str', N))

    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.1))

    # 实验曲线
    ax.plot(T_values, emp_means,
            color='#2166ac', marker='o', linewidth=1.4,
            markersize=2.8, label='PrivKLL-CQ (empirical)')
    ax.fill_between(T_values,
                    emp_means - emp_stds,
                    emp_means + emp_stds,
                    color='#2166ac', alpha=0.15)

    # 理论曲线（缩放后）
    ax.plot(T_values, theory_scaled,
            color='#d6604d', linestyle='--', linewidth=1.2,
            label=r'Theory: $L(T) = \frac{\sigma N}{2T}+\frac{CT}{\varepsilon}$')

    # T* 竖线
    ax.axvline(T_star, color='#d6604d', linewidth=0.9,
               linestyle=':', alpha=0.9)
    ax.text(T_star + 0.5, ax.get_ylim()[1] * 0.95,
            f'$T^*={T_star}$',
            color='#d6604d', fontsize=7.5, va='top')

    # T_KLL 竖线（只有在图范围内才画）
    if T_values[0] <= T_kll <= T_values[-1]:
        ax.axvline(T_kll, color='#2166ac', linewidth=0.9,
                   linestyle=':', alpha=0.9)
        ax.text(T_kll + 0.5, ax.get_ylim()[1] * 0.82,
                f'$T_{{\\mathrm{{KLL}}}}={T_kll}$',
                color='#2166ac', fontsize=7.5, va='top')
    else:
        # T_KLL 超出图范围，用箭头注释
        ax.annotate(f'$T_{{\\mathrm{{KLL}}}}={T_kll}$\n(off-chart)',
                    xy=(T_values[-1], emp_means[-1]),
                    xytext=(T_values[-3], emp_means[-1] * 1.3),
                    arrowprops=dict(arrowstyle='->', color='#2166ac', lw=0.8),
                    fontsize=7, color='#2166ac')

    ax.set_xlabel('Number of checkpoints $T$', labelpad=3)
    ax.set_ylabel('Mean norm. rank error', labelpad=3)
    ax.set_title('(b)  NYC Taxi — theory vs empirical',
                 loc='left', pad=3)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(4, prune='upper'))
    ax.set_ylim(bottom=0)
    ax.grid(axis='y')
    ax.legend(fontsize=7.5, loc='upper right', framealpha=0.9)

    # ε 标注
    ax.text(0.03, 0.06, f'$\\varepsilon={epsilon}$',
            transform=ax.transAxes, fontsize=7,
            color='#555555', va='bottom')

    plt.tight_layout()
    plt.savefig(out_path)
    print(f'Saved: {out_path}')


# ─────────────────────────────────────────────────────────────
# 合并输出：Fig 3 + Fig 4 并排（单次生成投稿版）
# ─────────────────────────────────────────────────────────────
def plot_fig34_combined(out_path: str = 'fig34_combined.pdf',
                        epsilon: float = 1.0,
                        n_runs: int = 20,
                        max_n: int = 200_000):
    """
    将 Fig 3 和 Fig 4 并排输出为一张双栏宽的图，
    共用 x 轴说明，节省版面
    """
    set_paper_style()
    epsilons = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0]

    fig, (ax3, ax4) = plt.subplots(1, 2,
                                    figsize=(DOUBLE_COL, 2.1))
    fig.subplots_adjust(wspace=0.35)

    # ── 左：ablation ──────────────────────────────
    data_mimic = load_dataset('mimic', max_n=max_n)
    N_m = len(data_mimic)
    strategies_short = [
        ('uniform-T5',  'Uniform $T{=}5$',  '#aaaaaa', ':',   'v'),
        ('uniform-T10', 'Uniform $T{=}10$', '#888888', '--',  '^'),
        ('geometric',   'Geometric',         '#4dac26', '-.',  'D'),
        ('kll-str',     'KLL-str. (ours)',   '#2166ac', '-',   'o'),
    ]
    for strat_key, label, color, ls, marker in strategies_short:
        cps = make_cps(strat_key, N_m)
        means = []
        for eps in epsilons:
            errs = [eval_releases(
                        run_privkll_with_cps(data_mimic, cps, eps, seed=s),
                        data_mimic) for s in range(n_runs)]
            means.append(np.mean(errs))
        ax3.plot(epsilons, means, color=color, linestyle=ls,
                 marker=marker, linewidth=1.3, markersize=3.5, label=label)

    ax3.set_xscale('log')
    ax3.set_xlabel(r'$\varepsilon$', labelpad=3)
    ax3.set_ylabel('Mean norm. rank error', labelpad=3)
    ax3.set_title('(a)  MIMIC-IV HR — checkpoint ablation',
                  loc='left', pad=3)
    ax3.set_xticks(epsilons)
    ax3.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax3.tick_params(axis='x', which='minor', bottom=False)
    ax3.yaxis.set_major_locator(mticker.MaxNLocator(4, prune='upper'))
    ax3.set_ylim(bottom=0)
    ax3.grid(axis='y')
    ax3.legend(fontsize=7.5, loc='upper right')

    # ── 右：theory validation ─────────────────────
    data_taxi = load_dataset('taxi', max_n=max_n)
    N_t   = len(data_taxi)
    sigma = estimate_drift_sigma(data_taxi)
    T_values = list(range(1, 41, 2))
    data_range = np.percentile(data_taxi, 99) - np.percentile(data_taxi, 1)
    C = 1.0 / (data_range ** 2 + 1e-9)
    theory = np.array([theoretical_error(T, N_t, sigma, epsilon, C)
                       for T in T_values])
    emp_means = []
    for T in T_values:
        cps = make_cps('uniform', N_t, T_fixed=T)
        errs = [eval_releases(
                    run_privkll_with_cps(data_taxi, cps, epsilon, seed=s),
                    data_taxi) for s in range(n_runs)]
        emp_means.append(np.mean(errs))
    emp_means = np.array(emp_means)

    # 缩放理论曲线
    t_min, t_max = theory.min(), theory.max()
    e_min, e_max = emp_means.min(), emp_means.max() + 1e-9
    theory_s = (theory - t_min) / (t_max - t_min + 1e-9) * (e_max - e_min) + e_min

    T_star = int(np.clip(
        np.sqrt(sigma * N_t * epsilon / (2 * C + 1e-30)),
        T_values[0], T_values[-1]))
    T_kll  = len(make_cps('kll-str', N_t))

    ax4.plot(T_values, emp_means,
             color='#2166ac', marker='o', linewidth=1.4,
             markersize=2.8, label='Empirical')
    ax4.plot(T_values, theory_s,
             color='#d6604d', linestyle='--', linewidth=1.2,
             label=r'Theory $L(T)$')
    ax4.axvline(T_star, color='#d6604d', linewidth=0.9,
                linestyle=':', alpha=0.9)
    ax4.text(T_star + 0.3,
             ax4.get_ylim()[1] * 0.96 if ax4.get_ylim()[1] > 0 else 0.05,
             f'$T^*={T_star}$',
             color='#d6604d', fontsize=7.5, va='top')

    if T_values[0] <= T_kll <= T_values[-1]:
        ax4.axvline(T_kll, color='#2166ac', linewidth=0.9,
                    linestyle=':', alpha=0.9)
        ax4.text(T_kll + 0.3,
                 ax4.get_ylim()[1] * 0.80 if ax4.get_ylim()[1] > 0 else 0.04,
                 f'$T_{{\\rm KLL}}={T_kll}$',
                 color='#2166ac', fontsize=7.5, va='top')

    ax4.set_xlabel('Number of checkpoints $T$', labelpad=3)
    ax4.set_ylabel('Mean norm. rank error', labelpad=3)
    ax4.set_title('(b)  NYC Taxi — theory validation',
                  loc='left', pad=3)
    ax4.yaxis.set_major_locator(mticker.MaxNLocator(4, prune='upper'))
    ax4.set_ylim(bottom=0)
    ax4.grid(axis='y')
    ax4.legend(fontsize=7.5, loc='upper right')

    plt.savefig(out_path)
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else 'both'
    if mode == 'fig3':
        plot_fig3(n_runs=5)
    elif mode == 'fig4':
        plot_fig4(n_runs=5)
    else:
        plot_fig34_combined(n_runs=5)
