"""
共用工具：数据加载接口、评估指标、matplotlib 样式
所有图共用此文件
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from pathlib import Path

# ── 论文排版参数 ─────────────────────────────────────────────
# VLDB/SIGMOD 单栏 = 3.33in, 双栏 = 7.0in
SINGLE_COL = 3.33
DOUBLE_COL = 7.0

def set_paper_style():
    plt.rcParams.update({
        'font.family':       'Times New Roman',
        'font.size':         9,
        'axes.titlesize':    9,
        'axes.labelsize':    9,
        'xtick.labelsize':   8,
        'ytick.labelsize':   8,
        'legend.fontsize':   8,
        'legend.framealpha': 0.9,
        'legend.edgecolor':  '#cccccc',
        'axes.linewidth':    0.8,
        'axes.spines.top':   False,
        'axes.spines.right': False,
        'lines.linewidth':   1.4,
        'lines.markersize':  4,
        'grid.linewidth':    0.4,
        'grid.alpha':        0.5,
        'figure.dpi':        300,
        'savefig.dpi':       300,
        'savefig.bbox':      'tight',
        'savefig.pad_inches':0.02,
    })

# ── 色板（colorblind-safe, 与线型配对） ───────────────────────
COLORS = {
    'privkll':  '#2166ac',   # 蓝
    'gkexp':    '#d6604d',   # 红橙
    'uniform':  '#999999',   # 灰
    'geom':     '#4dac26',   # 绿
    'kll_str':  '#2166ac',   # 与 privkll 同色（它就是 privkll 的 CP 策略）
    'drift_50': '#2166ac',
    'drift_90': '#d6604d',
    'drift_25': '#4dac26',
}
DASHES = {
    'privkll':  (None, None),      # solid
    'gkexp':    (4, 2),            # dashed
    'uniform':  (1, 1),            # dotted
    'geom':     (4, 2, 1, 2),      # dash-dot
    'kll_str':  (None, None),
}
MARKERS = {
    'privkll': 'o',
    'gkexp':   's',
    'uniform': '^',
    'geom':    'D',
    'kll_str': 'o',
}

# ── 评估指标 ──────────────────────────────────────────────────
def rank_error(estimated_quantile: float, true_data_window: np.ndarray) -> float:
    """
    归一化秩误差 = |rank(q̂) - true_rank| / n
    estimated_quantile: DP 机制输出的分位数值
    true_data_window:   该 checkpoint 对应的数据窗口
    """
    n = len(true_data_window)
    if n == 0:
        return 0.0
    true_rank = np.searchsorted(np.sort(true_data_window), estimated_quantile)
    target_rank = int(0.5 * n)   # 对 median；可参数化
    return abs(true_rank - target_rank) / n

def mean_rank_error_over_cps(releases: list, data: np.ndarray, phi: float = 0.5) -> float:
    """
    releases: list of dict {t: int, dp_q: float}
    data: 完整数据流
    返回所有 checkpoint 上的平均归一化秩误差
    """
    errors = []
    prev = 0
    for rel in releases:
        t = rel['t']
        window = data[prev:t]
        if len(window) == 0:
            continue
        n = len(window)
        true_rank = int(phi * n)
        estimated_rank = np.searchsorted(np.sort(window), rel['dp_q'])
        errors.append(abs(estimated_rank - true_rank) / n)
        prev = t
    return float(np.mean(errors)) if errors else 0.0

# ── 数据加载接口（替换为真实路径） ────────────────────────────
# fig_utils.py 中的修改

def load_dataset(name: str, max_n: int = None) -> np.ndarray:
    """
    name: 'mimic' | 'power' | 'taxi'
    max_n: 截断长度，None 表示全量
    """
    # 获取当前工作空间根目录（或者直接指定你的 .npy 文件存放目录）
    ROOT_DIR = Path(__file__).resolve().parents[1]
    
    # ── 映射真实提取的三个数据集 ──────────────────────────────────
    if name == 'mimic':
        file_path = ROOT_DIR / "data/mimic_hr.npy"
    elif name == 'power':
        file_path = ROOT_DIR / "data/power_active.npy"
    elif name == 'taxi':
        file_path = ROOT_DIR / "data/taxi_duration.npy"
    else:
        raise ValueError(f"Unknown dataset name: {name}")

    # ── 内存安全的高速加载 ──────────────────────────────────
    if file_path.exists():
        # 使用 mmap_mode='r'，切片时才会加载数据到内存，极其安全高速
        data = np.load(file_path, mmap_mode='r')
        if max_n:
            # 切片并复制到常驻内存中，转换成 float64 供算法解算
            return np.array(data[:max_n], dtype=np.float64)
        return np.array(data, dtype=np.float64)
    
    # ── Fallback 机制：如果文件不存在，保留原有的模拟生成，防止代码报错 ──
    print(f"[Warning] {file_path} not found. Falling back to synthetic generation.")
    rng = np.random.default_rng(42)
    N = max_n or 2000_000
    if name == 'mimic':
        regimes = [(0, 0.22, 75, 10), (0.22, 0.45, 120, 14), (0.45, 0.62, 70, 8), (0.62, 0.80, 45, 7), (0.80, 1.00, 82, 10)]
        data = np.empty(N)
        for s, e, mu, sd in regimes:
            lo, hi = int(s*N), int(e*N)
            data[lo:hi] = rng.normal(mu, sd, hi-lo)
        data = np.clip(data, 20, 220)
    elif name == 'power':
        t = np.linspace(0, 4*np.pi, N)
        trend = 1.0 + 0.5 * np.sin(t / 2)
        data = np.abs(trend + 0.15 * rng.standard_normal(N))
        data = np.clip(data, 0.01, 5.0)
    else:   # taxi
        data = rng.lognormal(mean=0.5, sigma=0.8, size=N)
        data = np.clip(data, 0.1, 30.0)
        
    if max_n:
        data = data[:max_n]
    return data.astype(np.float64)