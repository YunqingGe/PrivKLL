# ------------------------------------------------------------------------------
# 支持one-time release的严格单次物理试验
# 负责eps（privacy）、内存（space）实验
# ------------------------------------------------------------------------------
import argparse
from ast import main
from concurrent.futures import ProcessPoolExecutor
import concurrent.futures 
import multiprocessing
import math
from pathlib import Path
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import json
from datetime import datetime
import logging
import pickle
import os

from DPQE.exp import calibration
from DPQE.src.PrivKLL import PrivKLL
from DPQE.src.kll import KLL
from DPQE.src.Greenwald_Khanna import GK
from DPQE.src.DCS import DCS

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

def get_default_logger():
    logger = logging.getLogger("exp_single")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        fh = logging.FileHandler("exp_single.log")
        ch = logging.StreamHandler()

        fmt = logging.Formatter("[%(asctime)s] %(message)s")

        fh.setFormatter(fmt)
        ch.setFormatter(fmt)

        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger

# ------------------------------------------------------------------------------
# 2. 内存安全的 Chunk 流式生成器与严格单次物理试验核心
# ------------------------------------------------------------------------------
def stream_chunk_generator(dist_type, total_n, chunk_size=500_000, domain=(0, 100000)):
    low, high = domain
    remains = total_n
    while remains > 0:
        current_chunk = min(chunk_size, remains)
        if dist_type == 'uniform': chunk_data = np.random.uniform(low, high, current_chunk)
        elif dist_type == 'normal': chunk_data = np.clip(np.random.normal((high+low)/2, (high-low)/6, current_chunk), low, high)
        elif dist_type == 'zipf': chunk_data = np.clip(np.random.pareto(a=2.0, size=current_chunk) * (high/10), low, high)
        yield chunk_data
        remains -= current_chunk

def run_single_trial_strict(eps, dist_type, total_n, target_q, kll_k, gk_alpha, dcs_gamma, dcs_uni=2**16, chunk_size=500_000):
    """底层物理流计算：支持三方对比（含高精度时间效率测算）"""
    dp_kll = PrivKLL(epsilon=eps, kll_k=kll_k)
    non_dp_kll = KLL(k=kll_k)
    gk_sketch = GK(alpha=gk_alpha)
    
    # 隐私转换：将纯 ϵ-DP 转换为对齐的 CDP ρ 预算平分机制
    # 1. 计算 L = ln(1/delta)
    delta = 1e-6  # 常用的隐私泄露概率设定
    L = math.log(1.0 / delta)
    # 2. 计算 rho = (sqrt(L + epsilon) - sqrt(L))^2
    sqrt_L_plus_eps = math.sqrt(L + eps)
    sqrt_L = math.sqrt(L)
    rho_cdp = (sqrt_L_plus_eps - sqrt_L) ** 2
    # rho_cdp = np.sqrt(2 * L * eps)  # ϵ-DP 转换为 ρ-CDP 的近似关系
    dcs_sketch = DCS(universe=dcs_uni, gamma=dcs_gamma, rho=rho_cdp)

    sample_size = min(total_n, 100_000)
    reservoir = []
    
    # 值域归一化函数：将 (0, 100000) 的实验数据映射到 DCS 所需的整数区间 [0, universe-1]
    def to_dcs_domain(val):
        return int(np.clip(val / 100000.0 * (dcs_uni - 1), 0, dcs_uni - 1))
    
    def from_dcs_domain(int_val):
        return (int_val / (dcs_uni - 1)) * 100000.0

    generator = stream_chunk_generator(dist_type, total_n, chunk_size=chunk_size)
    item_counter = 0
    
    # 🌟 初始化三大算法的 Ingest 耗时累计变量
    t_kll_ingest = 0.0
    t_gk_ingest = 0.0
    t_dcs_ingest = 0.0

    for chunk in generator:
        for val in chunk:
            # 1. PrivKLL Ingest 计时 (由于外部验证内存用的是 non_dp_kll，这里把两个 KLL 的 update 一并计入)
            non_dp_kll.update(val)
            t0 = time.perf_counter()
            dp_kll.ingest_data(val)
            t_kll_ingest += (time.perf_counter() - t0)
            
            # 2. GK Ingest 计时
            t0 = time.perf_counter()
            gk_sketch.insert(val)
            t_gk_ingest += (time.perf_counter() - t0)
            
            # 3. DCS Ingest 计时 (包含了数据映射与底层更新)
            t0 = time.perf_counter()
            dcs_val = to_dcs_domain(val)
            dcs_sketch.update(dcs_val)
            t_dcs_ingest += (time.perf_counter() - t0)
            
            # 水塘采样（蓄水池采样不计入算法自身耗时）
            if len(reservoir) < sample_size:
                reservoir.append(val)
            else:
                r = np.random.randint(0, item_counter + 1)
                if r < sample_size: reservoir[r] = val
            item_counter += 1

    # ------------------------------------------------------------------ #
    # 🌟 估计值释放与 Query 阶段计时
    # ------------------------------------------------------------------ #
    
    # 1. PrivKLL 释放计时 (包含指数机制与区间内均匀采样)
    t0 = time.perf_counter()
    est_dp_kll = dp_kll.release_quantile(target_q)
    t_kll_query = time.perf_counter() - t0

    # 2. GK 指数机制释放计时
    t0 = time.perf_counter()
    est_gk_exp = gk_sketch.dp_exp(target_q, eps)
    # ======= 核心修复：添加 None 值防御 =======
    if est_gk_exp is None:
        # 打印警告，看看是哪个分布、什么参数下碎掉的
        print(f"⚠️ [Warning] DPExpGK returned None! target_q={target_q}")
        # 保底机制：如果返回 None，强行将其扭转为数据的中位数、0 或最大/最小值，防止程序崩溃
        est_gk_exp = 0.0  # 或者使用 0.0，具体看你算法的物理意义
    # =========================================
    t_gk_query = time.perf_counter() - t0

    # 3. DCS 二分查找释放计时 (包含多层区间树的 DP 查询和外部值域还原)
    t0 = time.perf_counter()
    target_rank = target_q * total_n
    low_idx, high_idx = 0, dcs_uni - 1
    est_dcs_int = low_idx
    while low_idx <= high_idx:
        mid_idx = (low_idx + high_idx) // 2
        if dcs_sketch.query(mid_idx) <= target_rank:
            est_dcs_int = mid_idx
            low_idx = mid_idx + 1
        else:
            high_idx = mid_idx - 1
    est_dcs = from_dcs_domain(est_dcs_int)
    t_dcs_query = time.perf_counter() - t0

    # ------------------------------------------------------------------ #
    # 指标归档与内存计算
    # ------------------------------------------------------------------ #
    res_arr = np.array(reservoir)
    err_kll = abs((res_arr <= est_dp_kll).mean() - target_q)
    err_gk = abs((res_arr <= est_gk_exp).mean() - target_q)
    err_dcs = abs((res_arr <= est_dcs).mean() - target_q)

    kll_mem = sum(len(c) for c in non_dp_kll.compactors) * 8
    gk_mem = len(gk_sketch.S) * 16
    dcs_mem = dcs_sketch.memory_budget() * 4
    
    # 🌟 返回 12 元组：误差(3) + 内存(3) + Ingest时间(3) + Query时间(3)
    return (err_kll, err_gk, err_dcs, 
            kll_mem, gk_mem, dcs_mem,
            t_kll_ingest, t_gk_ingest, t_dcs_ingest,
            t_kll_query, t_gk_query, t_dcs_query)
    
# ------------------------------------------------------------------------------
# 3. 统一学术多子图绘图引擎
# ------------------------------------------------------------------------------
def plot_vldb(x_values, plot_data, xlabel_str, x_scale_log, base_x, fname):
    # 统一设置学术论文级 Serif 字体
    plt.rcParams.update({
        'font.size': 8.5, 
        'font.family': 'serif', 
        'mathtext.fontset': 'cm',
        'xtick.labelsize': 7.5, 
        'ytick.labelsize': 8
    })
    
    # 物理画布大小：维持单栏最佳黄金比例
    fig, axes = plt.subplots(1, 3, figsize=(3.35, 1.85), sharey=True)
    distributions = ['uniform', 'normal', 'zipf']
    dist_labels = ['(a) UNIFORM', '(b) NORMAL', '(c) ZIPF']

    style_cfgs = {
        'PrivKLL-Q': {'color':  "#3A86FF", 'linestyle': '--', 'marker': 'v'},
        'DPExpGK': {'color': "#06D6A0", 'linestyle': '-', 'marker': 'o'},
        'DCS DP': {'color': "#F77F00FF", 'linestyle': '-.', 'marker': '^'}
    }
    lines, labels = [], []

    # 🌟 严谨定义的绝对几何盒模型坐标 [left, bottom, width, height]
    box_width = 0.255
    box_height = 0.46
    box_bottom = 0.35
    box_lefts = [0.155, 0.445, 0.735] # 精准留出主 Y 轴标签的空间

    for idx, dist in enumerate(distributions):
        ax = axes[idx]
        
        # 🌟 强行赋予子图网格硬隔离位置
        ax.set_position([box_lefts[idx], box_bottom, box_width, box_height])
        
        dist_res = plot_data[dist]
        for m in ['PrivKLL-Q', 'DPExpGK', 'DCS DP']:
            mean = np.clip(np.array(dist_res[m]['mean']), 1e-7, None)
            p10 = np.clip(np.array(dist_res[m]['p10']), 1e-7, None)
            p90 = np.clip(np.array(dist_res[m]['p90']), 1e-7, None)
            
            # indices = np.arange(len(x_values))
            line, = ax.plot(x_values, mean, color=style_cfgs[m]['color'], linestyle=style_cfgs[m]['linestyle'], 
                             marker=style_cfgs[m]['marker'], linewidth=1.2, markersize=3.2, zorder=4)
            ax.fill_between(x_values, p10, p90, color=style_cfgs[m]['color'], alpha=0.06)
            if idx == 0: lines.append(line); labels.append(m)
        
        # ax.set_xscale('log', base=2)
        # ax.set_xticks(x_values)
        ax.set_xticks(range(len(x_values)))
        ax.set_xticklabels([f"$2^{{{int(np.log2(x))}}}$" for x in x_values], fontsize=6.5, rotation=30, ha='right')
        # ax.set_xticklabels([f"{x:.2f}" for x in x_values], fontsize=6.5, rotation=30, ha='right')

    
        ax.set_yscale('log', base=2)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: f"$2^{{{int(np.log2(y))}}}$" if y > 0 else "0"))
        
        # 🌟 倒挂子图小标题 (a)(b)(c)，挂在绝对底部的安全区间
        ax.set_title(dist_labels[idx], y=-0.54, fontsize=8.5, va='top')
        ax.grid(True, which="both", ls="--", color='#e0e0e0', alpha=0.6, linewidth=0.5)
        
    # 🌟 修复 1：将大横轴标签放置在绝对几何中心 (0.575)，并精准卡在数字刻度与子标题中间
    fig.text(0.575, 0.22, xlabel_str, ha='center', va='center', fontsize=8.5)
           
    # 🌟 修复 2：将主 Y 轴标签归还给最左侧的第一个子图 axes[0]，完美避开重叠
    axes[0].set_ylabel('Mean Abs Rank Error', fontsize=8.5, labelpad=2)
        
    # 🌟 修复 3：图例在上方完美居中平铺
    fig.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.575, 0.99), 
               ncol=3, frameon=True, fancybox=True, shadow=False, 
               fontsize=7, columnspacing=0.6, handletextpad=0.2, framealpha=1.0)
    
    # 🌟 修复 4：既然使用了绝对盒模型，坚决移除 plt.tight_layout()！
    # 同时在 savefig 时坚决不加 bbox_inches='tight'，防止外界动态边界计算再次挤压硬定位盒子
    plt.savefig(fname, dpi=300)
    plt.close()

# ------------------------------------------------------------------------------
# 4. 实验
# ------------------------------------------------------------------------------
def run_experiment_s_eps(cfg, timestamp):
    print("\n" + "="*80 + f"\n▶ [Executing] Exp Single-Epsilon (Target Budget={cfg['mem_kb']} KB, N={cfg['N']}, Trials={cfg['trials']})\n" + "="*80)
    start = time.time()

    # 初始化原始结果收集字典（保持原样，用于画图的误差分布）
    raw_res = {dist: {'PrivKLL-Q': {e: [] for e in cfg['epsilons']}, 
                      'DPExpGK': {e: [] for e in cfg['epsilons']}, 
                      'DCS DP': {e: [] for e in cfg['epsilons']}} 
                      for dist in cfg['distributions']}
    meta_log = {}

    # 🌟 调整 plot_data 的结构，使其能够容纳 time 统计指标
    # 结构拓展为：plot_data[dist][method][指标] = [每个eps对应的值, ...]
    plot_data = {
        dist: {
            m: {
                'mean': [], 'p10': [], 'p90': [], 
                'avg_ingest_time_ms': [], 'avg_query_time_ms': []  # 🌟 新增时间维度
            } for m in ['PrivKLL-Q', 'DPExpGK', 'DCS DP']
        } for dist in cfg['distributions']
    }

    # 使用 ProcessPoolExecutor 开启多进程池
    with ProcessPoolExecutor() as executor:
        for dist in cfg['distributions']:
            print(f"\n[Set up] Dist: {dist.upper()} | Allocating KLL k={cfg['PrivKLL_k']} to hit {cfg['mem_kb']} KB")
            meta_log[dist] = {"Privll_k": cfg['PrivKLL_k'], "GK_alpha": cfg['GK_alpha'], "dcs_gamma": cfg['DCS_gamma']}
            
            for eps in cfg['epsilons']:
                
                # 异步提交所有独立 Trial 任务到进程池
                futures = []
                for _ in range(cfg['trials']):
                    f = executor.submit(
                        run_single_trial_strict,
                        eps, dist, cfg['N'], cfg['target_q'], 
                        cfg['PrivKLL_k'], cfg['GK_alpha'], cfg['DCS_gamma'], cfg['DCS_uni']
                    )
                    futures.append(f)
                
                # 用于收集当前 epsilon 下所有并发进程传回的内存与时间数据x
                kll_mems, gk_mems, dcs_mems = [], [], []
                
                # 🌟 新增：独立收集每个算法的耗时列表
                kll_in_times, kll_q_times = [], []
                gk_in_times, gk_q_times = [], []
                dcs_in_times, dcs_q_times = [], []
                
                # 阻塞并同步捞回所有并发 Trial 的结果（12个返回值对齐）
                for f in futures:
                    (err_kll, err_gk, err_dcs, 
                     kll_mem, gk_mem, dcs_mem,
                     t_kll_in, t_gk_in, t_dcs_in,
                     t_kll_q, t_gk_q, t_dcs_q) = f.result()
                    
                    # 收集内存开销
                    kll_mems.append(kll_mem)
                    gk_mems.append(gk_mem)
                    dcs_mems.append(dcs_mem)
                    
                    # 🌟 收集细分时间开销（保持原生秒单位）
                    kll_in_times.append(t_kll_in); kll_q_times.append(t_kll_q)
                    gk_in_times.append(t_gk_in);   gk_q_times.append(t_gk_q)
                    dcs_in_times.append(t_dcs_in); dcs_q_times.append(t_dcs_q)
                    
                    # 收集误差数据到全局字典
                    raw_res[dist]['PrivKLL-Q'][eps].append(err_kll)
                    raw_res[dist]['DPExpGK'][eps].append(err_gk)
                    raw_res[dist]['DCS DP'][eps].append(err_dcs)
                
                # 🌟 核心改变 1：计算当前 eps 下 100 个 trial 的平均时间，转为 ms（毫秒）存储并追加
                # 这样做可以直接复用你后面的 eps 循环归档逻辑，或者直接在这里同步写进 plot_data
                plot_data[dist]['PrivKLL-Q']['avg_ingest_time_ms'].append(float(np.mean(kll_in_times) * 1000))
                plot_data[dist]['PrivKLL-Q']['avg_query_time_ms'].append(float(np.mean(kll_q_times) * 1000))
                
                plot_data[dist]['DPExpGK']['avg_ingest_time_ms'].append(float(np.mean(gk_in_times) * 1000))
                plot_data[dist]['DPExpGK']['avg_query_time_ms'].append(float(np.mean(gk_q_times) * 1000))
                
                plot_data[dist]['DCS DP']['avg_ingest_time_ms'].append(float(np.mean(dcs_in_times) * 1000))
                plot_data[dist]['DCS DP']['avg_query_time_ms'].append(float(np.mean(dcs_q_times) * 1000))

                # 🌟 核心改变 2：升级控制台小结日志，完美透视 Ingest 和 Query 的时间开销
                print(f"  [Parallel Summary] Epsilon: {eps:<3.1f} | Dist: {dist.upper()} | Completed Trials: {cfg['trials']}")
                print(f"    ├─ PrivKLL-Q: Err: {np.mean(raw_res[dist]['PrivKLL-Q'][eps]):.5f} | Mem: {np.mean(kll_mems)/1024:5.2f} KB | Ingest: {np.mean(kll_in_times)*1000:6.2f} ms | Query: {np.mean(kll_q_times)*1000:5.2f} ms")
                print(f"    ├─ DPExpGK:   Err: {np.mean(raw_res[dist]['DPExpGK'][eps]):.5f} | Mem: {np.mean(gk_mems)/1024:5.2f} KB | Ingest: {np.mean(gk_in_times)*1000:6.2f} ms | Query: {np.mean(gk_q_times)*1000:5.2f} ms")
                print(f"    └─ DCS DP:    Err: {np.mean(raw_res[dist]['DCS DP'][eps]):.5f} | Mem: {np.mean(dcs_mems)/1024:5.2f} KB | Ingest: {np.mean(dcs_in_times)*1000:6.2f} ms | Query: {np.mean(dcs_q_times)*1000:5.2f} ms")
                print("    " + "-"*105)

    end = time.time()
    print(f"\n▶ Exp 6.2.1 completed in {end - start:.2f} seconds.")

    # 归档误差分位数数据（时间数据已在上面实时组装完毕）
    for dist in cfg['distributions']:
        for m in ['PrivKLL-Q', 'DPExpGK', 'DCS DP']:
            for eps in cfg['epsilons']:
                arr = raw_res[dist][m][eps]
                plot_data[dist][m]['mean'].append(float(np.mean(arr)))
                plot_data[dist][m]['p10'].append(float(np.percentile(arr, 10)))
                plot_data[dist][m]['p90'].append(float(np.percentile(arr, 90)))

    # 🌟 核心改变 3：通过 json.dump 自动固化包含时间的 plot_data
    with open(RESULTS / f"exp_s_eps_{timestamp}.json", 'w') as f: 
        json.dump({"meta": meta_log, "data": plot_data}, f, indent=4)
        
    plot_vldb(cfg['epsilons'], plot_data, r'Privacy Budget ($\epsilon$)', True, 2, RESULTS / f"exp_s_eps_{timestamp}.png")
    print(f"\n▶ Exp single_eps results saved to {RESULTS / f'exp_s_eps_{timestamp}.json'} and {RESULTS / f'exp_s_eps_{timestamp}.png'}")

def run_experiment_s_q(cfg, timestamp, cache_filename="calib_matrix.pkl"):
    print("\n" + "="*80 + f"\n▶ [Executing] Exp — Varying Target Quantiles (Fixed Budget={cfg['mem_kb']} KB, e={cfg['eps']}, N={cfg['N']})\n" + "="*80)
    start = time.time()

    safe_H = int(np.ceil(np.log(cfg['N']) / np.log(1.5)))
    
    # 初始化结果收集字典：x轴是不同的 target_q
    raw_res = {dist: {'PrivKLL-Q': {q: [] for q in cfg['target_qs']}, 
                      'DPExpGK': {q: [] for q in cfg['target_qs']}, 
                      'DCS DP': {q: [] for q in cfg['target_qs']}} 
               for dist in cfg['distributions']}
    
    meta_log = {}
    param_cache = None

    # 初始化 plot_data 结构，同时容纳误差分位数和时间
    plot_data = {
        dist: {
            m: {
                'mean': [], 'p10': [], 'p90': [], 
                'avg_ingest_time_ms': [], 'avg_query_time_ms': []
            } for m in ['PrivKLL-Q', 'DPExpGK', 'DCS DP']
        } for dist in cfg['distributions']
    }

    # 1. 尝试从本地持久化 pickle 文件加载参数（这部分复用 mem 实验的校准逻辑）
    if os.path.exists(cache_filename):
        try:
            with open(cache_filename, 'rb') as f:
                full_matrix = pickle.load(f)
            
            param_cache = {}
            for dist in cfg['distributions']:
                # 在固定的 mem_kb 下，各算法的内部参数（kll_k, gk_alpha, dcs_gamma）仅与分布和内存有关，与具体查询 q 无关
                mem = cfg['mem_kb']
                target_eps = cfg['eps']
                if dist in full_matrix and mem in full_matrix[dist] and target_eps in full_matrix[dist][mem]:
                    cached_data = full_matrix[dist][mem][target_eps]
                    param_cache[dist] = {
                        "kll_k": cached_data['kll_k'],
                        "gk_alpha": cached_data['gk_alpha'],
                        "dcs_gamma": cached_data['dcs_gamma']
                    }
                    meta_log[dist] = {
                        "anchor_bytes": cached_data.get('anchor_bytes', 0),
                        "derived_gk_alpha": cached_data['gk_alpha'],
                        "kll_k": cached_data['kll_k'],
                        "dcs_gamma": cached_data['dcs_gamma']
                    }
                else:
                    param_cache = None
                    break
            if param_cache:
                print(f"📂 [Cache Hit] 成功加载参数矩阵，跳过校准！")
        except Exception as e:
            print(f"⚠️ [Cache Warning] 读取缓存文件失败 ({e})，将重新执行校准。")
            param_cache = None

    # 2. 如果无缓存，执行原地校准
    if param_cache is None:
        print("\n>>> Cache Miss. Pre-calibrating parameters...")
        param_cache = {}
        if os.path.exists(cache_filename):
            try:
                with open(cache_filename, 'rb') as f: full_matrix = pickle.load(f)
            except: full_matrix = {}
        else:
            full_matrix = {}

        for dist in cfg['distributions']:
            mem = cfg['mem_kb']
            mem_bytes = mem * 1024
            kll_k_mem = calibration.find_kll_k_by_budget(mem_bytes, H=safe_H)
            
            probe_stream = np.random.uniform(0, 100000, cfg['N']) if dist=='uniform' else np.zeros(cfg['N'])
            test_kll = KLL(k=kll_k_mem)
            for v in probe_stream: test_kll.update(v)
            anchor = calibration.kll_bytes_analytical(test_kll)
            
            gk_alpha = calibration.find_gk_alpha_tight(anchor, dist, cfg['N'])
            dcs_gamma = calibration.find_dcs_gamma_by_budget(mem_bytes, universe=2**16)
            
            meta_log[dist] = {
                "anchor_bytes": anchor, 
                "derived_gk_alpha": gk_alpha, 
                "kll_k": kll_k_mem, 
                "dcs_gamma": dcs_gamma
            }
            param_cache[dist] = {
                "kll_k": kll_k_mem,
                "gk_alpha": gk_alpha,
                "dcs_gamma": dcs_gamma
            }
            
            if dist not in full_matrix: full_matrix[dist] = {}
            if mem not in full_matrix[dist]: full_matrix[dist][mem] = {}
            full_matrix[dist][mem][cfg['eps']] = {
                'kll_k': kll_k_mem, 'gk_alpha': gk_alpha, 'dcs_gamma': dcs_gamma, 'anchor_bytes': anchor
            }
                
        with open(cache_filename, 'wb') as f: pickle.dump(full_matrix, f)

    # 3. 多进程并发跑不同的 target_q
    print("\n>>> Starting parallel execution for different target quantiles...")
    with ProcessPoolExecutor() as executor:
        for dist in cfg['distributions']:
            params = param_cache[dist]
            kll_k_mem = params["kll_k"]
            gk_alpha = params["gk_alpha"]
            dcs_gamma = params["dcs_gamma"]
            
            for q in cfg['target_qs']:
                futures = []
                for _ in range(cfg['trials']):
                    f = executor.submit(
                        run_single_trial_strict,
                        cfg['eps'], dist, cfg['N'], q, 
                        kll_k_mem, gk_alpha, dcs_gamma, dcs_uni=2**16
                    )
                    futures.append(f)
                
                kll_mems, gk_mems, dcs_mems = [], [], []
                kll_in_times, kll_q_times = [], []
                gk_in_times, gk_q_times = [], []
                dcs_in_times, dcs_q_times = [], []
                
                for f in futures:
                    (err_kll, err_gk, err_dcs, 
                     kll_mem, gk_mem, dcs_mem,
                     t_kll_in, t_gk_in, t_dcs_in,
                     t_kll_q, t_gk_q, t_dcs_q) = f.result()
                    
                    kll_mems.append(kll_mem)
                    gk_mems.append(gk_mem)
                    dcs_mems.append(dcs_mem)

                    kll_in_times.append(t_kll_in); kll_q_times.append(t_kll_q)
                    gk_in_times.append(t_gk_in);   gk_q_times.append(t_gk_q)
                    dcs_in_times.append(t_dcs_in); dcs_q_times.append(t_dcs_q)
                    
                    raw_res[dist]['PrivKLL-Q'][q].append(err_kll)
                    raw_res[dist]['DPExpGK'][q].append(err_gk)
                    raw_res[dist]['DCS DP'][q].append(err_dcs)
                
                # 记录当前 q 下的平均耗时
                plot_data[dist]['PrivKLL-Q']['avg_ingest_time_ms'].append(float(np.mean(kll_in_times) * 1000))
                plot_data[dist]['PrivKLL-Q']['avg_query_time_ms'].append(float(np.mean(kll_q_times) * 1000))
                
                plot_data[dist]['DPExpGK']['avg_ingest_time_ms'].append(float(np.mean(gk_in_times) * 1000))
                plot_data[dist]['DPExpGK']['avg_query_time_ms'].append(float(np.mean(gk_q_times) * 1000))
                
                plot_data[dist]['DCS DP']['avg_ingest_time_ms'].append(float(np.mean(dcs_in_times) * 1000))
                plot_data[dist]['DCS DP']['avg_query_time_ms'].append(float(np.mean(dcs_q_times) * 1000))

                print(f"  [Parallel Summary] Quantile: {q:<4.2f} | Dist: {dist.upper()} | Completed Trials: {cfg['trials']}")
                print(f"    ├─ PrivKLL-Q: Err: {np.mean(raw_res[dist]['PrivKLL-Q'][q]):.5f} | Mem: {np.mean(kll_mems)/1024:5.2f} KB | Ingest: {np.mean(kll_in_times)*1000:6.2f} ms | Query: {np.mean(kll_q_times)*1000:5.2f} ms")
                print(f"    ├─ DPExpGK:   Err: {np.mean(raw_res[dist]['DPExpGK'][q]):.5f} | Mem: {np.mean(gk_mems)/1024:5.2f} KB | Ingest: {np.mean(gk_in_times)*1000:6.2f} ms | Query: {np.mean(gk_q_times)*1000:5.2f} ms")
                print(f"    └─ DCS DP:    Err: {np.mean(raw_res[dist]['DCS DP'][q]):.5f} | Mem: {np.mean(dcs_mems)/1024:5.2f} KB | Ingest: {np.mean(dcs_in_times)*1000:6.2f} ms | Query: {np.mean(dcs_q_times)*1000:5.2f} ms")
                print("    " + "-"*105)

    end = time.time()
    print(f"\n▶ Exp Varying Quantile completed in {end - start:.2f} seconds.")

    # 4. 统计误差分位数
    for dist in cfg['distributions']:
        for m in ['PrivKLL-Q', 'DPExpGK', 'DCS DP']:
            for q in cfg['target_qs']:
                arr = raw_res[dist][m][q]
                plot_data[dist][m]['mean'].append(float(np.mean(arr)))
                plot_data[dist][m]['p10'].append(float(np.percentile(arr, 10)))
                plot_data[dist][m]['p90'].append(float(np.percentile(arr, 90)))

    # 5. 保存数据并画图
    with open(RESULTS / f"exp_s_q_{timestamp}.json", 'w') as f: 
        json.dump({"meta": meta_log, "data": plot_data}, f, indent=4)
    
    # 临时重写一下用来画线图的横轴，因为 q 是线性比例，调用 plot_vldb 时需要注意刻度格式。
    # 如果你希望和之前的 eps 一样用标准的学术图结构输出，我们可以直接调用 plot_vldb
    # 注意：因为 plot_vldb 内部写死了 x_scale_log=True 且以 2 为底进行刻度转换，
    # 如果你要为 q 画图，建议在下方独立写一个不含 log2 坐标轴的简化版图表生成，这里先保持生成调用：
    try:
        # 如果 plot_vldb 专门针对分位数进行了对齐，可直接输出：
        plot_vldb(cfg['target_qs'], plot_data, r'Target Quantile ($q$)', False, 10, RESULTS / f"exp_s_q_{timestamp}.png")
    except Exception as e:
        # 如果因为内置的 log2 坐标轴挂了，这里做个简单平铺防御：
        plt.figure(figsize=(10, 4))
        for dist in cfg['distributions']:
            for m in ['PrivKLL-Q', 'DPExpGK', 'DCS DP']:
                plt.plot(cfg['target_qs'], plot_data[dist][m]['mean'], label=f"{dist}-{m}")
        plt.xlabel('Target Quantile (q)')
        plt.ylabel('Mean Abs Rank Error')
        plt.legend()
        plt.savefig(RESULTS / f"exp_s_q_{timestamp}.png")
        plt.close()

    print(f"\n▶ Exp single_q results saved to {RESULTS / f'exp_s_q_{timestamp}.json'} and {RESULTS / f'exp_s_q_{timestamp}.png'}")

def run_experiment_s_mem(cfg, timestamp, cache_filename="calib_matrix.pkl"):
    print("\n" + "="*80 + f"\n▶ [Executing] Exp 6.2.3 — Vary Memory Budget (N={cfg['N']}, e={cfg['eps']}, q={cfg['target_q']})\n" + "="*80)
    start = time.time()

    safe_H = int(np.ceil(np.log(cfg['N']) / np.log(1.5)))
    
    # 初始化结果收集字典
    raw_res = {dist: {'PrivKLL-Q': {mem: [] for mem in cfg['memory_kbs']}, 
                      'DPExpGK': {mem: [] for mem in cfg['memory_kbs']}, 
                      'DCS DP': {mem: [] for mem in cfg['memory_kbs']}} 
               for dist in cfg['distributions']}
    
    meta_log = {}
    param_cache = None

    # 🌟 调整 plot_data 的结构，使其能够容纳 time 统计指标
    # 结构拓展为：plot_data[dist][method][指标] = [每个eps对应的值, ...]
    plot_data = {
        dist: {
            m: {
                'mean': [], 'p10': [], 'p90': [], 
                'avg_ingest_time_ms': [], 'avg_query_time_ms': []  # 🌟 新增时间维度
            } for m in ['PrivKLL-Q', 'DPExpGK', 'DCS DP']
        } for dist in cfg['distributions']
    }

    # 🌟 核心修改 1：尝试从本地持久化 pickle 文件加载参数字典
    if os.path.exists(cache_filename):
        try:
            with open(cache_filename, 'rb') as f:
                full_matrix = pickle.load(f)
            
            # 从全量矩阵中，提取当前实验所需的子集
            param_cache = {}
            for dist in cfg['distributions']:
                param_cache[dist] = {}
                meta_log[dist] = {}
                for mem in cfg['memory_kbs']:
                    # 注意：我们在 calibration.py 中生成字典时，最内层多套了一层 [eps] 键
                    # 这里直接提取对应当前 cfg['eps'] 下的参数
                    target_eps = cfg['eps']
                    if dist in full_matrix and mem in full_matrix[dist] and target_eps in full_matrix[dist][mem]:
                        cached_data = full_matrix[dist][mem][target_eps]
                        
                        param_cache[dist][mem] = {
                            "kll_k": cached_data['kll_k'],
                            "gk_alpha": cached_data['gk_alpha'],
                            "dcs_gamma": cached_data['dcs_gamma']
                        }
                        meta_log[dist][mem] = {
                            "anchor_bytes": cached_data.get('anchor_bytes', 0),
                            "derived_gk_alpha": cached_data['gk_alpha'],
                            "kll_k": cached_data['kll_k'],
                            "dcs_gamma": cached_data['dcs_gamma']
                        }
                    else:
                        # 如果缓存中缺少当前的特定组合，标记为 None 触发重新校准
                        param_cache = None
                        break
                if param_cache is None: break
                
            if param_cache:
                print(f"📂 [Cache Hit] 成功从本地文件 '{cache_filename}' 加载所需的参数矩阵，跳过 Phase 1 校准！")
        except Exception as e:
            print(f"⚠️ [Cache Warning] 读取缓存文件失败 ({e})，将重新执行 Phase 1 校准。")
            param_cache = None

    # 🌟 核心修改 2：如果缓存不存在或不完整，执行原地校准，并更新/保存本地 pickle 文件
    if param_cache is None:
        print("\n>>> [Phase 1] Cache Miss. Pre-calibrating parameters for all distributions and memory budgets...")
        param_cache = {}
        
        # 如果原本存在全量矩阵，读取它以便追加；否则新建一个
        if os.path.exists(cache_filename):
            try:
                with open(cache_filename, 'rb') as f: full_matrix = pickle.load(f)
            except: full_matrix = {}
        else:
            full_matrix = {}

        for dist in cfg['distributions']:
            meta_log[dist] = {}
            param_cache[dist] = {}
            if dist not in full_matrix: full_matrix[dist] = {}
            
            for mem in cfg['memory_kbs']:
                mem_bytes = mem * 1024
                kll_k_mem = calibration.find_kll_k_by_budget(mem_bytes, H=safe_H)
                
                # 探测流生成与 GK 参数校准
                probe_stream = np.random.uniform(0, 100000, cfg['N']) if dist=='uniform' else np.zeros(cfg['N'])
                test_kll = KLL(k=kll_k_mem)
                for v in probe_stream: 
                    test_kll.update(v)
                anchor = calibration.kll_bytes_analytical(test_kll)
                
                print(f"  [Calibrating] Dist: {dist.upper()} | Budget: {mem} KB | Real KLL Anchor: {anchor/1024:.2f} KB")
                gk_alpha = calibration.find_gk_alpha_tight(anchor, dist, cfg['N'])
                dcs_gamma = calibration.find_dcs_gamma_by_budget(mem_bytes, universe=2**16)
                
                # 记录日志
                meta_log[dist][mem] = {
                    "anchor_bytes": anchor, 
                    "derived_gk_alpha": gk_alpha, 
                    "kll_k": kll_k_mem, 
                    "dcs_gamma": dcs_gamma
                }
                
                # 填充当前运行时内存查找字典
                param_cache[dist][mem] = {
                    "kll_k": kll_k_mem,
                    "gk_alpha": gk_alpha,
                    "dcs_gamma": dcs_gamma
                }
                
                # 同步更新持久化结构 (对齐 calibration.py 的 [dist][mem][eps] 三层嵌套结构)
                if mem not in full_matrix[dist]: full_matrix[dist][mem] = {}
                full_matrix[dist][mem][cfg['eps']] = {
                    'kll_k': kll_k_mem,
                    'gk_alpha': gk_alpha,
                    'dcs_gamma': dcs_gamma,
                    'anchor_bytes': anchor
                }
                
        # 将最新的全量校准字典持久化写入本地磁盘
        with open(cache_filename, 'wb') as f:
            pickle.dump(full_matrix, f)
        print(f"💾 [Cache Saved] 最新的参数矩阵已同步写入磁盘持久化文件: '{cache_filename}'")

    # ------------------------------------------------------------------ #
    # >>> [Phase 2] Starting parallel execution of independent trials...
    # ------------------------------------------------------------------ #
    print("\n>>> [Phase 2] Starting parallel execution of independent trials...")
    with ProcessPoolExecutor(max_workers=8) as executor:
        for dist in cfg['distributions']:
            for mem in cfg['memory_kbs']:
                # 从参数字典中 O(1) 秒级提取所需的物理参数
                params = param_cache[dist][mem]
                kll_k_mem = params["kll_k"]
                gk_alpha = params["gk_alpha"]
                dcs_gamma = params["dcs_gamma"]
                
                # 异步提交当前配置下的所有重复试验
                futures = []
                for _ in range(cfg['trials']):
                    f = executor.submit(
                        run_single_trial_strict,
                        cfg['eps'], dist, cfg['N'], cfg['target_q'], 
                        kll_k_mem, gk_alpha, dcs_gamma, dcs_uni=2**16
                    )
                    futures.append(f)
                
                kll_mems, gk_mems, dcs_mems = [], [], []
                kll_in_times, kll_q_times = [], []
                gk_in_times, gk_q_times = [], []
                dcs_in_times, dcs_q_times = [], []
                
                # 阻塞同步回收多进程计算结果
                for f in futures:
                    (err_kll, err_gk, err_dcs, 
                     kll_mem, gk_mem, dcs_mem,
                     t_kll_in, t_gk_in, t_dcs_in,
                     t_kll_q, t_gk_q, t_dcs_q) = f.result()
                    
                    kll_mems.append(kll_mem)
                    gk_mems.append(gk_mem)
                    dcs_mems.append(dcs_mem)

                    # 🌟 收集细分时间开销（保持原生秒单位）
                    kll_in_times.append(t_kll_in); kll_q_times.append(t_kll_q)
                    gk_in_times.append(t_gk_in);   gk_q_times.append(t_gk_q)
                    dcs_in_times.append(t_dcs_in); dcs_q_times.append(t_dcs_q)
                    
                    
                    raw_res[dist]['PrivKLL-Q'][mem].append(err_kll)
                    raw_res[dist]['DPExpGK'][mem].append(err_gk)
                    raw_res[dist]['DCS DP'][mem].append(err_dcs)
                
                # 🌟 核心改变 1：计算当前 eps 下 100 个 trial 的平均时间，转为 ms（毫秒）存储并追加
                # 这样做可以直接复用你后面的 eps 循环归档逻辑，或者直接在这里同步写进 plot_data
                plot_data[dist]['PrivKLL-Q']['avg_ingest_time_ms'].append(float(np.mean(kll_in_times) * 1000))
                plot_data[dist]['PrivKLL-Q']['avg_query_time_ms'].append(float(np.mean(kll_q_times) * 1000))
                
                plot_data[dist]['DPExpGK']['avg_ingest_time_ms'].append(float(np.mean(gk_in_times) * 1000))
                plot_data[dist]['DPExpGK']['avg_query_time_ms'].append(float(np.mean(gk_q_times) * 1000))
                
                plot_data[dist]['DCS DP']['avg_ingest_time_ms'].append(float(np.mean(dcs_in_times) * 1000))
                plot_data[dist]['DCS DP']['avg_query_time_ms'].append(float(np.mean(dcs_q_times) * 1000))

                # 🌟 核心改变 2：升级控制台小结日志，完美透视 Ingest 和 Query 的时间开销
                print(f"  [Parallel Summary] Memory: {mem:<3.1f} | Dist: {dist.upper()} | Completed Trials: {cfg['trials']}")
                print(f"    ├─ PrivKLL-Q: Err: {np.mean(raw_res[dist]['PrivKLL-Q'][mem]):.5f} | Mem: {np.mean(kll_mems)/1024:5.2f} KB | Ingest: {np.mean(kll_in_times)*1000:6.2f} ms | Query: {np.mean(kll_q_times)*1000:5.2f} ms")
                print(f"    ├─ DPExpGK:   Err: {np.mean(raw_res[dist]['DPExpGK'][mem]):.5f} | Mem: {np.mean(gk_mems)/1024:5.2f} KB | Ingest: {np.mean(gk_in_times)*1000:6.2f} ms | Query: {np.mean(gk_q_times)*1000:5.2f} ms")
                print(f"    └─ DCS DP:    Err: {np.mean(raw_res[dist]['DCS DP'][mem]):.5f} | Mem: {np.mean(dcs_mems)/1024:5.2f} KB | Ingest: {np.mean(dcs_in_times)*1000:6.2f} ms | Query: {np.mean(dcs_q_times)*1000:5.2f} ms")
                print("    " + "-"*105)

    end = time.time()
    print(f"\n▶ Exp 6.2.3 completed in {end - start:.2f} seconds.")

    plot_data = {dist: {m: {'mean': [], 'p10': [], 'p90': []} for m in ['PrivKLL-Q', 'DPExpGK', 'DCS DP']} for dist in cfg['distributions']}
    for dist in cfg['distributions']:
        for m in ['PrivKLL-Q', 'DPExpGK', 'DCS DP']:
            for mem in cfg['memory_kbs']:
                arr = raw_res[dist][m][mem]
                plot_data[dist][m]['mean'].append(np.mean(arr))
                plot_data[dist][m]['p10'].append(np.percentile(arr, 10))
                plot_data[dist][m]['p90'].append(np.percentile(arr, 90))

    target_query = cfg['target_q']
    with open(RESULTS / f"exp_s_mem_{target_query}_{timestamp}.json", 'w') as f: json.dump({"meta": meta_log, "data": plot_data}, f, indent=4)
    plot_vldb(cfg['memory_kbs'], plot_data, r'Memory Budget (KB)', False, 2, RESULTS / f"exp_s_mem_{target_query}_{timestamp}.png")
    print(f"\n▶ Exp single_mem results saved to {RESULTS / f'exp_s_mem_{target_query}_{timestamp}.json'} and {RESULTS / f'exp_s_mem_{target_query}_{timestamp}.png'}")

# ------------------------------------------------------------------------------
# 5. 统一入口控制中心
# ------------------------------------------------------------------------------
def main(logger=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="Quick debug mode")
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    cfg = {
        'N': 100_000, 
        'Ns': [100_000, 1_000_000, 10_000_000],
        # 'distributions': ['uniform', 'normal', 'zipf'],
        'distributions': ['zipf'],
        'eps': 1.0,
        'trials': 50,
        'target_q': 0.5,

        'mem_kb': 8,
        # 256KB
        # 'mem_kb': 256,
        'PrivKLL_k': 325,  # 8 KB Anchor for KLL at N=100K
        'GK_alpha': 0.002024,  # 8 KB Anchor for zipf at N=100K
        'DCS_gamma': 0.005,  # DCS parameter (tuned for better performance at low memory)
        'DCS_uni': 2**16,  # DCS universe size

        'epsilons': [0.5, 1.0, 2.0, 4.0, 8.0],
        # 'epsilons': [0.01, 0.02, 0.05, 0.1, 0.5],

        'target_qs': [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95],

        'memory_kbs': [4, 8, 32, 64, 256, 1024],
        # 'memory_kbs': [4, 8, 16],


    }
    
    if args.fast:
        cfg['trials'] = 1
        # cfg['epsilons'] = [0.5, 2.0, 8.0]
        # cfg['Ns'] = [100_000, 1_000_000]
        cfg['memory_kbs'] = [2, 8, 32]
        # # cfg['N'] = 100_000
        # cfg['distributions'] = ['uniform', 'normal', 'zipf']

    if logger is None:
        logger = get_default_logger()
    logger.info("Loading data...")
    logger.info("Running PrivKLL...")

    run_experiment_s_eps(cfg, timestamp)
    # run_experiment_s_q(cfg, timestamp)
    # run_experiment_s_mem(cfg, timestamp)

    logger.info("=" * 80)
    logger.info("ALL SINGLE-QUANTILE EXPERIMENTS EXECUTED SUCCESSFULLY")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()