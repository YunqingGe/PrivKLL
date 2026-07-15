# ------------------------------------------------------------------------------
# 支持 one-time release 的严格单次物理试验 —— 固定 N 随 Mem 变化实验
# 1*5 列并排学术大图输出，参数根据 Target Mem 在线对齐，文件以 N 结尾
# ------------------------------------------------------------------------------
import argparse
from concurrent.futures import ProcessPoolExecutor
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

# 导入您提供的校准核心算法
from DPQE.exp import calibration
from DPQE.src.PrivKLL import PrivKLL
from DPQE.src.kll import KLL
from DPQE.src.Greenwald_Khanna import GK
from DPQE.src.DCS import DCS

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

def get_default_logger():
    logger = logging.getLogger("exp_mem_fixed_n")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler("exp_mem_fixed_n.log")
        ch = logging.StreamHandler()
        fmt = logging.Formatter("[%(asctime)s] %(message)s")
        fh.setFormatter(fmt)
        ch.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger

# ------------------------------------------------------------------------------
# 1. 严格单次物理试验核心（物理流计算）
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
    dp_kll = PrivKLL(epsilon=eps, kll_k=kll_k)
    non_dp_kll = KLL(k=kll_k)
    gk_sketch = GK(alpha=gk_alpha)
    
    delta = 1e-6
    L = math.log(1.0 / delta)
    sqrt_L_plus_eps = math.sqrt(L + eps)
    sqrt_L = math.sqrt(L)
    rho_cdp = (sqrt_L_plus_eps - sqrt_L) ** 2
    dcs_sketch = DCS(universe=dcs_uni, gamma=dcs_gamma, rho=rho_cdp)

    sample_size = min(total_n, 100_000)
    reservoir = []
    
    def to_dcs_domain(val): return int(np.clip(val / 100000.0 * (dcs_uni - 1), 0, dcs_uni - 1))
    def from_dcs_domain(int_val): return (int_val / (dcs_uni - 1)) * 100000.0

    generator = stream_chunk_generator(dist_type, total_n, chunk_size=chunk_size)
    item_counter = 0
    
    t_kll_ingest, t_gk_ingest, t_dcs_ingest = 0.0, 0.0, 0.0

    for chunk in generator:
        for val in chunk:
            non_dp_kll.update(val)
            t0 = time.perf_counter()
            dp_kll.ingest_data(val)
            t_kll_ingest += (time.perf_counter() - t0)
            
            t0 = time.perf_counter()
            gk_sketch.insert(val)
            t_gk_ingest += (time.perf_counter() - t0)
            
            t0 = time.perf_counter()
            dcs_val = to_dcs_domain(val)
            dcs_sketch.update(dcs_val)
            t_dcs_ingest += (time.perf_counter() - t0)
            
            if len(reservoir) < sample_size:
                reservoir.append(val)
            else:
                r = np.random.randint(0, item_counter + 1)
                if r < sample_size: reservoir[r] = val
            item_counter += 1

    t0 = time.perf_counter()
    est_dp_kll = dp_kll.release_quantile(target_q)
    t_kll_query = time.perf_counter() - t0

    t0 = time.perf_counter()
    est_gk_exp = gk_sketch.dp_exp(target_q, eps)
    if est_gk_exp is None: est_gk_exp = 0.0
    t_gk_query = time.perf_counter() - t0

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

    res_arr = np.array(reservoir)
    err_kll = abs((res_arr <= est_dp_kll).mean() - target_q)
    err_gk = abs((res_arr <= est_gk_exp).mean() - target_q)
    err_dcs = abs((res_arr <= est_dcs).mean() - target_q)

    kll_mem = sum(len(c) for c in non_dp_kll.compactors) * 8
    gk_mem = len(gk_sketch.S) * 16
    dcs_mem = dcs_sketch.memory_budget() * 4
    
    return (err_kll, err_gk, err_dcs, 
            kll_mem, gk_mem, dcs_mem,
            t_kll_ingest, t_gk_ingest, t_dcs_ingest,
            t_kll_query, t_gk_query, t_dcs_query)

# ------------------------------------------------------------------------------
# 2. 统一学术论文级 1 * 6 多子图硬定位绘图引擎
# ------------------------------------------------------------------------------
def plot_vldb_1x6_transposed(epsilons, memory_kbs, plot_data, xlabel_str, fname):
    plt.rcParams.update({
        'font.size': 8.0, 
        'font.family': 'serif', 
        'mathtext.fontset': 'cm',
        'xtick.labelsize': 6.5,    # 稍微压缩字体，使 6 子图平铺更美观
        'ytick.labelsize': 7.5
    })
    
    # 强制选取 6 个内存预算映射到 6 个物理子图
    selected_mems = memory_kbs[:6]
    
    fig, axes = plt.subplots(1, 6, figsize=(7.2, 1.7), sharey=True)
    sub_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']

    # 🌟 双层安全防御：首先完全隐藏所有初始子图
    for ax in axes: 
        ax.set_visible(False)

    style_cfgs = {
        'PrivKLL-Q': {'color':  "#3A86FF", 'linestyle': '--', 'marker': 'v'},
        'DPExpGK': {'color': "#06D6A0", 'linestyle': '-', 'marker': 'o'},
        'DCS DP': {'color': "#F77F00FF", 'linestyle': '-.', 'marker': '^'}
    }
    lines, labels = [], []

    # 🌟 1x6 学术紧凑型盒模型定位参数
    box_width = 0.118   
    box_height = 0.52
    box_bottom = 0.32
    box_lefts = [0.075 + i * (box_width + 0.032) for i in range(6)]

    for idx, mem in enumerate(selected_mems):
        ax = axes[idx]
        ax.set_visible(True) # 🌟 显现当前已被正确指派数据的物理盒子
        ax.set_position([box_lefts[idx], box_bottom, box_width, box_height])
        
        for m in ['PrivKLL-Q', 'DPExpGK', 'DCS DP']:
            means, p10s, p90s = [], [], []
            
            # 🌟【核心修复点】：锁死生成 JSON 时的绝对满状态序列
            # 这样无论你的 main 传入什么切片，都能精准从 JSON 数组中抽取出真实的物理数据，绝不发生错位
            full_json_sequence = [4, 8, 32, 64, 256, 1024]
            try:
                mem_idx = full_json_sequence.index(mem)
            except ValueError:
                mem_idx = idx
            
            for eps in epsilons:
                # 兼容历史遗留数据，强转字符串键读取
                eps_str = str(eps)
                means.append(plot_data[eps_str][m]['mean'][mem_idx])
                p10s.append(plot_data[eps_str][m]['p10'][mem_idx])
                p90s.append(plot_data[eps_str][m]['p90'][mem_idx])
                
            means = np.clip(np.array(means), 1e-7, None)
            p10s = np.clip(np.array(p10s), 1e-7, None)
            p90s = np.clip(np.array(p90s), 1e-7, None)
            
            line, = ax.plot(range(len(epsilons)), means, color=style_cfgs[m]['color'], linestyle=style_cfgs[m]['linestyle'], 
                             marker=style_cfgs[m]['marker'], linewidth=1.1, markersize=2.8, zorder=4)
            ax.fill_between(range(len(epsilons)), p10s, p90s, color=style_cfgs[m]['color'], alpha=0.06)
            
            if idx == 0: 
                lines.append(line)
                labels.append(m)
        
        ax.set_xticks(range(len(epsilons)))
        ax.set_xticklabels([f"{e}" for e in epsilons], fontsize=6.0)
    
        ax.set_yscale('log', base=2)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: f"$2^{{{int(np.log2(y))}}}$" if y > 0 else "0"))
        
        ax.set_title(f"{sub_labels[idx]} {mem} KB", y=-0.48, fontsize=7.5, va='top')
        ax.grid(True, which="both", ls="--", color='#e0e0e0', alpha=0.6, linewidth=0.4)
        
    fig.text(0.5, 0.18, xlabel_str, ha='center', va='center', fontsize=8.0)
    axes[0].set_ylabel('Mean Abs Rank Error', fontsize=8.0, labelpad=2)
        
    fig.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98), 
               ncol=3, frameon=True, fancybox=True, shadow=False, 
               fontsize=7, columnspacing=1.0, handletextpad=0.3, framealpha=1.0)
    
    plt.savefig(fname, dpi=300)
    plt.close()

# ------------------------------------------------------------------------------
# 3. 核心实验流程控制逻辑（按 Target Mem 实时校准并对齐）
# ------------------------------------------------------------------------------
def run_mem_experiment_fixed_n(cfg):
    target_n = cfg['N']
    print("\n" + "="*80 + f"\n▶ [Executing] Exp Fixed N — Varying Memory Budget (N={target_n}, Trials={cfg['trials']})\n" + "="*80)
    
    # 定义各文件的 N 结尾命名规则
    cache_filename = f"calib_matrix_N{target_n}.pkl"
    json_filename = RESULTS / f"exp_s_mem_fixed_N{target_n}.json"
    png_filename = RESULTS / f"exp_s_mem_fixed_N{target_n}.png"

    safe_H = int(np.ceil(np.log(target_n) / np.log(1.5)))
    dist = cfg['distributions'][0] # 'zipf'

    # ------------------------------------------------------------------ #
    # Phase 1: 依据内存对齐校准三大算法参数（加入本地 Pickle 缓存调用防御机制）
    # ------------------------------------------------------------------ #
    param_cache = {}
    if os.path.exists(cache_filename):
        try:
            with open(cache_filename, 'rb') as f:
                param_cache = pickle.load(f)
            print(f"📂 [Cache Hit] 成功从本地文件 '{cache_filename}' 加载校准参数，跳过校准流程。")
        except Exception as e:
            print(f"⚠️ [Cache Warning] 读取缓存失败 ({e})，将重新执行 Phase 1 校准。")
            param_cache = {}

    if not param_cache:
        print("\n>>> [Phase 1] Cache Miss. Pre-calibrating parameters based on memory budgets...")
        for mem in cfg['memory_kbs']:
            mem_bytes = int(mem * 1024)
            
            # 1. 逆向推导 KLL 
            kll_k_mem = calibration.find_kll_k_by_budget(mem_bytes, H=safe_H)
            
            # 2. 模拟流并计算真实的理论物理字节锚点 (Anchor)
            probe_stream = np.clip(np.random.pareto(a=2.0, size=min(50000, target_n)) * 10000.0, 0, 100000)
            test_kll = KLL(k=kll_k_mem)
            for v in probe_stream: test_kll.update(v)
            anchor_bytes = calibration.kll_bytes_analytical(test_kll)
            
            # 3. 通过 Anchor 对齐 GK 紧致边界并计算 DCS gamma
            gk_alpha = calibration.find_gk_alpha_tight(anchor_bytes, dist, target_n)
            dcs_gamma = calibration.find_dcs_gamma_by_budget(mem_bytes, universe=2**16)
            
            param_cache[mem] = {
                "kll_k": kll_k_mem,
                "gk_alpha": gk_alpha,
                "dcs_gamma": dcs_gamma,
                "anchor_kb": anchor_bytes / 1024
            }
            print(f"  [Calibrated Budget] Mem: {mem} KB -> KLL_k: {kll_k_mem}, GK_α: {gk_alpha:.6f}, DCS_γ: {dcs_gamma if dcs_gamma else 0:.6f}")
        
        with open(cache_filename, 'wb') as f:
            pickle.dump(param_cache, f)
        print(f"💾 [Cache Saved] 参数对齐矩阵已成功写入磁盘持久化文件: '{cache_filename}'")

    # ------------------------------------------------------------------ #
    # Phase 2: 并发执行严格物理试验
    # ------------------------------------------------------------------ #
    # 初始化作图数据字典结构：plot_data[eps][method][metric] = [对应每个mem的序列值]
    plot_data = {
        eps: {
            m: {'mean': [], 'p10': [], 'p90': [], 'avg_ingest_time_ms': [], 'avg_query_time_ms': []}
            for m in ['PrivKLL-Q', 'DPExpGK', 'DCS DP']
        } for eps in cfg['epsilons']
    }

    with ProcessPoolExecutor() as executor:
        for eps in cfg['epsilons']:
            print(f"\n>>> 🚀 Streaming Parallel Engine for Epsilon = {eps}")
            
            for mem in cfg['memory_kbs']:
                params = param_cache[mem]
                
                futures = []
                for _ in range(cfg['trials']):
                    f = executor.submit(
                        run_single_trial_strict,
                        eps, dist, target_n, cfg['target_q'], 
                        params["kll_k"], params["gk_alpha"], params["dcs_gamma"], dcs_uni=2**16
                    )
                    futures.append(f)
                
                err_kll_list, err_gk_list, err_dcs_list = [], [], []
                kll_in_times, kll_q_times = [], []
                gk_in_times, gk_q_times = [], []
                dcs_in_times, dcs_q_times = [], []
                
                for f in futures:
                    (err_kll, err_gk, err_dcs, _, _, _,
                     t_kll_in, t_gk_in, t_dcs_in, t_kll_q, t_gk_q, t_dcs_q) = f.result()
                    
                    err_kll_list.append(err_kll); err_gk_list.append(err_gk); err_dcs_list.append(err_dcs)
                    kll_in_times.append(t_kll_in); kll_q_times.append(t_kll_q)
                    gk_in_times.append(t_gk_in);   gk_q_times.append(t_gk_q)
                    dcs_in_times.append(t_dcs_in); dcs_q_times.append(t_dcs_q)
                
                # 记录误差分位数
                for m, err_arr in zip(['PrivKLL-Q', 'DPExpGK', 'DCS DP'], [err_kll_list, err_gk_list, err_dcs_list]):
                    plot_data[eps][m]['mean'].append(float(np.mean(err_arr)))
                    plot_data[eps][m]['p10'].append(float(np.percentile(err_arr, 10)))
                    plot_data[eps][m]['p90'].append(float(np.percentile(err_arr, 90)))
                
                # 记录时间
                plot_data[eps]['PrivKLL-Q']['avg_ingest_time_ms'].append(float(np.mean(kll_in_times) * 1000))
                plot_data[eps]['PrivKLL-Q']['avg_query_time_ms'].append(float(np.mean(kll_q_times) * 1000))
                plot_data[eps]['DPExpGK']['avg_ingest_time_ms'].append(float(np.mean(gk_in_times) * 1000))
                plot_data[eps]['DPExpGK']['avg_query_time_ms'].append(float(np.mean(gk_q_times) * 1000))
                plot_data[eps]['DCS DP']['avg_ingest_time_ms'].append(float(np.mean(dcs_in_times) * 1000))
                plot_data[eps]['DCS DP']['avg_query_time_ms'].append(float(np.mean(dcs_q_times) * 1000))

                print(f"  [Summary] Epsilon: {eps} | Space Budget: {mem:<4d} KB | PrivKLL-Q Err: {np.mean(err_kll_list):.5f} ")

    # 保存 JSON 文件与 1x5 学术曲线图
    with open(json_filename, 'w') as f: 
        json.dump({"alignment_meta": param_cache, "data": plot_data}, f, indent=4)
        
    plot_vldb_1x6_transposed(cfg['memory_kbs'], plot_data, cfg['epsilons'], r'Memory Budget (KB)', png_filename)
    print(f"\n✔ [Completed] 结果成功导出到以 N 结尾的文件:\n  JSON: {json_filename}\n  PNG: {png_filename}")

# ------------------------------------------------------------------------------
# 4. 主入口控制中心
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", default=100_000, type=int, help="Stream size")
    parser.add_argument("--replot", action="store_true", help="Only replot from existing JSON data")
    args = parser.parse_args()
    
    target_n = args.N
    json_filename = RESULTS / f"exp_s_mem_fixed_N{target_n}.json"
    png_filename = RESULTS / f"exp_s_mem_fixed_N{target_n}.png"
    
    logger = get_default_logger()

    # 🌟 新增：如果指定了 --replot 参数，直接读取本地 JSON 绘图并退出
    if args.replot:
        if not json_filename.exists():
            logger.error(f"找不到缓存的 JSON 数据文件: {json_filename}，无法重新绘图！")
            return
            
        logger.info(f"📂 [Replot Mode] 正在从本地读取数据: {json_filename}")
        with open(json_filename, 'r') as f:
            saved_data = json.load(f)
            
        # 提取历史物理试验结果数据
        plot_data = saved_data["data"]
        
        # 这里的配置需要和生成 JSON 时的参数严格对齐
        epsilons = [0.5, 1.0, 2.0, 4.0, 8.0]
        memory_kbs = [4, 8, 32, 64, 256, 1024]
        
        logger.info("🎨 正在使用重构后的转置引擎重新渲染学术 1x6 大图...")
        plot_vldb_1x6_transposed(epsilons, memory_kbs, plot_data, r'Privacy Budget ($\epsilon$)', png_filename)
        logger.info(f"✔ [Success] 重新绘图完成！新图已导出至: {png_filename}")
        return

    # ------------------------------------------------------------------ #
    # 标准物理试验运行模式（未指定 --replot 时走原本的老逻辑）
    # ------------------------------------------------------------------ #
    cfg = {
        'N': target_n,                                
        'distributions': ['zipf'],
        'eps': 1.0,
        'trials': 10,
        'target_q': 0.5,
        'epsilons': [0.5, 1.0, 2.0, 4.0, 8.0],     
        'memory_kbs': [4, 8, 32, 64, 256, 1024],         
    }

    logger.info(f"Launching fixed-N transposed memory experiment (N={cfg['N']})...")
    run_mem_experiment_fixed_n(cfg)
    
    logger.info("=" * 80)
    logger.info("FIXED-N TRANSPOSED MEMORY EXPERIMENT COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()