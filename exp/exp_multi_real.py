'''
Author: Jean jean_green@163.com
Date: 2026-06-26 01:30:00
Description: 真实世界数据集专属多分位数差分隐私仿真审计控制中心（支持 Varying m 和 Varying Epsilon 实验，1x3 并排 VLDB 画布输出）
'''
import argparse
import os
from pathlib import Path
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import json
from datetime import datetime
import logging
from concurrent.futures import ProcessPoolExecutor

# 从你建立的系统架构组件和算法模块导入
from DPQE.src.PrivKLL import PrivKLL
from DPQE.src.kll import KLL
from DPQE.src.Greenwald_Khanna import GK
from DPQE.src.DCS import DCS

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

def get_default_logger():
    logger = logging.getLogger("exp_real_world")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler("exp_real_world_pipeline.log")
        ch = logging.StreamHandler()
        fmt = logging.Formatter("[%(asctime)s] %(message)s")
        fh.setFormatter(fmt)
        ch.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger

# ------------------------------------------------------------------------------
# 1. 内存映射（mmap）流式分块生成器
# ------------------------------------------------------------------------------
def stream_chunk_generator(file_path, total_n, chunk_size=500_000):
    """
    流式分块数据生成器，基于内存映射（mmap_mode='r'），零内存占用高速切片。
    """
    remains = total_n
    npy_data = np.load(file_path, mmap_mode='r')
    total_n = min(total_n, len(npy_data))
    remains = total_n

    current_idx = 0
    while remains > 0:
        current_chunk = min(chunk_size, remains)
        chunk_data = npy_data[current_idx : current_idx + current_chunk]
        current_idx += current_chunk
        yield chunk_data
        remains -= current_chunk

def run_multi_trial_strict(eps, dist_type, target_qs, kll_k, gk_alpha, dcs_gamma, dcs_uni=2**16, chunk_size=500000):
    """
    底层物理流计算：优化后 PrivKLL-Q 和 PrivKLL-JQ 复用同一个底层的私有 KLL Sketch。
    数据摄入共同计时，终点站查询释放分别独立计时。
    """
    method_times = {'PrivKLL-Q': 0.0, 'PrivKLL-JQ': 0.0, 'DPExpGK': 0.0, 'DCS DP': 0.0}

    # 1. 初始化各算法实例（KLL 只需初始化一个共享实例）
    t_start_init = time.perf_counter()
    shared_priv_kll = PrivKLL(epsilon=1.0, kll_k=kll_k)  # 共享的隐私KLL Sketch
    t_init_kll = time.perf_counter() - t_start_init
    # 将初始化的开销均摊（或完全计入两者，这里为了公平均摊）
    method_times['PrivKLL-Q'] += t_init_kll
    method_times['PrivKLL-JQ'] += t_init_kll

    t_start_init = time.perf_counter()
    gk_indep = GK(alpha=gk_alpha)
    method_times['DPExpGK'] += time.perf_counter() - t_start_init

    t_start_init = time.perf_counter()
    rho_cdp = 0.5 * (eps ** 2)
    dcs_sketch = DCS(universe=dcs_uni, gamma=dcs_gamma, rho=rho_cdp)
    method_times['DCS DP'] += time.perf_counter() - t_start_init

    total_n = 100_000 
    sample_size = min(total_n, 100_000)
    reservoir = []
    item_counter = 0
    
    def to_dcs_domain(val):
        return int(np.clip(val / 100_000.0 * (dcs_uni - 1), 0, dcs_uni - 1))
    
    def from_dcs_domain(int_val):
        return (int_val / (dcs_uni - 1)) * 100_000.0

    # 2. 物理数据流高速摄入与维护阶段（KLL 复用同一个 ingest 计时）
    generator = stream_chunk_generator(dist_type, total_n, chunk_size=chunk_size)
    for chunk in generator:
        for val in chunk:
            # non_dp_kll.update(val)
            
            # 【优化重点】对共享的隐私KLL进行单次 ingest，时间同时累加给两个方法
            start_kll = time.perf_counter()
            shared_priv_kll.ingest_data(val)
            time_kll = time.perf_counter() - start_kll
            method_times['PrivKLL-Q'] += time_kll
            method_times['PrivKLL-JQ'] += time_kll

            # 统计 DPExpGK 插入耗时
            start_gk = time.perf_counter()
            gk_indep.insert(val)
            method_times['DPExpGK'] += time.perf_counter() - start_gk

            # 统计 DCS DP 状态更新耗时
            start_dcs = time.perf_counter()
            dcs_sketch.update(to_dcs_domain(val))
            method_times['DCS DP'] += time.perf_counter() - start_dcs

            if len(reservoir) < sample_size:
                reservoir.append(val)
            else:
                r = np.random.randint(0, item_counter + 1)
                if r < sample_size: reservoir[r] = val
            item_counter += 1

    total_n = item_counter 
    m = len(target_qs)
    final_estimates = {'PrivKLL-Q': [], 'PrivKLL-JQ': [], 'DPExpGK': [], 'DCS DP': []}
    
    # 3. 终点站查询与解算阶段（完全分开计时计算）
    # 策略 1: PrivKLL-Q 独立发布模式 (调用共享 Sketch，串行平分预算查询)
    start_q_indep = time.perf_counter()
    eps_kll_indep = eps / m
    for q in target_qs:
        shared_priv_kll.epsilon = eps_kll_indep  # 动态调整当前查询的局部预算
        final_estimates['PrivKLL-Q'].append(shared_priv_kll.release_quantile(q))
    method_times['PrivKLL-Q'] += time.perf_counter() - start_q_indep
        
    # 策略 2: PrivKLL-JQ 联合发布模式 (调用同一个共享 Sketch，利用覆盖树，无需平分预算)
    start_q_joint = time.perf_counter()
    shared_priv_kll.epsilon = eps  # 恢复全局完整的总预算进行联合查询
    # final_estimates['PrivKLL-JQ'].append(shared_priv_kll.release_multi_quantiles(target_qs))
    final_estimates['PrivKLL-JQ'].extend(shared_priv_kll.release_multi_quantiles(target_qs))
    method_times['PrivKLL-JQ'] += time.perf_counter() - start_q_joint
        
    # 策略 3: GK 独立发布
    start_q = time.perf_counter()
    eps_gk_indep = eps / m
    for q in target_qs:
        final_estimates['DPExpGK'].append(gk_indep.dp_exp(q, eps_gk_indep))
    method_times['DPExpGK'] += time.perf_counter() - start_q
        
    # 策略 4: DCS DP 发布 
    start_q = time.perf_counter()
    for q in target_qs:
        target_rank = q * total_n
        low_idx, high_idx = 0, dcs_uni - 1
        est_dcs_int = low_idx
        while low_idx <= high_idx:
            mid_idx = (low_idx + high_idx) // 2
            if dcs_sketch.query(mid_idx) <= target_rank:
                est_dcs_int = mid_idx
                low_idx = mid_idx + 1
            else:
                high_idx = mid_idx - 1
        final_estimates['DCS DP'].append(from_dcs_domain(est_dcs_int))
    method_times['DCS DP'] += time.perf_counter() - start_q

    # 4. 计算误差及内存大小（由于复用底层的 KLL 结构，kll_mem 保持不变）
    res_arr = np.array(reservoir)
    final_errors = {}
    for method, ests in final_estimates.items():
        q_errors = []
        for idx, q in enumerate(target_qs):
            cur_est = ests[idx] if ests[idx] is not None else res_arr[-1]
            actual_percentile = (res_arr <= cur_est).mean()
            q_errors.append(abs(actual_percentile - q))
        final_errors[method] = np.mean(q_errors)
        
    kll_mem = sum(len(c) for c in shared_priv_kll.compactors) * 8
    gk_mem = len(gk_indep.S) * 16
    dcs_mem = dcs_sketch.memory_budget() * 4
    
    return final_errors, kll_mem, gk_mem, dcs_mem, method_times

# ------------------------------------------------------------------------------
# 3. 统一 VLDB 风格真实数据 1x3 排版绘图引擎
# ------------------------------------------------------------------------------
def plot_vldb_real_1x3_fig(x_values, plot_data, xlabel_str, x_scale_log, base_x, fname, datasets_list):
    plt.rcParams.update({
        'font.size': 8.5, 
        'font.family': 'serif', 
        'mathtext.fontset': 'cm',
        'xtick.labelsize': 7.5, 
        'ytick.labelsize': 8
    })
    
    fig, axes = plt.subplots(1, 3, figsize=(3.35, 1.85), sharey=True)
    dist_labels = [f"({chr(97+i)}) {name.upper().replace('_UNLIMITED','')}" for i, name in enumerate(datasets_list)]
    
    methods = ['PrivKLL-Q', 'PrivKLL-JQ', 'DPExpGK', 'DCS DP']
    style_cfgs = {
        'PrivKLL-Q': {'color': '#3A86FF', 'linestyle': '--', 'marker': 'v'},
        'PrivKLL-JQ': {'color': '#FF006E', 'linestyle': ':', 'marker': 'x'},
        'DPExpGK': {'color': '#06D6A0', 'linestyle': '-', 'marker': 'o'},
        'DCS DP': {'color': '#6C757D', 'linestyle': '-.', 'marker': '^'}
    }
    lines, labels = [], []

    box_width, box_height, box_bottom = 0.26, 0.46, 0.34
    box_lefts = [0.14, 0.43, 0.72]

    for idx, dist in enumerate(datasets_list):
        ax = axes[idx]
        ax.set_position([box_lefts[idx], box_bottom, box_width, box_height])
        
        dist_res = plot_data[dist]
        for m in methods:
            mean = np.clip(np.array(dist_res[m]['mean']), 1e-7, None)
            p10 = np.clip(np.array(dist_res[m]['p10']), 1e-7, None)
            p90 = np.clip(np.array(dist_res[m]['p90']), 1e-7, None)
            
            line, = ax.plot(x_values, mean, color=style_cfgs[m]['color'], linestyle=style_cfgs[m]['linestyle'], 
                             marker=style_cfgs[m]['marker'], linewidth=1.1, markersize=3.0, zorder=4)
            ax.fill_between(x_values, p10, p90, color=style_cfgs[m]['color'], alpha=0.06)
            if idx == 0: lines.append(line); labels.append(m)
        
        if x_scale_log: ax.set_xscale('log', base=base_x)
        ax.set_xticks(x_values)
        ax.set_xticklabels([str(x) for x in x_values])
        
        ax.set_yscale('log', base=2)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: f"$2^{{{int(np.log2(y))}}}$" if y > 0 else "0"))
        
        ax.set_title(dist_labels[idx], y=-0.52, fontsize=7.0, va='top')
        ax.grid(True, which="both", ls="--", color='#e0e0e0', alpha=0.6, linewidth=0.5)
        
    fig.text(0.56, 0.21, xlabel_str, ha='center', va='center', fontsize=8.5)
    axes[0].set_ylabel('Mean Abs Rank Error', fontsize=8.5, labelpad=2)
    
    fig.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.56, 1.0), 
               ncol=4, frameon=True, fancybox=True, shadow=False, 
               fontsize=7, columnspacing=0.5, handletextpad=0.2, framealpha=1.0)
    
    plt.savefig(fname, dpi=300)
    plt.close()

# ------------------------------------------------------------------------------
# 4. 核心实验驱动：1) Varying m 实验控制中心
# ------------------------------------------------------------------------------
def run_experiment_m_M_real(cfg, timestamp):
    print("\n" + "="*80 + f"\n▶ [Executing Real-World] Exp 1 — Varying Number of Quantiles m (Fixed Epsilon={cfg['eps']})\n" + "="*80)
    start = time.time()
    methods = ['PrivKLL-Q', 'PrivKLL-JQ', 'DPExpGK', 'DCS DP']
    
    raw_res = {dist: {m: {mv: [] for mv in cfg['exp1_Ms']} for m in methods} for dist in cfg['real_datasets']}
    # 用于收集并存储时间数据的容器
    time_res = {dist: {m: {mv: [] for mv in cfg['exp1_Ms']} for m in methods} for dist in cfg['real_datasets']}
    meta_log = {}

    with ProcessPoolExecutor() as executor:
        for dist in cfg['real_datasets']:
            meta_log[dist] = {"GK_alpha": cfg['GK_alpha'], "PrivKLL_k": cfg['PrivKLL_k'], "DCS_gamma": cfg['DCS_gamma']}

            for mv in cfg['exp1_Ms']:
                quantiles = [j / (mv + 1) for j in range(1, mv + 1)]
                
                actual_file_path = DATA / f"{dist}.npy"
                futures = []
                for _ in range(cfg['trials']):
                    f = executor.submit(
                        run_multi_trial_strict, 
                        cfg['eps'], actual_file_path, quantiles, cfg['PrivKLL_k'], cfg['GK_alpha'], cfg['DCS_gamma'], cfg['DCS_uni']
                    )
                    futures.append(f)
                
                kll_mems, gk_mems, dcs_mems = [], [], []
                for f in futures:
                    errs, kll_mem, gk_mem, dcs_mem, method_times = f.result()
                    kll_mems.append(kll_mem); gk_mems.append(gk_mem); dcs_mems.append(dcs_mem)
                    for m in methods:
                        raw_res[dist][m][mv].append(errs[m])
                        time_res[dist][m][mv].append(method_times[m])
                    
                print(f"  [Summary] m: {mv:<2} | Dataset: {dist.upper()} | Trials: {cfg['trials']}")
                print(f"    ├─ KLL-Indep: Error: {np.mean(raw_res[dist]['PrivKLL-Q'][mv]):.5f} | Mem: {np.mean(kll_mems)/1024:.2f} KB | Time: {np.mean(time_res[dist]['PrivKLL-Q'][mv]):.4f}s")
                print(f"    ├─ KLL-Joint: Error: {np.mean(raw_res[dist]['PrivKLL-JQ'][mv]):.5f} | Mem: {np.mean(kll_mems)/1024:.2f} KB | Time: {np.mean(time_res[dist]['PrivKLL-JQ'][mv]):.4f}s")
                print(f"    ├─ GK-Indep:  Error: {np.mean(raw_res[dist]['DPExpGK'][mv]):.5f} | Mem: {np.mean(gk_mems)/1024:.2f} KB | Time: {np.mean(time_res[dist]['DPExpGK'][mv]):.4f}s")
                print(f"    └─ DCS DP:    Error: {np.mean(raw_res[dist]['DCS DP'][mv]):.5f} | Mem: {np.mean(dcs_mems)/1024:.2f} KB | Time: {np.mean(time_res[dist]['DCS DP'][mv]):.4f}s")
                print("    " + "-"*65)

    end = time.time()
    print(f"\n▶ Exp Varying m completed in {end - start:.2f} seconds.")

    plot_data = {dist: {m: {'mean': [], 'p10': [], 'p90': []} for m in methods} for dist in cfg['real_datasets']}
    for dist in cfg['real_datasets']:
        for m in methods:
            for mv in cfg['exp1_Ms']:
                arr = raw_res[dist][m][mv]
                plot_data[dist][m]['mean'].append(np.mean(arr))
                plot_data[dist][m]['p10'].append(np.percentile(arr, 10))
                plot_data[dist][m]['p90'].append(np.percentile(arr, 90))

    with open(RESULTS / f"exp_m_M_real1x3_{timestamp}.json", 'w') as f: 
        json.dump({"meta": meta_log, "data": plot_data}, f, indent=4)
    plot_vldb_real_1x3_fig(cfg['exp1_Ms'], plot_data, 'Number of Quantiles ($m$)', False, 2, RESULTS / f"exp_m_M_real1x3_{timestamp}.png", cfg['real_datasets'])
    print(f"▶ Results saved to JSON and PNG.\n")

# ------------------------------------------------------------------------------
# 5. 核心实验驱动：2) Varying Epsilon 实验控制中心
# ------------------------------------------------------------------------------
def run_experiment_m_eps_real(cfg, timestamp):
    print("\n" + "="*80 + f"\n▶ [Executing Real-World] Exp 2 — Varying Privacy Budget (Fixed m={cfg['M']}, Trials={cfg['trials']} KB)\n" + "="*80)
    start = time.time()
    methods = ['PrivKLL-Q', 'PrivKLL-JQ', 'DPExpGK', 'DCS DP']
    
    raw_res = {dist: {m: {ev: [] for ev in cfg['exp2_epsilons']} for m in methods} for dist in cfg['real_datasets']}
    time_res = {dist: {m: {ev: [] for ev in cfg['exp2_epsilons']} for m in methods} for dist in cfg['real_datasets']}
    meta_log = {}
    quantiles = [j / (cfg['M'] + 1) for j in range(1, cfg['M'] + 1)]

    with ProcessPoolExecutor() as executor:
        for dist in cfg['real_datasets']:
            meta_log[dist] = {"GK_alpha": cfg['GK_alpha'], "PrivKLL_k": cfg['PrivKLL_k'], "DCS_gamma": cfg['DCS_gamma']}

            for ev in cfg['exp2_epsilons']:
                futures = []
                for _ in range(cfg['trials']):
                    f = executor.submit(
                        run_multi_trial_strict,
                        ev, dist, quantiles, cfg['PrivKLL_k'], cfg['GK_alpha'], cfg['DCS_gamma'], cfg['DCS_uni']
                    )
                    futures.append(f)
                
                kll_mems, gk_mems, dcs_mems = [], [], []
                for f in futures:
                    errs, kll_mem, gk_mem, dcs_mem, method_times = f.result()
                    kll_mems.append(kll_mem); gk_mems.append(gk_mem); dcs_mems.append(dcs_mem)
                    for m in methods:
                        raw_res[dist][m][ev].append(errs[m])
                        time_res[dist][m][ev].append(method_times[m])
                        
                print(f"  [Parallel Summary] Epsilon: {ev:<3.1f} | Dataset: {dist.upper()} | Completed Trials: {cfg['trials']}")
                print(f"    ├─ KLL-Indep: Error: {np.mean(raw_res[dist]['PrivKLL-Q'][ev]):.5f} | Mem: {np.mean(kll_mems)/1024:.2f} KB | Time: {np.mean(time_res[dist]['PrivKLL-Q'][ev]):.4f}s")
                print(f"    ├─ KLL-Joint: Error: {np.mean(raw_res[dist]['PrivKLL-JQ'][ev]):.5f} | Mem: {np.mean(kll_mems)/1024:.2f} KB | Time: {np.mean(time_res[dist]['PrivKLL-JQ'][ev]):.4f}s")
                print(f"    ├─ GK-Indep:  Error: {np.mean(raw_res[dist]['DPExpGK'][ev]):.5f} | Mem: {np.mean(gk_mems)/1024:.2f} KB | Time: {np.mean(time_res[dist]['DPExpGK'][ev]):.4f}s")
                print(f"    └─ DCS DP:    Error: {np.mean(raw_res[dist]['DCS DP'][ev]):.5f} | Mem: {np.mean(dcs_mems)/1024:.2f} KB | Time: {np.mean(time_res[dist]['DCS DP'][ev]):.4f}s")
                print("    " + "-"*65)

    end = time.time()
    print(f"\n▶ Exp Varying Epsilon completed in {end - start:.2f} seconds.")

    plot_data = {dist: {m: {'mean': [], 'p10': [], 'p90': []} for m in methods} for dist in cfg['real_datasets']}
    for dist in cfg['real_datasets']:
        for m in methods:
            for ev in cfg['exp2_epsilons']:
                arr = raw_res[dist][m][ev]
                plot_data[dist][m]['mean'].append(np.mean(arr))
                plot_data[dist][m]['p10'].append(np.percentile(arr, 10))
                plot_data[dist][m]['p90'].append(np.percentile(arr, 90))

    with open(RESULTS / f"exp_m_eps_real1x3_{timestamp}.json", 'w') as f: 
        json.dump({"meta": meta_log, "data": plot_data}, f, indent=4)
    plot_vldb_real_1x3_fig(cfg['exp2_epsilons'], plot_data, r'Privacy Budget ($\epsilon$)', True, 2, RESULTS / f"exp_m_eps_real1x3_{timestamp}.png", cfg['real_datasets'])
    print(f"▶ Results saved to JSON and PNG.\n")

# ------------------------------------------------------------------------------
# 6. 主控制中心
# ------------------------------------------------------------------------------
def main(logger=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="Quick debug mode")
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    cfg = {
        'M': 9,
        'eps': 1.0,
        'trials': 30, # 真实数据集推荐跑20-100次
        'real_datasets': ['mimic_hr', 'power_active', 'taxi_duration'],

        'mem_kb': 32, 
        'PrivKLL_k': 1344,  
        'GK_alpha': 0.000575,  
        'DCS_gamma': 0.074726,  # DCS parameter 
        'DCS_uni': 2**16,  

        # 实验一参数梯度 (Varying M)
        'exp1_Ms': [1, 3, 5, 9, 15, 21],

        # 实验二参数梯度 (Varying Epsilon)
        'exp2_epsilons': [0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4],
    }
    
    if args.fast:
        cfg['trials'] = 1
        cfg['exp1_Ms'] = [1, 5]
        cfg['exp2_epsilons'] = [0.5, 4.0]
        cfg['real_datasets'] = ['mimic_hr', 'power_active', 'taxi_duration']

    if logger is None:
        logger = get_default_logger()
    logger.info("Initializing Real-World Datasets Multiple Quantiles Benchmark Pipeline...")

    run_experiment_m_M_real(cfg, timestamp)
    
    # run_experiment_m_eps_real(cfg, timestamp)
    
    logger.info("=" * 80)
    logger.info("ALL REAL-WORLD EXPERIMENTS EXECUTED AND REDIRECTED TO 1X3 FIG SUCCESSFULLY")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()