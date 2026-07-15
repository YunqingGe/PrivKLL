import argparse
from pathlib import Path
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import json
from datetime import datetime
import logging
from concurrent.futures import ProcessPoolExecutor

# 统一从外部核心组件和校准模块导入
from DPQE.exp import calibration
from DPQE.src.PrivKLL import PrivKLL
from DPQE.src.kll import KLL
from DPQE.src.Greenwald_Khanna import GK, GKTuple
from DPQE.src.DCS import DCS

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

def get_default_logger():
    logger = logging.getLogger("exp_multi")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler("exp_multi.log")
        ch = logging.StreamHandler()
        fmt = logging.Formatter("[%(asctime)s] %(message)s")
        fh.setFormatter(fmt)
        ch.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger

# ------------------------------------------------------------------------------
# 2. 内存安全的 Chunk 流式生成器与核心多分位数仿真审计流
# ------------------------------------------------------------------------------
def stream_chunk_generator(dist_type, total_n, chunk_size=500_000, domain=(0, 100_000)):
    low, high = domain
    remains = total_n
    while remains > 0:
        current_chunk = min(chunk_size, remains)
        if dist_type == 'uniform': chunk_data = np.random.uniform(low, high, current_chunk)
        elif dist_type == 'normal': chunk_data = np.clip(np.random.normal((high+low)/2, (high-low)/6, current_chunk), low, high)
        elif dist_type == 'zipf': chunk_data = np.clip(np.random.pareto(a=2.0, size=current_chunk) * (high/10), low, high)
        yield chunk_data
        remains -= current_chunk

def run_multi_trial_strict(eps, dist_type, total_n, target_qs, kll_k, dcs_uni_cfg=None, chunk_size=500000):
    # 🌟 核心优化：只初始化一个 PrivKLL 实例即可！
    priv_kll = PrivKLL(epsilon=eps, kll_k=kll_k)

    reservoir = []
    sample_rate = max(1, total_n // 100000) 

    t_kll_ingest = 0.0
    global_idx = 0
    generator = stream_chunk_generator(dist_type, total_n, chunk_size=chunk_size)
    
    # 🌟 优化：流式推进阶段，只更新这一个 sketch
    for chunk in generator:
        for val in chunk:
            if global_idx % sample_rate == 0:
                reservoir.append(val)
            global_idx += 1

            t0 = time.perf_counter()
            priv_kll.ingest_data(val) # 仅 ingest 一次，三大算法共用
            t_kll_ingest += (time.perf_counter() - t0)

    # 映射插入耗时：因为结构完全一样，三种策略的 Ingest Time 直接拉平
    t_kll_Q_ingest = t_kll_ingest
    t_kll_JQ_ingest = t_kll_ingest

    m = len(target_qs)
    final_estimates = {'KLL-NonDP': [], 'PrivKLL-Q': [], 'PrivKLL-JQ': []}
    
    # ── 终点站解算发布与耗时审计 ──
    
    # 策略 0: KLL-NonDP (直接调用底层非隐私 sketch 的解算方法)
    t0 = time.perf_counter()
    for q in target_qs:
        final_estimates['KLL-NonDP'].append(priv_kll.sketch.quantile(q))
    t_kll_non_dp_release = time.perf_counter() - t0

    # 策略 1: PrivKLL-Q (独立查询：临时改变 epsilon 分配查询)
    t0 = time.perf_counter()
    orig_eps = priv_kll.epsilon
    eps_kll_indep = eps / m
    for q in target_qs:
        priv_kll.epsilon = eps_kll_indep # 动态切换单次预算
        final_estimates['PrivKLL-Q'].append(priv_kll.release_quantile(q))
    t_kll_Q_release = time.perf_counter() - t0
        
    # 策略 2: PrivKLL-JQ (联合查询：恢复总预算，一次性调用多查询接口)
    t0 = time.perf_counter()
    priv_kll.epsilon = orig_eps # 恢复总隐私预算
    jq_res = priv_kll.release_multi_quantiles(target_qs)
    
    if isinstance(jq_res, (list, np.ndarray)):
        for est in jq_res:
            final_estimates['PrivKLL-JQ'].append(est)
    else:
        final_estimates['PrivKLL-JQ'].append(jq_res)
    t_kll_JQ_release = time.perf_counter() - t0
        
    # 计算绝对排名误差
    res_arr = np.array(reservoir)
    final_errors = {}
    for method, ests in final_estimates.items():
        q_errors = []
        for idx, q in enumerate(target_qs):
            cur_est = ests[idx] if ests[idx] is not None else res_arr[-1]
            actual_percentile = (res_arr <= cur_est).mean()
            q_errors.append(abs(actual_percentile - q))
        final_errors[method] = np.mean(q_errors)
        
    # 🌟 精准计算 KLL 物理内存（Bytes）
    kll_mem = sum(len(c) for c in priv_kll.compactors) * 8
    
    return (final_errors, kll_mem,
            t_kll_ingest, t_kll_Q_ingest, t_kll_JQ_ingest, 
            t_kll_non_dp_release, t_kll_Q_release, t_kll_JQ_release)

# ------------------------------------------------------------------------------
# 3. 统一 VLDB 风格多子图并排绘图引擎
# ------------------------------------------------------------------------------
def plot_vldb_multi_fig(x_values, plot_data, xlabel_str, x_scale_log, base_x, fname):
    plt.rcParams.update({
        'font.size': 8.5, 'font.family': 'serif', 'mathtext.fontset': 'cm',
        'xtick.labelsize': 7.5, 'ytick.labelsize': 8
    })
    
    fig, axes = plt.subplots(1, 3, figsize=(3.35, 1.85), sharey=True)
    distributions = ['uniform', 'normal', 'zipf']
    dist_labels = ['(a) UNIFORM', '(b) NORMAL', '(c) ZIPF']
    
    # 🌟 修复：方法名与上游清洗对齐
    methods = ['KLL-NonDP', 'PrivKLL-Q', 'PrivKLL-JQ']
    style_cfgs = {
        'KLL-NonDP': {'color': "#6C757D", 'linestyle': '-', 'marker': 'o'},
        'PrivKLL-Q': {'color': '#3A86FF', 'linestyle': '--', 'marker': 'v'},
        'PrivKLL-JQ': {'color': '#FF006E', 'linestyle': ':', 'marker': 'x'},
    }
    lines, labels = [], []

    box_width = 0.26
    box_height = 0.46
    box_bottom = 0.34
    box_lefts = [0.14, 0.43, 0.72]

    for idx, dist in enumerate(distributions):
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
        
        ax.set_title(dist_labels[idx], y=-0.52, fontsize=8.5, va='top')
        ax.grid(True, which="both", ls="--", color='#e0e0e0', alpha=0.6, linewidth=0.5)
        
    fig.text(0.56, 0.21, xlabel_str, ha='center', va='center', fontsize=8.5)
    axes[0].set_ylabel('Mean Abs Rank Error', fontsize=8.5, labelpad=2)
    
    fig.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.56, 1.0), 
               ncol=3, frameon=True, fancybox=True, shadow=False, 
               fontsize=7, columnspacing=0.5, handletextpad=0.2, framealpha=1.0)
    
    plt.savefig(fname, dpi=300)
    plt.close()

# ------------------------------------------------------------------------------
# 🌟 并行独立工作子进程任务分发器
# ------------------------------------------------------------------------------
def _worker_single_trial(seed, eps, dist, tv, quantiles, kll_k_t, dcs_uni_cfg=None):
    np.random.seed(seed)
    outputs = run_multi_trial_strict(
        eps, dist, tv, quantiles, kll_k_t, dcs_uni_cfg
    )
    return outputs

# ------------------------------------------------------------------------------
# 🌟 修改后的主循环控制中心
# ------------------------------------------------------------------------------
def run_experiment_m_KLL(cfg, timestamp):
    print("\n" + "="*80 + f"\n▶ [Executing] Exp multi_N (Parallel) — Varying Stream Length T (Fixed m={cfg['M']}, Target Budget={cfg['mem_kb']} KB)\n" + "="*80)
    start_time = time.time()
    
    methods = ['KLL-NonDP', 'PrivKLL-Q', 'PrivKLL-JQ']    
    raw_res = {dist: {m: {tv: [] for tv in cfg['exp3_Ns']} for m in methods} for dist in cfg['distributions']}
    
    # 建立多维性能指标追踪池
    metrics_tracker = {
        dist: {
            tv: {
                m: {'mem': [], 't_ingest': [], 't_release': []} for m in methods
            } for tv in cfg['exp3_Ns']
        } for dist in cfg['distributions']
    }
    
    meta_log = {}
    quantiles = [j / (cfg['M'] + 1) for j in range(1, cfg['M'] + 1)]

    # 🌟 修改点 1：定义 N 到 trials 的刚性映射字典
    # 这样大样本量（10^9）跑较少轮次，小样本量跑更多轮次以稳定方差，极其符合学术规范
    n_to_trials = {
        10_000: 100,
        100_000: 100,
        1_000_000: 50,
        10_000_000: 30,
        100_000_000: 20,
        1_000_000_000: 5
    }

    with ProcessPoolExecutor() as executor:
        for dist in cfg['distributions']:
            meta_log[dist] = {}
            
            for tv in cfg['exp3_Ns']:
                # 🌟 修改点 2：动态获取当前 tv 对应的 trials 次数，如果找不到则用 cfg['trials'] 兜底
                current_trials = n_to_trials.get(tv, cfg.get('trials', 5))

                target_bytes = cfg['mem_kb'] * 1024
                safe_H = int(np.ceil(np.log(tv) / np.log(1.5)))
                kll_k_t = calibration.find_kll_k_by_budget(target_bytes, H=safe_H)
                meta_log[dist][tv] = {"PrivKLL_k": kll_k_t}
                
                futures = []
                base_seed = int(time.time()) % 100000
                
                for trial_idx in range(current_trials):
                    unique_seed = base_seed + trial_idx + int(np.log10(tv)) * 100
                    f = executor.submit(
                        _worker_single_trial,
                        seed=unique_seed,
                        eps=cfg['eps'],
                        dist=dist,
                        tv=tv,
                        quantiles=quantiles,
                        kll_k_t=kll_k_t,
                        dcs_uni_cfg=cfg.get('DCS_uni', None)
                    )
                    futures.append(f)
                
                # 收集多进程并发传回的多元数据
                for f in futures:
                    (errs, kll_mem, 
                     t_non_in, t_q_in, t_jq_in, 
                     t_non_re, t_q_re, t_jq_re) = f.result()
                    
                    for m in methods:
                        raw_res[dist][m][tv].append(errs[m])
                    
                    # 归类映射时间与空间负载指标
                    metrics_tracker[dist][tv]['KLL-NonDP']['mem'].append(kll_mem)
                    metrics_tracker[dist][tv]['KLL-NonDP']['t_ingest'].append(t_non_in)
                    metrics_tracker[dist][tv]['KLL-NonDP']['t_release'].append(t_non_re)

                    metrics_tracker[dist][tv]['PrivKLL-Q']['mem'].append(kll_mem)
                    metrics_tracker[dist][tv]['PrivKLL-Q']['t_ingest'].append(t_q_in)
                    metrics_tracker[dist][tv]['PrivKLL-Q']['t_release'].append(t_q_re)

                    metrics_tracker[dist][tv]['PrivKLL-JQ']['mem'].append(kll_mem)
                    metrics_tracker[dist][tv]['PrivKLL-JQ']['t_ingest'].append(t_jq_in)
                    metrics_tracker[dist][tv]['PrivKLL-JQ']['t_release'].append(t_jq_re)

                # 🌟 核心打印控制台：全景展现 Accuracy, Memory, Ingest & Query 耗时
                print(f"  [Summary Pipeline] Stream Length T: {tv:<11} | Distribution: {dist.upper()} | Completed Trials: {current_trials}")
                for m in methods:
                    m_data = metrics_tracker[dist][tv][m]
                    avg_err = np.mean(raw_res[dist][m][tv])
                    avg_mem = np.mean(m_data['mem']) / 1024.0
                    avg_in  = np.mean(m_data['t_ingest'])
                    avg_re  = np.mean(m_data['t_release'])
                    
                    print(f"    ├─ [{m:<12}] -> Accuracy (Rank Error): {avg_err:.6f} | Memory: {avg_mem:5.2f} KB | "
                          f"Ingest Time: {avg_in:7.4f}s | Query Time: {avg_re:7.4f}s")
                print("    " + "-" * 115)

    end_time = time.time()
    print(f"\n▶ Exp multi_N parallel run completed in {end_time - start_time:.2f} seconds.")

    # 后处理打包数据落盘
    plot_data = {dist: {m: {'mean': [], 'p10': [], 'p90': []} for m in methods} for dist in cfg['distributions']}
    for dist in cfg['distributions']:
        for m in methods:
            for tv in cfg['exp3_Ns']:
                arr = raw_res[dist][m][tv]
                plot_data[dist][m]['mean'].append(np.mean(arr))
                plot_data[dist][m]['p10'].append(np.percentile(arr, 10))
                plot_data[dist][m]['p90'].append(np.percentile(arr, 90))

    with open(RESULTS / f"exp_m_N_{timestamp}.json", 'w') as f: 
        json.dump({"meta": meta_log, "data": plot_data}, f, indent=4)
        
    plot_vldb_multi_fig(cfg['exp3_Ns'], plot_data, 'Stream Length ($T$)', True, 10, RESULTS / f"exp_m_N_{timestamp}.png")    
    print(f"\n▶ Exp multi_N results saved to {RESULTS / f'exp_m_N_{timestamp}.json'} and {RESULTS / f'exp_m_N_{timestamp}.png'}")

def run_experiment_m_N(cfg, timestamp):
    print("\n" + "="*80 + f"\n▶ [Executing] Exp multi — Varying Stream Length T (Fixed m={cfg['M']}, Target Budget={cfg['mem_kb']} KB)\n" + "="*80)
    start = time.time()
    methods = ['PrivKLL-Q', 'PrivKLL-JQ', 'DPExpGK', 'DCS DP']    

    raw_res = {dist: {m: {tv: [] for tv in cfg['exp3_Ns']} for m in methods} for dist in cfg['distributions']}
    meta_log = {}
    quantiles = [j / (cfg['M'] + 1) for j in range(1, cfg['M'] + 1)]

    for dist in cfg['distributions']:
        meta_log[dist] = {}
        for tv in cfg['exp3_Ns']:
            target_bytes = cfg['mem_kb'] * 1024
            safe_H = int(np.ceil(np.log(tv) / np.log(1.5)))
            kll_k_t = calibration.find_kll_k_by_budget(target_bytes, H=safe_H)
            gk_alpha = calibration.find_gk_alpha_tight(target_bytes, dist, tv)
            dcs_gamma = calibration.find_dcs_gamma_by_budget(target_bytes, universe=cfg['DCS_uni'])
            meta_log[dist][tv] = {"PrivKLL_k": kll_k_t, "GK_alpha": gk_alpha, "DCS_gamma": dcs_gamma}
            
            kll_mems, gk_mems, dcs_mems = [], [], []
            for _ in range(cfg['trials']):
                errs, kll_mem, gk_mem, dcs_mem = run_multi_trial_strict(
                    cfg['eps'], dist, tv, quantiles, kll_k_t, gk_alpha, dcs_gamma, cfg['DCS_uni']
                )
                kll_mems.append(kll_mem); gk_mems.append(gk_mem); dcs_mems.append(dcs_mem)
                for m in methods:
                    raw_res[dist][m][tv].append(errs[m])

                                  
            print(f"  [Summary] Stream T: {tv:<8} | Dist: {dist.upper()} | Trials: {cfg['trials']}")
            print(f"    ├─ GK-Indep: {np.mean(raw_res[dist]['DPExpGK'][tv]):.5f} | Avg Mem: {np.mean(gk_mems)/1024:.2f} KB")
            print(f"    ├─ KLL-Indep: {np.mean(raw_res[dist]['PrivKLL-Q'][tv]):.5f} | Avg Mem: {np.mean(kll_mems)/1024:.2f} KB")
            print(f"    ├─ KLL-Joint: {np.mean(raw_res[dist]['PrivKLL-JQ'][tv]):.5f} | Avg Mem: {np.mean(kll_mems)/1024:.2f} KB")
            print(f"    └─ DCS DP: {np.mean(raw_res[dist]['DCS DP'][tv]):.5f} | Avg Mem: {np.mean(dcs_mems)/1024:.2f} KB")
            print("    " + "-"*65)

    end = time.time()
    print(f"\n▶ Exp Continual Mult-q completed in {end - start:.2f} seconds.")

    plot_data = {dist: {m: {'mean': [], 'p10': [], 'p90': []} for m in methods} for dist in cfg['distributions']}
    for dist in cfg['distributions']:
        for m in methods:
            for tv in cfg['exp3_Ns']:
                arr = raw_res[dist][m][tv]
                plot_data[dist][m]['mean'].append(np.mean(arr))
                plot_data[dist][m]['p10'].append(np.percentile(arr, 10))
                plot_data[dist][m]['p90'].append(np.percentile(arr, 90))

    with open(RESULTS / f"exp_m_N_{timestamp}.json", 'w') as f: 
        json.dump({"meta": meta_log, "data": plot_data}, f, indent=4)
    plot_vldb_multi_fig(cfg['exp3_Ns'], plot_data, 'Stream Length ($T$)', True, 10, RESULTS / f"exp_m_N_{timestamp}.png")
    print(f"\n▶ Exp multi_N results saved to {RESULTS / f'exp_m_N_{timestamp}.json'} and {RESULTS / f'exp_m_N_{timestamp}.png'}")

# ------------------------------------------------------------------------------
# 5. 主入口控制中心
# ------------------------------------------------------------------------------
def main(logger=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="Quick debug mode")
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    cfg = {
        'N': 100_000,
        'M': 9,
        'mem_kb': 8, 
        'distributions': ['uniform', 'normal', 'zipf'],
        'eps': 2.0,
        'trials': 1,

        # 实验三 (Varying N)
        'exp3_Ns': [10_000, 100_000, 1_000_000, 10_000_000, 100_000_000, 1_000_000_000],
    }
    
    if args.fast:
        cfg['trials'] = 2
        cfg['exp3_Ns'] = [100_000, 1_000_000]
        cfg['distributions'] = ['uniform', 'normal', 'zipf']

    if logger is None:
        logger = get_default_logger()
    logger.info("Loading multiple quantiles data pipeline...")
    logger.info("Executing Multi-Quantile Benchmark Experiments...")


    run_experiment_m_N(cfg, timestamp)

    logger.info("=" * 80)
    logger.info("ALL MULTI-QUANTILE EXPERIMENTS EXECUTED SUCCESSFULLY")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()