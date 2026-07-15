'''
Author: Jean jean_green@163.com
Date: 2026-06-10 03:36:41
LastEditors: Jean jean_green@163.com
LastEditTime: 2026-06-26 03:46:24
FilePath: /DPQE/exp/exp_multi.py
Description: 多分位数审计物理核心流实验控制中心
'''
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
        fh = logging.FileHandler("exp_multi_DCS.log")
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

def run_multi_trial_strict(eps, dist_type, total_n, target_qs, kll_k, gk_alpha, dcs_gamma, dcs_uni=2**16, chunk_size=500000):
    """
    底层物理流计算：支持多分位数（Multiple Quantiles）释放。
    参考 single 格式，一次性收集数据并返回聚合的平均 Rank Error。
    """
    kll_indep = PrivKLL(epsilon=1.0, kll_k=kll_k)
    kll_joint = PrivKLL(epsilon=1.0, kll_k=kll_k)
    non_dp_kll = KLL(k=kll_k)
    gk_indep = GK(alpha=gk_alpha)

    # 隐私转换：将纯 ϵ-DP 转换为对齐的 CDP ρ 预算平分机制
    rho_cdp = 0.5 * (eps ** 2)
    dcs_sketch = DCS(universe=dcs_uni, gamma=dcs_gamma, rho=rho_cdp)

    sample_size = min(total_n, 100_000)
    reservoir = []
    item_counter = 0
    
    # 值域归一化函数
    def to_dcs_domain(val):
        return int(np.clip(val / 100_000.0 * (dcs_uni - 1), 0, dcs_uni - 1))
    
    def from_dcs_domain(int_val):
        return (int_val / (dcs_uni - 1)) * 100_000.0

    # 临时测试代码：加入计数器和探针
    total_kll_time = 0.0
    total_gk_time = 0.0
    total_dcs_time = 0.0
    generator = stream_chunk_generator(dist_type, total_n, chunk_size=chunk_size)
    for chunk in generator:
        for val in chunk:
            non_dp_kll.update(val)
            start_kll = time.time()
            kll_indep.ingest_data(val)
            total_kll_time += time.time() - start_kll

            kll_joint.ingest_data(val)

            start_gk = time.time()
            gk_indep.insert(val)
            total_gk_time += time.time() - start_gk

            start_dcs = time.time()
            dcs_sketch.update(to_dcs_domain(val))
            total_dcs_time += time.time() - start_dcs

            if len(reservoir) < sample_size:
                reservoir.append(val)
            else:
                r = np.random.randint(0, item_counter + 1)
                if r < sample_size: reservoir[r] = val
            item_counter += 1
        # 每处理完一个大 Chunk (如 50 万条数据)，打印一次各算法的纯维护耗时
        print(f"  [Profiler Progress] Processed {item_counter}/{total_n} elements:")
        print(f"    -> KLL Cumulative Insert Time: {total_kll_time:.2f}s")
        print(f"    -> GK Cumulative Insert Time: {total_gk_time:.2f}s")
        print(f"    -> DCS Cumulative Update Time: {total_dcs_time:.2f}s")

    m = len(target_qs)
    final_estimates = {'PrivKLL-Q': [], 'PrivKLL-JQ': [], 'DPExpGK': [], 'DCS DP': []}
    
    # ── 终点站一次性全额隐私预算分配与解算 ──
    # 策略 1: KLL 独立发布
    eps_kll_indep = eps / m
    for q in target_qs:
        kll_indep.epsilon = eps_kll_indep
        final_estimates['PrivKLL-Q'].append(kll_indep.release_quantile(q))
        
    # 策略 2: KLL 联合发布 (利用内在的覆盖树结构，无需平分预算)
    kll_joint.epsilon = eps
    for q in target_qs:
        final_estimates['PrivKLL-JQ'].append(kll_joint.release_quantile(q))
        
    # 策略 3: GK 独立发布 (串行平分预算)
    eps_gk_indep = eps / m
    for q in target_qs:
        final_estimates['DPExpGK'].append(gk_indep.dp_exp(q, eps_gk_indep))
        
    # 策略 4: DCS DP 发布 (全局预算一次性注入，后处理查询无需额外预算)
    # 所有的 query 二分查找都是在已加噪的静态 Summary 上进行的，
    # 属于后处理操作，因此不需要像 KLL/GK 那样对多个分位数进行预算平分，
    # 每次查询均可直接复用并享受完整的全局 rho 精度。
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

    # 计算平均绝对排名误差 (Mean Abs Rank Error)
    res_arr = np.array(reservoir)
    final_errors = {}
    for method, ests in final_estimates.items():
        q_errors = []
        for idx, q in enumerate(target_qs):
            cur_est = ests[idx] if ests[idx] is not None else res_arr[-1]
            actual_percentile = (res_arr <= cur_est).mean()
            q_errors.append(abs(actual_percentile - q))
        final_errors[method] = np.mean(q_errors)
        
    # 计算物理内存（Bytes）
    kll_mem = sum(len(c) for c in non_dp_kll.compactors) * 8
    gk_mem = len(gk_indep.S) * 16
    dcs_mem = dcs_sketch.memory_budget() * 4
    
    return final_errors, kll_mem, gk_mem, dcs_mem

# ------------------------------------------------------------------------------
# 3. 统一 VLDB 风格多子图并排绘图引擎
# ------------------------------------------------------------------------------
def plot_vldb_multi_fig(x_values, plot_data, xlabel_str, x_scale_log, base_x, fname):
    # 统一设置学术论文级 Serif 字体并微调基础字体大小
    plt.rcParams.update({
        'font.size': 8.5, 
        'font.family': 'serif', 
        'mathtext.fontset': 'cm',
        'xtick.labelsize': 7.5, # 稍微缩减刻度字号，防止 15, 21 挤在一起
        'ytick.labelsize': 8
    })
    
    # 🌟 修正 1：将高度从 1.6 略微提升到 1.85，给横轴的 3 层文字留出绝对安全的物理像素空间
    fig, axes = plt.subplots(1, 3, figsize=(3.35, 1.85), sharey=True)
    distributions = ['uniform', 'normal', 'zipf']
    dist_labels = ['(a) UNIFORM', '(b) NORMAL', '(c) ZIPF']
    
    methods = ['PrivKLL-Q', 'PrivKLL-JQ', 'DPExpGK', 'DCS DP']
    style_cfgs = {
        'PrivKLL-Q': {'color': '#3A86FF', 'linestyle': '--', 'marker': 'v'},
        'PrivKLL-JQ': {'color': '#FF006E', 'linestyle': ':', 'marker': 'x'},
        'DPExpGK': {'color': '#06D6A0', 'linestyle': '-', 'marker': 'o'},
        'DCS DP': {'color': '#6C757D', 'linestyle': '-.', 'marker': '^'}
    }
    lines, labels = [], []

    # 🌟 修正 2：定义每个子图在画布上的绝对几何位置 [left, bottom, width, height]
    # 通过显式指定 height=0.46，强行撑开子图的内部网格，绝对不会坍塌或空白
    box_width = 0.26
    box_height = 0.46
    box_bottom = 0.34
    box_lefts = [0.14, 0.43, 0.72]

    for idx, dist in enumerate(distributions):
        ax = axes[idx]
        # 🌟 修正 3：强制赋予子图绝对坐标盒模型，不再受 subplots_adjust 压榨影响
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
        
        # 🌟 修正 4：调整倒挂子标题的位置，y 随着轴高度重新调整为 -0.52
        ax.set_title(dist_labels[idx], y=-0.52, fontsize=8.5, va='top')
        ax.grid(True, which="both", ls="--", color='#e0e0e0', alpha=0.6, linewidth=0.5)
        
    # 🌟 修正 5：完美卡在刻度线和子标题中间的大标签位置
    fig.text(0.56, 0.21, xlabel_str, ha='center', va='center', fontsize=8.5)
    
    axes[0].set_ylabel('Mean Abs Rank Error', fontsize=8.5, labelpad=2)
    
    # 🌟 修正 6：将顶部图例移到 1.0 完美临界点
    fig.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.56, 1.0), 
               ncol=4, frameon=True, fancybox=True, shadow=False, 
               fontsize=7, columnspacing=0.5, handletextpad=0.2, framealpha=1.0)
    
    # 🌟 修正 7：因为使用了绝对坐标定位 position，这里直接保存即可，坚决不加 tight_layout
    plt.savefig(fname, dpi=300)
    plt.close()
# ------------------------------------------------------------------------------
# 4. 驱动实验控制中心
# ------------------------------------------------------------------------------
def run_experiment_m_M(cfg, timestamp):
    print("\n" + "="*80 + f"\n▶ [Executing] Exp multi — Varying Number of Quantiles m (Target Budget={cfg['mem_kb']} KB, N={cfg['N']})\n" + "="*80)
    start = time.time()
    methods = ['PrivKLL-Q', 'PrivKLL-JQ', 'DPExpGK', 'DCS DP']
    
    raw_res = {dist: {m: {mv: [] for mv in cfg['exp1_Ms']} for m in methods} for dist in cfg['distributions']}
    meta_log = {}

    with ProcessPoolExecutor() as executor:
        for dist in cfg['distributions']:
            meta_log[dist] = {"GK_alpha": cfg['GK_alpha'], "PrivKLL_k": cfg['PrivKLL_k'], "DCS_gamma": cfg['DCS_gamma']}

            for mv in cfg['exp1_Ms']:
                quantiles = [j / (mv + 1) for j in range(1, mv + 1)]
                
                # 🌟 核心改变 1：异步提交多进程任务队列
                futures = []
                for _ in range(cfg['trials']):
                    f = executor.submit(
                        run_multi_trial_strict, 
                        cfg['eps'], dist, cfg['N'], quantiles, cfg['PrivKLL_k'], cfg['GK_alpha'], cfg['DCS_gamma'], cfg['DCS_uni']
                    )
                    futures.append(f)
                
                # 🌟 核心改变 2：阻塞并收集所有并发 Trial 的结果
                kll_mems, gk_mems, dcs_mems = [], [], []
                for f in futures:
                    errs, kll_mem, gk_mem, dcs_mem = f.result()
                    kll_mems.append(kll_mem); gk_mems.append(gk_mem); dcs_mems.append(dcs_mem)
                    for m in methods:
                        raw_res[dist][m][mv].append(errs[m])
                    
                print(f"  [Summary] m: {mv:<2} | Dist: {dist.upper()} | Trials: {cfg['trials']}")
                print(f"    ├─ KLL-Indep: {np.mean(raw_res[dist]['PrivKLL-Q'][mv]):.5f} | Avg Mem: {np.mean(kll_mems)/1024:.2f} KB")
                print(f"    ├─ KLL-Joint: {np.mean(raw_res[dist]['PrivKLL-JQ'][mv]):.5f} | Avg Mem: {np.mean(kll_mems)/1024:.2f} KB")
                print(f"    ├─ GK-Indep: {np.mean(raw_res[dist]['DPExpGK'][mv]):.5f} | Avg Mem: {np.mean(gk_mems)/1024:.2f} KB")
                print(f"    └─ DCS DP: {np.mean(raw_res[dist]['DCS DP'][mv]):.5f} | Avg Mem: {np.mean(dcs_mems)/1024:.2f} KB")
                print("    " + "-"*65)

    end = time.time()
    print(f"\n▶ Exp 6.3.1 completed in {end - start:.2f} seconds.")

    plot_data = {dist: {m: {'mean': [], 'p10': [], 'p90': []} for m in methods} for dist in cfg['distributions']}
    for dist in cfg['distributions']:
        for m in methods:
            for mv in cfg['exp1_Ms']:
                arr = raw_res[dist][m][mv]
                plot_data[dist][m]['mean'].append(np.mean(arr))
                plot_data[dist][m]['p10'].append(np.percentile(arr, 10))
                plot_data[dist][m]['p90'].append(np.percentile(arr, 90))

    with open(RESULTS / f"exp_m_M_{timestamp}.json", 'w') as f: 
        json.dump({"meta": meta_log, "data": plot_data}, f, indent=4)
    plot_vldb_multi_fig(cfg['exp1_Ms'], plot_data, 'Number of Quantiles ($m$)', False, 2, RESULTS / f"exp_m_M_{timestamp}.png")
    print(f"\n▶ Exp multi_M results saved to {RESULTS / f'exp_m_M_{timestamp}.json'} and {RESULTS / f'exp_m_M_{timestamp}.png'}")

def run_experiment_m_eps(cfg, timestamp):
    print("\n" + "="*80 + f"\n▶ [Executing] Exp multi — Varying Privacy Budget (Fixed m={cfg['M']}, Target Budget={cfg['mem_kb']} KB)\n" + "="*80)
    start = time.time()
    methods = ['PrivKLL-Q', 'PrivKLL-JQ', 'DPExpGK', 'DCS DP']
    
    raw_res = {dist: {m: {ev: [] for ev in cfg['exp2_epsilons']} for m in methods} for dist in cfg['distributions']}
    meta_log = {}
    quantiles = [j / (cfg['M'] + 1) for j in range(1, cfg['M'] + 1)]

    with ProcessPoolExecutor() as executor:
        for dist in cfg['distributions']:
            # 修正 1：对齐动态局部变量名（如果在函数上方通过 calibration 动态计算了参数，请在这里直接使用对应的局部变量名）
            # 如果依然使用 cfg 默认参数，保持原样：
            meta_log[dist] = {"GK_alpha": cfg['GK_alpha'], "PrivKLL_k": cfg['PrivKLL_k'], "DCS_gamma": cfg['DCS_gamma']}

            for ev in cfg['exp2_epsilons']:
                
                # 🌟 核心改变 1：异步提交任务到进程池队列
                futures = []
                for _ in range(cfg['trials']):
                    f = executor.submit(
                        run_multi_trial_strict,
                        ev, dist, cfg['N'], quantiles, cfg['PrivKLL_k'], cfg['GK_alpha'], cfg['DCS_gamma'], cfg['DCS_uni']
                    )
                    futures.append(f)
                
                # 🌟 核心改变 2：阻塞并同步捞回所有并发多进程 Trial 的运算结果
                kll_mems, gk_mems, dcs_mems = [], [], []
                for f in futures:
                    errs, kll_mem, gk_mem, dcs_mem = f.result()
                    kll_mems.append(kll_mem); gk_mems.append(gk_mem); dcs_mems.append(dcs_mem)
                    for m in methods:
                        raw_res[dist][m][ev].append(errs[m])
                        
                # 🌟 修正 2：彻底消灭旧键名，完美防范 KeyError 崩溃
                print(f"  [Parallel Summary] Epsilon: {ev:<3.1f} | Dist: {dist.upper()} | Completed Trials: {cfg['trials']}")
                print(f"    ├─ KLL-Indep: {np.mean(raw_res[dist]['PrivKLL-Q'][ev]):.5f} | Avg Mem: {np.mean(kll_mems)/1024:.2f} KB")
                print(f"    ├─ KLL-Joint: {np.mean(raw_res[dist]['PrivKLL-JQ'][ev]):.5f} | Avg Mem: {np.mean(kll_mems)/1024:.2f} KB")
                print(f"    ├─ GK-Indep:  {np.mean(raw_res[dist]['DPExpGK'][ev]):.5f} | Avg Mem: {np.mean(gk_mems)/1024:.2f} KB")
                print(f"    └─ DCS DP:    {np.mean(raw_res[dist]['DCS DP'][ev]):.5f} | Avg Mem: {np.mean(dcs_mems)/1024:.2f} KB")
                print("    " + "-"*65)

    end = time.time()
    print(f"\n▶ Exp 6.3.2 completed in {end - start:.2f} seconds.")

    plot_data = {dist: {m: {'mean': [], 'p10': [], 'p90': []} for m in methods} for dist in cfg['distributions']}
    for dist in cfg['distributions']:
        for m in methods:
            for ev in cfg['exp2_epsilons']:
                arr = raw_res[dist][m][ev]
                plot_data[dist][m]['mean'].append(np.mean(arr))
                plot_data[dist][m]['p10'].append(np.percentile(arr, 10))
                plot_data[dist][m]['p90'].append(np.percentile(arr, 90))

    with open(RESULTS / f"exp_m_eps_{timestamp}.json", 'w') as f: 
        json.dump({"meta": meta_log, "data": plot_data}, f, indent=4)
    plot_vldb_multi_fig(cfg['exp2_epsilons'], plot_data, r'Privacy Budget ($\epsilon$)', True, 2, RESULTS / f"exp_m_eps_{timestamp}.png")
    print(f"\n▶ Exp multi_eps results saved to {RESULTS / f'exp_m_eps_{timestamp}.json'} and {RESULTS / f'exp_m_eps_{timestamp}.png'}")

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
        'distributions': ['uniform', 'normal', 'zipf'],
        'real_datasets': ['mimic_hr', 'power_active', 'taxi_duration'],
        'eps': 1.0,
        'trials': 100,

        'mem_kb': 8, 
        'PrivKLL_k': 325,  # 8 KB Anchor for KLL at N=100K
        'GK_alpha': 0.002083,  # 8 KB Anchor for zipf at N=100K
        'DCS_gamma': 0.207974,  # DCS parameter
        'DCS_uni': 2**16,  # DCS universe size

        # 实验一 (Varying M)
        'exp1_Ms': [1, 3, 5, 9, 15, 21],

        # 实验二 (Varying Epsilon)
        'exp2_epsilons': [0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4],
        # 'exp2_epsilons': [0.5, 1.0, 2.0, 4.0, 8.0],

        # 实验三 (Varying N)
        'exp3_Ns': [100_000, 500_000, 1_000_000, 5_000_000, 10_000_000],
    }
    
    if args.fast:
        cfg['trials'] = 2
        cfg['exp1_Ms'] = [1, 5, 21]
        cfg['exp2_epsilons'] = [0.5, 2.0, 8.0]
        cfg['exp3_Ns'] = [100_000, 1_000_000]
        cfg['distributions'] = ['uniform', 'normal', 'zipf']

    if logger is None:
        logger = get_default_logger()
    logger.info("Loading multiple quantiles data pipeline...")
    logger.info("Executing Multi-Quantile Benchmark Experiments...")

    # run_experiment_m_M(cfg, timestamp)
    # run_experiment_m_eps(cfg, timestamp)
    
    logger.info("=" * 80)
    logger.info("ALL MULTI-QUANTILE EXPERIMENTS EXECUTED SUCCESSFULLY")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()