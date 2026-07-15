import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import json
import argparse
from datetime import datetime
import logging
from pathlib import Path
import sys
from concurrent.futures import ProcessPoolExecutor

# 统一从外部核心组件和校准模块导入
from DPQE.src.PrivKLL import PrivKLL
from DPQE.src.Greenwald_Khanna import GK
from DPQE.src.scheduler import CheckpointScheduler 
from DPQE.src.kll import KLL

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

def get_deep_sizeof(obj, seen=None):
    """递归计算各数据结构真实占用的物理内存(Bytes)"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)
    
    if isinstance(obj, dict):
        size += sum([get_deep_sizeof(v, seen) for v in obj.values()])
        size += sum([get_deep_sizeof(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_deep_sizeof(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_deep_sizeof(i, seen) for i in obj])
    return size

def load_mimic_data(data_path, n_max):
    p = Path(data_path)
    if not p.exists():
        raise FileNotFoundError(f"未找到预处理好的 MIMIC 数据文件: {data_path}")
    print(f"正在从 {p.name} 加载预处理数据...")
    data = np.load(p)
    data = data.flatten()
    if len(data) > n_max:
        data = data[:n_max]
    return data

def make_data(dist_type, n, domain, data_path=None):
    low, high = domain
    if dist_type == 'zipf':
        return np.clip(np.random.pareto(2.0, n) * (high/10), low, high)
    elif dist_type == 'mimic':
        if not data_path:
            raise ValueError("选择 'mimic' 数据集时必须指定 --data_path")
        return load_mimic_data(data_path, n)
    return np.random.uniform(low, high, n)

# ------------------------------------------------------------------------------
# 优化后的单次 Trial 核心流（多进程任务单元）
# ------------------------------------------------------------------------------
def run_single_trial(trial_idx, data, gk_cps, gk_budgets, kll_cps, kll_budgets_dict, cfg):
    print(f"-> Trial [{trial_idx + 1}] 进入核心计算流...")
    T_gk = len(gk_cps)
    T_kll = len(kll_cps)
    
    models = {
        'PrivKLL-Q': PrivKLL(epsilon=cfg['epsilon'], kll_k=cfg['PrivKLL_k']),
        'PrivKLL-JQ': PrivKLL(epsilon=cfg['epsilon'], kll_k=cfg['PrivKLL_k']),
        'ExpGK': GK(alpha=cfg['GK_alpha'])
    }
    
    # 🌟 核心提速武器：无偏高精度 KLL 真相树，彻底消除 O(N) 切片
    gt_sketch = KLL(k=cfg['PrivKLL_k'], c=2.0/3.0) 
    
    last_estimates = {m: {q: None for q in cfg['qs']} for m in models}
    kll_cp_idx = {m: 0 for m in ['PrivKLL-Q', 'PrivKLL-JQ']}
    gk_cp_idx = 0
    
    global_eval_steps = sorted(list(set(gk_cps + kll_cps)))
    errors_history = {m: [] for m in models}
    
    for i, val in enumerate(data):
        n_current = i + 1
        
        for m in ['PrivKLL-Q', 'PrivKLL-JQ']:
            models[m].ingest_data(val)
        models['ExpGK'].insert(val)
        gt_sketch.update(val) # 真相树同步前推
            
        curr_kll_idx = kll_cp_idx['PrivKLL-Q']
        if curr_kll_idx < T_kll and n_current == kll_cps[curr_kll_idx]:
            safe_idx = min(curr_kll_idx, T_kll - 1)
            
            # 1. PrivKLL-Q
            m_q = 'PrivKLL-Q'
            eps_per_q = kll_budgets_dict[m_q][safe_idx] / len(cfg['qs'])
            models[m_q].epsilon = eps_per_q
            for q in cfg['qs']:
                est = models[m_q].release_quantile(q)
                if est is not None: 
                    last_estimates[m_q][q] = est
            kll_cp_idx[m_q] += 1
            
            # 2. PrivKLL-JQ
            m_jq = 'PrivKLL-JQ'
            models[m_jq].epsilon = kll_budgets_dict[m_jq][safe_idx]
            
            ests = models[m_jq].release_multi_quantiles(cfg['qs'])
            
            if ests is not None and len(ests) == len(cfg['qs']):
                for idx, q in enumerate(cfg['qs']):
                    last_estimates[m_jq][q] = ests[idx]
            kll_cp_idx[m_jq] += 1

        # 触发 ExpGK 发布
        if gk_cp_idx < T_gk and n_current == gk_cps[gk_cp_idx]:
            eps_t = gk_budgets[gk_cp_idx]
            for q in cfg['qs']:
                est = models['ExpGK'].dp_exp(q, eps_t) if hasattr(models['ExpGK'], 'dp_exp') else models['ExpGK'].query(q)
                if est is not None: 
                    last_estimates['ExpGK'][q] = est
            gk_cp_idx += 1
            
        # 考察多目标分位数的平均相对秩误差
        if n_current in global_eval_steps:
            # 🌟 仅在第 1 轮 Trial 时，在控制台打印实时内存监控看板，避免日志刷屏
            if trial_idx == 0:
                print(f"\n[Checkpoint: N = {n_current}] 内存与状态看板:")
                for m, model_obj in models.items():
                    mem_mb = get_deep_sizeof(model_obj) / (1024 * 1024)
                    print(f"  -> Method: {m:<15} | Memory: {mem_mb:.4f} MB")
                print(f"  -> GroundTruth-KLL    | Memory: {get_deep_sizeof(gt_sketch)/(1024*1024):.4f} MB")
                print("-" * 55)

            for m in models:
                q_errors = []
                for q in cfg['qs']:
                    current_est = last_estimates[m][q] if last_estimates[m][q] is not None else val
                    # 🌟 速度质跃：直接利用 KLL 树查询 Rank
                    actual_percentile = gt_sketch.rank(current_est) / n_current
                    q_errors.append(abs(actual_percentile - q))
                errors_history[m].append(np.mean(q_errors))
                
    return errors_history

# ------------------------------------------------------------------------------
# 实验管理与精细化单栏绘图
# ------------------------------------------------------------------------------
def aggregate_trials(trials_raw, eval_steps):
    methods = trials_raw[0].keys()
    agg = {}
    for m in methods:
        matrix = np.array([t[m] for t in trials_raw])
        aae_mean = np.mean(matrix, axis=0)
        p10 = np.percentile(matrix, 10, axis=0)
        p90 = np.percentile(matrix, 90, axis=0)
        agg[m] = {
            'ns': eval_steps,
            'mean': aae_mean.tolist(),
            'p10': p10.tolist(),
            'p90': p90.tolist()
        }
    return agg

def plot_vldb_single_column_refined(agg_data, xlabel_str, fname):
    json_fname = Path(fname).with_suffix('.json')
    json_output_file = RESULTS / json_fname
    try:
        with open(json_output_file, 'w', encoding='utf-8') as f:
            json.dump(agg_data, f, indent=4, ensure_ascii=False)
        print(f"数据备份成功 -> {json_output_file}")
    except Exception as e:
        print(f"警告：数据备份失败 -> {e}")

    plt.rcParams.update({
        'font.size': 8.5, 'font.family': 'serif', 'mathtext.fontset': 'cm',
        'xtick.labelsize': 7.5, 'ytick.labelsize': 8
    })
    
    fig, ax = plt.subplots(figsize=(4.2, 2.2))
    methods = ['PrivKLL-Q', 'PrivKLL-JQ', 'ExpGK']
    style_cfgs = {
        'PrivKLL-Q': {'color': '#3A86FF', 'linestyle': '--'},
        'PrivKLL-JQ': {'color': '#FF006E', 'linestyle': ':'},
        'ExpGK': {'color': '#06D6A0', 'linestyle': '-'}
    }
    
    lines, labels = [], []
    first_method = list(agg_data.keys())[0]
    x_values = np.array(agg_data[first_method]['ns'])

    for m in methods:
        if m not in agg_data: continue
        mean = np.clip(np.array(agg_data[m]['mean']), 1e-7, None)
        p10 = np.clip(np.array(agg_data[m]['p10']), 1e-7, None)
        p90 = np.clip(np.array(agg_data[m]['p90']), 1e-7, None)
        cur_x = x_values[:len(mean)]
        
        line, = ax.plot(cur_x, mean, color=style_cfgs[m]['color'], linestyle=style_cfgs[m]['linestyle'], 
                         marker=None, linewidth=0.8, alpha=0.9, zorder=4)
        ax.fill_between(cur_x, p10, p90, color=style_cfgs[m]['color'], alpha=0.04, zorder=3)
        lines.append(line)
        labels.append(m)
    
    ax.set_xscale('log', base=10)
    ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=5))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"$10^{{{int(np.log10(x))}}}$" if x > 0 else "0"))
    ax.set_yscale('log', base=2)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: f"$2^{{{int(np.log2(y))}}}$" if y > 0 else "0"))
    
    ax.set_xlabel(xlabel_str, fontsize=8.5, labelpad=3)
    ax.set_ylabel('Mean Abs Rank Error', fontsize=8.5, labelpad=2)
    ax.grid(True, which="both", ls="--", color='#e0e0e0', alpha=0.6, linewidth=0.4)
    
    ax.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 1.25), 
              ncol=3, frameon=True, fancybox=True, shadow=False, 
              fontsize=7.5, columnspacing=0.5, handletextpad=0.2, handlelength=1.2, framealpha=1.0)
    
    plt.tight_layout()
    fig.subplots_adjust(top=0.85)
    output_file = RESULTS / fname
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

# ------------------------------------------------------------------------------
# 实验主循环控制流
# ------------------------------------------------------------------------------
def run_experiment_multi(cfg, timestamp):
    print("\n" + "="*70)
    print("RUNNING EXP: CONTINUAL MULTI-QUANTILE (Metric: Avg Absolute Rank Error)")
    print("="*70)
    
    for dist in cfg['distributions']:
        print(f"\n--- Dataset/Distribution: {dist.upper()} ---")
        raw_data = make_data(dist, cfg['N'], cfg['domain'], cfg.get('data_path'))
        actual_N = len(raw_data)
        
        print(f"流属性：实际总长度(N) = {actual_N}, 目标值域(Domain) = {cfg['domain']}")

        # ── 1. 生成 GKExp Checkpoints ──
        gk_cps = []
        curr = cfg['n_min']
        while curr < actual_N:
            gk_cps.append(int(curr))
            curr = curr * (1 + cfg['GK_alpha']/2)
        if not gk_cps or gk_cps[-1] != actual_N:
            gk_cps.append(actual_N)
        gk_cps = sorted(list(set(gk_cps)))
        gk_budgets = [cfg['epsilon'] / len(gk_cps)] * len(gk_cps)

        # ── 2. 维持原始逻辑：单次无隐私泄露 KLL 探测 Checkpoints ──
        print("正在基于公共知识探测生成全局唯一的 KLL 跃迁 Checkpoints 序列...")
        kll_cps = CheckpointScheduler.generate_kll_cps(
            n_min=cfg['n_min'], N=actual_N, kll_k=cfg['PrivKLL_k'], c=2.0/3.0,
            data=None, domain=cfg['domain'], verbose=False
        )
        
        # ── 3. 一次性分配多策略隐私预算账本 ──
        kll_sched_obj = CheckpointScheduler(cfg['n_min'], actual_N, kll_k=cfg['PrivKLL_k'], c=2.0/3.0, preset_cps=kll_cps)
        kll_budgets_dict = {
            'PrivKLL-Q': kll_sched_obj.budgets(cfg['epsilon'], 'uniform', custom_cps=kll_cps),     
            'PrivKLL-JQ': kll_sched_obj.budgets(cfg['epsilon'], 'uniform', custom_cps=kll_cps) 
        }

        global_steps = sorted(list(set(gk_cps + kll_cps)))
        
        # ── 4. 核心升级：多进程并发分配 Trials ──
        print(f"Trials = {cfg['trials']}...")
        trials_raw = []
        
        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(run_single_trial, t, raw_data, gk_cps, gk_budgets, kll_cps, kll_budgets_dict, cfg)
                for t in range(cfg['trials'])
            ]
            for fut in futures:
                trials_raw.append(fut.result())
            
        agg = aggregate_trials(trials_raw, global_steps)
        
        # 控制台看板汇总
        print(f"\n{'Eps':<6} | {'PrivKLL-Q':<15} | {'PrivKLL-JQ':<15} | {'ExpGK':<15}")
        print(f"{cfg['epsilon']:<6.2f} | {np.mean(agg['PrivKLL-Q']['mean']):<15.5f} | "
                f"{np.mean(agg['PrivKLL-JQ']['mean']):<15.5f} | "
                f"{np.mean(agg['ExpGK']['mean']):<15.5f}")
        
        xlabel = "Stream Sequence Length (N)"
        fig_name = f"exp_c_multi_{dist}_{timestamp}.png"
        plot_vldb_single_column_refined(agg, xlabel, fig_name)
        
    print("\n多分位数新版并发发布实验全部圆满完成。")

def main():
    parser = argparse.ArgumentParser(description="DP Stream Multi-Quantile Benchmark")
    parser.add_argument("--dist", type=str, default="mimic", choices=["mimic", "zipf"], help="指定数据集类型")
    parser.add_argument("--data_path", type=str, default="DPQE/data/mimic_hr.npy", help="心率流数据路径")
    parser.add_argument("--trials", type=int, default=30, help="设置运行的总试验轮数(多核并发)")
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = get_default_logger()
    
    if args.dist == 'mimic':
        target_N = 8_700_000         
        target_domain = (30.0, 220.0) 
    else:
        target_N = 10_000_000        
        target_domain = (0.0, 100000.0)
    
    cfg = {
        'N': target_N,
        'n_min': 10000,
        'domain': target_domain,
        'distributions': [args.dist],
        'data_path': args.data_path,
        'epsilon': 2.0,
        'trials': args.trials,
        'qs': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],       
        'PrivKLL_k': 1343,  
        'GK_alpha': 0.000575,  
    }
    
    logger.info(f"开启高速加速流水线，类别: {args.dist.upper()}, 总并发轮次: {args.trials}")
    if args.dist == 'mimic' and not Path(args.data_path).exists():
        logger.error(f"核心文件不存在: {args.data_path}")
        return

    run_experiment_multi(cfg, timestamp)

if __name__ == "__main__":
    main()