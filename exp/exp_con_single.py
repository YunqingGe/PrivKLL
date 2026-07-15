import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import json
import argparse
from datetime import datetime
import logging
from pathlib import Path
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
    logger = logging.getLogger("exp_single")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler("exp_single_release.log")
        ch = logging.StreamHandler()
        fmt = logging.Formatter("[%(asctime)s] %(message)s")
        fh.setFormatter(fmt)
        ch.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger

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
    if dist_type == 'normal':
        return np.clip(np.random.normal((high+low)/2, (high-low)/6, n), low, high)
    elif dist_type == 'zipf':
        return np.clip(np.random.pareto(2.0, n) * (high/10), low, high)
    elif dist_type == 'mimic':
        if not data_path:
            raise ValueError("选择 'mimic' 数据集时必须指定 --data_path")
        return load_mimic_data(data_path, n)
    return np.random.uniform(low, high, n)

def run_single_trial(trial_idx, data, gk_cps, gk_budgets, kll_cps, kll_budgets_dict, cfg):
    print(f"-> Trial [{trial_idx + 1}] 开始跑流...")
    T_gk = len(gk_cps)
    T_kll = len(kll_cps)
    
    models = {
        'PrivKLL-uniform': PrivKLL(epsilon=cfg['epsilon'], kll_k=cfg['PrivKLL_k']),
        'PrivKLL-linear': PrivKLL(epsilon=cfg['epsilon'], kll_k=cfg['PrivKLL_k']),
        'PrivKLL-exponential': PrivKLL(epsilon=cfg['epsilon'], kll_k=cfg['PrivKLL_k']),
        'ExpGK': GK(alpha=cfg['GK_alpha'])
    }
    
    gt_sketch = KLL(k=cfg['PrivKLL_k'])
    last_estimates = {m: None for m in models}
    kll_cp_idx = {m: 0 for m in ['PrivKLL-uniform', 'PrivKLL-linear', 'PrivKLL-exponential']}
    gk_cp_idx = 0
    
    global_eval_steps = sorted(list(set(gk_cps + kll_cps)))
    errors_history = {m: [] for m in models}
    
    for i, val in enumerate(data):
        n_current = i + 1
        for m in ['PrivKLL-uniform', 'PrivKLL-linear', 'PrivKLL-exponential']:
            models[m].ingest_data(val)
        models['ExpGK'].insert(val)
        gt_sketch.update(val)
            
        curr_kll_idx = kll_cp_idx['PrivKLL-uniform']
        if curr_kll_idx < T_kll and n_current == kll_cps[curr_kll_idx]:
            safe_idx = min(curr_kll_idx, T_kll - 1)
            for m in ['PrivKLL-uniform', 'PrivKLL-linear', 'PrivKLL-exponential']:
                models[m].epsilon = kll_budgets_dict[m][safe_idx] 
                est = models[m].release_quantile(cfg['q'])
                if est is not None: last_estimates[m] = est
                kll_cp_idx[m] += 1

        if gk_cp_idx < T_gk and n_current == gk_cps[gk_cp_idx]:
            eps_t = gk_budgets[gk_cp_idx]
            est = models['ExpGK'].dp_exp(cfg['q'], eps_t) if hasattr(models['ExpGK'], 'dp_exp') else models['ExpGK'].query(cfg['q'])
            if est is not None: last_estimates['ExpGK'] = est
            gk_cp_idx += 1
            
        if n_current in global_eval_steps:
            for m in models:
                current_est = last_estimates[m] if last_estimates[m] is not None else val
                actual_percentile = gt_sketch.rank(current_est) / n_current
                errors_history[m].append(abs(actual_percentile - cfg['q']))
                
    return errors_history

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

# ------------------------------------------------------------------------------
# 🌟 核心修改：精细化图像渲染函数（Y 轴对齐图片区间）
# ------------------------------------------------------------------------------
def plot_vldb_single_column(agg_data, xlabel_str, fname, skip_backup=False):
    if not skip_backup:
        json_fname = Path(fname).with_suffix('.json')
        json_output_file = RESULTS / json_fname
        try:
            with open(json_output_file, 'w', encoding='utf-8') as f:
                json.dump(agg_data, f, indent=4, ensure_ascii=False)
            print(f"数据备份成功 -> {json_output_file}")
        except Exception as e:
            print(f"警告：数据 JSON 备份失败: {e}")

    plt.rcParams.update({
        'font.size': 8.5, 
        'font.family': 'serif', 
        'mathtext.fontset': 'cm',
        'xtick.labelsize': 7.5, 
        'ytick.labelsize': 8
    })
    
    fig, ax = plt.subplots(figsize=(4.2, 2.2))
    methods = ['PrivKLL-uniform', 'PrivKLL-linear', 'PrivKLL-exponential', 'ExpGK']
    display_names = {
        'PrivKLL-uniform': 'PrivKLL-Uni', 
        'PrivKLL-linear': 'PrivKLL-Linear',
        'PrivKLL-exponential': 'PrivKLL-Exp', 
        'ExpGK': 'DPExpGK'
    }
    style_cfgs = {
        'PrivKLL-uniform': {'color': '#0A2472', 'linestyle': '--' },
        'PrivKLL-linear': {'color': '#0077B6', 'linestyle': ':'},
        'PrivKLL-exponential': {'color': '#90CAF9', 'linestyle': '-.'},
        'ExpGK': {'color': '#06D6A0', 'linestyle': '-'}
    }
    SHOW_BAND = {'PrivKLL-uniform', 'ExpGK'}   
    
    lines, labels = [], []
    first_method = list(agg_data.keys())[0]
    x_values = np.array(agg_data[first_method]['ns'])

    for m in methods:
        if m not in agg_data: continue
        mean = np.clip(np.array(agg_data[m]['mean']), 1e-9, None)
        p10 = np.clip(np.array(agg_data[m]['p10']), 1e-9, None)
        p90 = np.clip(np.array(agg_data[m]['p90']), 1e-9, None)
        cur_x = x_values[:len(mean)]
        
        # 1. 绘制主线（使用 style_cfgs 中对应的颜色）
        line, = ax.plot(cur_x, mean, 
                         color=style_cfgs[m]['color'], 
                         linestyle=style_cfgs[m]['linestyle'], 
                         linewidth=1.0, alpha=0.95, zorder=4)
        
        # 2. 🌟 修复误差棒/置信阴影颜色：直接使用当前方法独有的颜色，完美对齐
        if m in SHOW_BAND:
            ax.fill_between(cur_x, p10, p90, 
                            color=style_cfgs[m]['color'],  # 确保阴影和主线颜色绝对一致
                            alpha=0.05,                   # 略微调深一点点，防止太淡看不见
                            zorder=3)
            
        lines.append(line)
        labels.append(display_names[m])
    
    # X 轴对数坐标格式化
    ax.set_xscale('log', base=10)
    ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=5))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"$10^{{{int(np.log10(x))}}}$" if x > 0 else "0"))
    
    # Y 轴对数坐标格式化
    ax.set_yscale('log', base=2)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: f"$2^{{{int(np.log2(y))}}}$" if y > 0 else "0"))
    
    # 🌟 Y 轴范围精确限定，保持与目标图像一致：从 2^-8 覆盖到 2^0
    ax.set_ylim(2**-8, 1.0)  
    
    ax.set_xlabel(xlabel_str, fontsize=8.5, labelpad=3)
    ax.set_ylabel('Mean Abs Rank Error', fontsize=8.5, labelpad=2)
    ax.grid(True, which="both", ls="--", color='#e8e8e8', alpha=0.5, linewidth=0.4, zorder=1)

    # 🌟 调整红色的 Max Allowed Error 水平基准线位置
    ax.axhline(y=0.5, color='#d90429', linestyle=':', linewidth=0.8, alpha=0.8, zorder=5)
    ax.text(x=x_values[-1] * 0.4, y=0.55, s=r'$\text{Max Allowed Error (0.5)}$', color='#d90429', fontsize=7.5, ha='right', va='bottom', alpha=0.9, zorder=5)
    
    ax.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=4, frameon=True, fancybox=True, shadow=False, fontsize=7.5, columnspacing=0.4, handletextpad=0.2, handlelength=1.2, framealpha=1.0)
    
    plt.tight_layout()
    fig.subplots_adjust(top=0.85)
    output_file = RESULTS / fname
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"高精细学术图成功渲染 -> {output_file}")
    plt.close()

def run_experiment_single(cfg, timestamp):
    print("\n" + "="*70)
    print("RUNNING EXP: CONTINUAL N (Metric: Average Absolute Rank Error)")
    print("="*70)
    
    for dist in cfg['distributions']:
        print(f"\n--- Dataset/Distribution: {dist.upper()} ---")
        raw_data = make_data(dist, cfg['N'], cfg['domain'], cfg.get('data_path'))
        actual_N = len(raw_data)
        
        print(f"流属性：实际样本数(N) = {actual_N}, 目标值域(Domain) = {cfg['domain']}")

        gk_cps = []
        curr = cfg['n_min']
        while curr < actual_N:
            gk_cps.append(int(curr))
            curr = curr * (1 + cfg['GK_alpha']/2)
        if not gk_cps or gk_cps[-1] != actual_N:
            gk_cps.append(actual_N)
        gk_cps = sorted(list(set(gk_cps)))
        gk_budgets = [cfg['epsilon'] / len(gk_cps)] * len(gk_cps)

        print("正在利用公共知识进行单次无偏 KLL Checkpoint 序列捕捉...")
        kll_cps = CheckpointScheduler.generate_kll_cps(
            n_min=cfg['n_min'], N=actual_N, kll_k=cfg['PrivKLL_k'], c=2.0/3.0,
            data=None, domain=cfg['domain'], verbose=False
        )
        
        kll_sched_obj = CheckpointScheduler(cfg['n_min'], actual_N, kll_k=cfg['PrivKLL_k'], preset_cps=kll_cps)
        kll_budgets_dict = {
            'PrivKLL-uniform': kll_sched_obj.budgets(cfg['epsilon'], 'uniform', custom_cps=kll_cps),
            'PrivKLL-linear': kll_sched_obj.budgets(cfg['epsilon'], 'linear', custom_cps=kll_cps),
            'PrivKLL-exponential': kll_sched_obj.budgets(cfg['epsilon'], 'exponential', custom_cps=kll_cps)
        }

        global_steps = sorted(list(set(gk_cps + kll_cps)))
        
        print(f"开始执行多进程试验并发分配，总计轮次 Trials = {cfg['trials']}...")
        trials_raw = []
        
        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(run_single_trial, t, raw_data, gk_cps, gk_budgets, kll_cps, kll_budgets_dict, cfg)
                for t in range(cfg['trials'])
            ]
            for fut in futures:
                trials_raw.append(fut.result())
            
        agg = aggregate_trials(trials_raw, global_steps)
        
        print(f"\n{'Eps':<6} | {'KLL-Uniform':<15} | {'KLL-Linear':<15} | {'KLL-Exponential':<20} | {'ExpGK':<15}")
        print(f"{cfg['epsilon']:<6.2f} | {np.mean(agg['PrivKLL-uniform']['mean']):<15.5f} | "
                f"{np.mean(agg['PrivKLL-linear']['mean']):<15.5f} | "
                f"{np.mean(agg['PrivKLL-exponential']['mean']):<20.5f} | "
                f"{np.mean(agg['ExpGK']['mean']):<15.5f}")
        
        xlabel = "Stream Sequence Length (N)"
        fig_name = f"exp_c_{dist}_eps{cfg['epsilon']}_{timestamp}.png"
        plot_vldb_single_column(agg, xlabel, fig_name)
        
    print("\n实验全部结束。学术高清图已保存至 'results/' 文件夹。")

# ------------------------------------------------------------------------------
# 🌟 核心修改：支持外部给定的 JSON 文件独立 Replot
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="DP Stream Quantile Benchmark")
    parser.add_argument("--dist", type=str, default="mimic", choices=["mimic", "zipf"], help="指定数据集类型")
    parser.add_argument("--data_path", type=str, default="DPQE/data/mimic_hr.npy", help="mimic 数据路径")
    parser.add_argument("--trials", type=int, default=30, help="设置运行的 Trial 轮数（支持高并发并行）")
    parser.add_argument("--replot", type=str, default=None, help="传入历史备份好的 JSON 数据路径，直接重绘出图")
    args = parser.parse_args()
    
    # 🌟 Replot 分支：彻底切断耗时的多进程流推进模拟，直接提取数据重绘图像
    if args.replot is not None:
        json_path = Path(args.replot)
        if not json_path.exists():
            raise FileNotFoundError(f"未找到指定的 JSON 文件: {args.replot}")
            
        print(f"==== 进入 REPLOT 渲染模式 ====")
        print(f"正在读取历史缓存数据: {json_path.name}")
        with open(json_path, 'r', encoding='utf-8') as f:
            cached_agg_data = json.load(f)
            
        xlabel = "Stream Sequence Length (N)"
        replot_fig_name = f"replot_{json_path.stem}.png"
        plot_vldb_single_column(cached_agg_data, xlabel, replot_fig_name, skip_backup=True)
        print("Replot 绘图指令安全执行完毕。")
        return

    # 标准实验流分支
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
        'n_min': 1000,
        'domain': target_domain,   
        'distributions': [args.dist],
        'data_path': args.data_path,
        'epsilon': 2.0,
        'trials': args.trials,  
        'q': 0.5,
        'PrivKLL_k': 1343,  
        'GK_alpha': 0.000575,  
    }
    
    logger.info(f"开启优化加速流水线，数据类型: {args.dist.upper()}, Trials: {args.trials}, N: {target_N}")
    run_experiment_single(cfg, timestamp)

if __name__ == "__main__":
    main()