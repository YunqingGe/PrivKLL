import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

def plot_onetime_single_column(x_values, plot_data, xlabel_str, x_scale_log, base_x, fname):
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
        'DCS DP': {'color': "#6C757D", 'linestyle': '-.', 'marker': '^'}
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

def plot_continual_single_column(agg_data, xlabel_str, fname):
    json_fname = Path(fname).with_suffix('.json')
    json_output_file = RESULTS / json_fname
    try:
        with open(json_output_file, 'w', encoding='utf-8') as f:
            json.dump(agg_data, f, indent=4, ensure_ascii=False)
        print(f"数据备份成功 -> {json_output_file}")
    except Exception as e:
        print(f"警告：数据备份失败 -> {e}")

    # 绘图样式加载
    plt.rcParams.update({
        'font.size': 8.5, 
        'font.family': 'serif', 
        'mathtext.fontset': 'cm',
        'xtick.labelsize': 7.5, 
        'ytick.labelsize': 8
    })
    
    fig, ax = plt.subplots(figsize=(3.35, 2.2))
    
    # 严格匹配你的三个核心模型
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
        if m not in agg_data:
            continue
            
        mean = np.clip(np.array(agg_data[m]['mean']), 1e-7, None)
        p10 = np.clip(np.array(agg_data[m]['p10']), 1e-7, None)
        p90 = np.clip(np.array(agg_data[m]['p90']), 1e-7, None)
        cur_x = x_values[:len(mean)]
        
        line, = ax.plot(cur_x, mean, color=style_cfgs[m]['color'], linestyle=style_cfgs[m]['linestyle'], 
                         marker=None, linewidth=0.8, alpha=0.9, zorder=4)
        ax.fill_between(cur_x, p10, p90, color=style_cfgs[m]['color'], alpha=0.05)
        
        lines.append(line)
        labels.append(m)
    
    ax.set_xscale('log', base=10)
    ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=5))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"$10^{{{int(np.log10(x))}}}$" if x > 0 else "0"))
    ax.set_yscale('log', base=2)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: f"$2^{{{int(np.log2(y))}}}$" if y > 0 else "0"))
    
    ax.set_xlabel(xlabel_str, fontsize=8.5, labelpad=3)
    ax.set_ylabel('Mean Abs Rank Error', fontsize=8.5, labelpad=2)
    ax.grid(True, which="both", ls="--", color='#e0e0e0', alpha=0.6, linewidth=0.5)
    
    # 🌟 调整顶部 label 为单行排列（ncol=3 或更大，此设置预留充裕空间）
    ax.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 1.25), 
              ncol=4, frameon=True, fancybox=True, shadow=False, 
              fontsize=7.5, columnspacing=0.5, handletextpad=0.2, handlelength=1.2, framealpha=1.0)
    
    plt.tight_layout()
    output_file = RESULTS / fname
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def plot_single_mem_4q(json_dir, save_path):
    # 🌟 调整 1：将画布拉宽（4.6 英寸），将高度压低（5.5 英寸），彻底告别瘦高身材
    fig, axes = plt.subplots(4, 3, figsize=(4.6, 5.5), sharey=False)
    
    quantiles = ['0.1', '0.5', '0.9', '0.95']
    distributions = ['uniform', 'normal', 'zipf']
    # 🌟 调整 2：分布标签放到最顶层作为列标题（Column Title）
    dist_titles = ['UNIFORM', 'NORMAL', 'ZIPF']
    x_values = [4, 8, 16, 32, 64, 128] 
    
    style_cfgs = {
        'PrivKLL-Q': {'color': "#3A86FF", 'linestyle': '--', 'marker': 'v'},
        'DPExpGK': {'color': "#06D6A0", 'linestyle': '-', 'marker': 'o'},
        'DCS DP': {'color': "#6C757D", 'linestyle': '-.', 'marker': '^'}
    }
    
    lines, labels = [], []

    # 🌟 调整 3：极限微调的盒模型坐标参数，将间距（w_space, h_space）大幅压缩，榨干所有空白
    box_width = 0.23    # 加宽子图盒子
    box_height = 0.165   # 保持高度
    w_space = 0.045      # 压缩横向间距
    h_space = 0.040      # 压缩纵向间距
    left_margin = 0.1  # 优化左边距
    bottom_margin = 0.09 # 优化底边距

    # 遍历 4 个分位数（行）
    for r_idx, q_str in enumerate(quantiles):
        # 自动扫描并提取对应分位数的最新 json 文件
        matching_files = [f for f in os.listdir(json_dir) if f.startswith(f"exp_s_mem_{q_str}_") and f.endswith(".json")]
        if not matching_files:
            print(f"⚠️ Missing data file for q={q_str}, skipped.")
            continue
            
        with open(os.path.join(json_dir, matching_files[0]), 'r') as f:
            file_payload = json.load(f)
            plot_data = file_payload['data']

        # 遍历 3 个分布（列）
        for c_idx, dist in enumerate(distributions):
            ax = axes[r_idx, c_idx]
            
            # 动态计算绝对盒模型位置
            box_left = left_margin + c_idx * (box_width + w_space)
            box_bottom = bottom_margin + (3 - r_idx) * (box_height + h_space)
            ax.set_position([box_left, box_bottom, box_width, box_height])
            
            dist_res = plot_data[dist]
            for m in ['PrivKLL-Q', 'DPExpGK', 'DCS DP']:
                mean = np.clip(np.array(dist_res[m]['mean']), 1e-7, None)
                p10 = np.clip(np.array(dist_res[m]['p10']), 1e-7, None)
                p90 = np.clip(np.array(dist_res[m]['p90']), 1e-7, None)
                
                line, = ax.plot(x_values, mean, color=style_cfgs[m]['color'], linestyle=style_cfgs[m]['linestyle'], 
                                 marker=style_cfgs[m]['marker'], linewidth=1.1, markersize=2.8, zorder=4)
                ax.fill_between(x_values, p10, p90, color=style_cfgs[m]['color'], alpha=0.05)
                
                if r_idx == 0 and c_idx == 0:
                    lines.append(line); labels.append(m)
            
            # X 轴属性调优
            ax.set_xscale('log', base=2)
            ax.set_xticks(x_values)
            
            # 只有最后一行显示 X 轴数字刻度
            if r_idx == 3:
                ax.set_xticklabels([f"$2^{{{int(np.log2(x))}}}$" for x in x_values], fontsize=6.5, rotation=30, ha='right')
                ax.tick_params(axis='x', pad=0.5)
            else:
                ax.set_xticklabels([])

            # Y 轴属性调优
            ax.set_yscale('log', base=2)
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: f"$2^{{{int(np.log2(y))}}}$" if y > 0 else "0"))
            ax.tick_params(axis='both', which='major', labelsize=6.5, pad=1)
            
            # 只有最左边那一列才显示 Y 轴刻度数字
            if c_idx != 0:
                ax.set_yticklabels([])

            ax.grid(True, which="both", ls="--", color='#e0e0e0', alpha=0.5, linewidth=0.4)
            
            # 🌟 调整 4：直接把 Distribution 信息写在第一行（r_idx=0）子图的正上方，保证不丢失且美观
            if r_idx == 0:
                ax.set_title(dist_titles[c_idx], fontsize=7.5, pad=4, weight='bold', color='#222222')
            
            # 🌟 调整 5：右侧行外挂标签，紧贴着最右侧子图，标明分位数 q
            if c_idx == 2:
                ax.text(1.05, 0.5, f"$q = {q_str}$", transform=ax.transAxes, ha='left', va='center', 
                        fontsize=7.5, weight='bold', color='#444444')

    # 🌟 调整 6：利用大画布全局坐标精确定位轴标签，完美避开空白和挤压
    # 大横轴标题
    fig.text(0.51, 0.015, 'Memory Budget (KB)', ha='center', va='center', fontsize=8.5)
           
    # 主 Y 轴标题（向内微调，紧贴刻度）
    fig.text(0.02, 0.5, 'Mean Abs Rank Error', ha='center', va='center', rotation='vertical', fontsize=8.5)
        
    # 🌟 调整 7：顶层平铺图例向下微调，刚好卡在顶部标题上面
    fig.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.51, 0.98), 
               ncol=3, frameon=True, fancybox=True, shadow=False, 
               fontsize=7.5, columnspacing=1.0, handletextpad=0.3, framealpha=1.0)
    
    # 坚决不使用 tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"🎉 Single-Mem全景图已成功保存至: {save_path}")

# ==================== 运行示例 ====================
if __name__ == "__main__":

    # 使用示例（假设你的 4 个 json 文件都放在当前目录 '.' 下）
    plot_single_mem_4q(json_dir='.', save_path='exp_s_mem_all4q.png')
