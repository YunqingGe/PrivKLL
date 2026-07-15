# calibration.py
import math
import pickle
import os
import numpy as np
from DPQE.src.kll import KLL
from DPQE.src.Greenwald_Khanna import GK
from DPQE.src.DCS import DCS

# ==========================================
# 1. 基础空间分析函数
# ==========================================
def kll_bytes_analytical(kll_instance):
    """计算 KLL 的理论物理内存字节数"""
    return sum(len(c) for c in kll_instance.compactors) * 8

def gk_bytes_analytical(gk_instance):
    """计算 GK 的理论物理内存字节数"""
    return len(gk_instance.S) * (8 + 4 + 4)

# ==========================================
# 2. 参数逆向推导与校准核心算法
# ==========================================
def find_kll_k_by_budget(target_bytes, H, c=2/3):
    max_items = target_bytes // 8
    low, high, best = 1, 100000, 128
    while low <= high:
        mid = (low + high) // 2
        cap = sum(int(np.ceil(c**(H-h-1) * mid)) + 1 for h in range(H))
        if cap <= max_items: 
            best = mid
            low = mid + 1
        else: 
            high = mid - 1
    return best

def find_gk_alpha_tight(anchor_bytes, dist, n, domain=(0, 100000)):
    probe_n = min(100000, n)
    low, high = domain
    if dist == 'uniform': 
        data = np.random.uniform(low, high, probe_n)
    elif dist == 'normal': 
        data = np.clip(np.random.normal((high+low)/2, (high-low)/6, probe_n), low, high)
    elif dist == 'zipf': 
        data = np.clip(np.random.pareto(a=2.0, size=probe_n) * (high/10), low, high)
    else:
        data = np.random.uniform(low, high, probe_n)
        
    low_a, high_a, best_alpha = 1e-5, 0.5, 0.01
    
    for _ in range(25):
        mid = (low_a + high_a) / 2
        gk = GK(alpha=mid)
        for v in data: 
            gk.insert(v)
        
        actual = gk_bytes_analytical(gk)
        if actual <= anchor_bytes: 
            high_a = mid
            best_alpha = mid
            print(f"  🔍 Tuning GK alpha down: {best_alpha:.6f} (bytes={actual})")
        else: 
            low_a = mid
            
    return best_alpha

def find_dcs_gamma_by_budget(target_bytes, universe):
    total_levels = math.ceil(math.log2(universe))
        
    # 理论极限最小空间：即使每一层只分配 1 行 1 列，也需要的最少字节数
    # (每格子 4 字节 * 1行 * 1列 * 16层)
    absolute_min_bytes = 1 * 1 * total_levels * 4 
    if target_bytes < absolute_min_bytes:
        return None  # 🔴 标记为内存不足
    
    low_g, high_g = 1e-5, 0.5
    best_gamma = high_g
    for _ in range(25):
        mid_g = (low_g + high_g) / 2
        columns = math.ceil((1.0 / mid_g) * math.sqrt(math.log(universe) * math.log(math.log(universe) / mid_g)))
        rows = math.ceil(math.log(math.log(universe) / mid_g))
        actual_bytes = rows * columns * total_levels * 4
        
        if actual_bytes <= target_bytes:
            best_gamma = mid_g
            high_g = mid_g  # 🌟 修正：内存没超，尝试寻找更小的 gamma（探索左半区）
            print(f"  🔍 Tuning DCS gamma up: {best_gamma:.6f} (rows={rows}, cols={columns}, bytes={actual_bytes})")
        else:
            low_g = mid_g   # 🌟 修正：内存超了，必须增大 gamma 缩小空间（探索右半区）
            print(f"  🔍 Tuning DCS gamma down: {mid_g:.6f} (rows={rows}, cols={columns}, bytes={actual_bytes})")
            
    # 下方的微调逻辑在二分查找正确后将恢复正常运作
    while True:
        next_gamma = best_gamma * 0.995
        if next_gamma < 1e-5:
            break
        columns = math.ceil((1.0 / next_gamma) * math.sqrt(math.log(universe) * math.log(math.log(universe) / next_gamma)))
        rows = math.ceil(math.log(math.log(universe) / next_gamma))
        if rows * columns * total_levels * 4 <= target_bytes:
            best_gamma = next_gamma
            print(f"  🔍 Fine-tuning DCS gamma down: {best_gamma:.6f} (bytes={rows * columns * total_levels * 4})")
        else:
            break
        
    return best_gamma

# ==========================================
# 3. 核心：字典文件生成、保存与读取
# ==========================================
def generate_and_save_calibration(cfg, filename="calib_matrix.pkl"):
    """
    运行校准算法，生成参数矩阵字典，并将其持久化保存为本地二进制文件。
    """
    print("=" * 80)
    print(f"🚀 [Calibration Center] Calculating and saving matrix to '{filename}'...")
    print("=" * 80)
    
    N = cfg['N']
    universe = 2**16
    safe_H = int(np.ceil(np.log(N) / np.log(1.5)))
    
    calib_dict = {}
    
    for dist in cfg['distributions']:
        calib_dict[dist] = {}
        print(f"\n▶ Calibrating Distribution: {dist.upper()}")
        print(f"  │   Mem (KB)   │   KLL k   │   GK Alpha   │   DCS Gamma   │")
        print(f"  ├──────────────┼───────────┼──────────────┼───────────────┤")
        
        for mem in cfg['memory_kbs']:
            calib_dict[dist][mem] = {}
            mem_bytes = int(mem * 1024)
            
            kll_k = find_kll_k_by_budget(mem_bytes, H=safe_H)
            
            gk_alpha = find_gk_alpha_tight(mem_bytes, dist, N)
            # dcs_gamma = find_dcs_gamma_by_budget(mem_bytes, universe)
            
            # 分发到各 epsilon 键下
            for eps in cfg['epsilons']:
                calib_dict[dist][mem][eps] = {
                    'kll_k': kll_k,
                    'gk_alpha': gk_alpha,
                    # 'dcs_gamma': dcs_gamma,
                }
                
            print(f"  │   {mem:<10d} │   {kll_k:<7d} │   {gk_alpha:.6f}   │")
            # print(f"  │   {mem:<10d} │   {kll_k:<7d} │   {gk_alpha:.6f}   │   {dcs_gamma:.6f}    │")

        print(f"  └──────────────┴───────────┴──────────────┴───────────────┘")
        
    # 🌟 将字典对象保存为本地文件
    with open(filename, 'wb') as f:
        pickle.dump(calib_dict, f)
        
    print(f"\n磁盘写入成功！参数字典文件已保存在: {os.path.abspath(filename)}")
    print("=" * 80 + "\n")
    return calib_dict


def load_calibration_matrix(filename="calib_matrix.pkl"):
    """
    从本地读取已保存的参数矩阵字典。文件不存在时返回 None。
    """
    if not os.path.exists(filename):
        print(f"⚠️ [Error] 找不到参数字典文件: '{filename}'，请先生成！")
        return None
        
    with open(filename, 'rb') as f:
        calib_dict = pickle.load(f)
    print(f"📂 [Loaded] 成功从本地文件读取参数矩阵: '{filename}'")
    return calib_dict


if __name__ == "__main__":
    # 本地独立运行时：生成并保存文件
    test_cfg = {
        'N': 10_000_000,
        'Ns': [100_000, 1_000_000, 10_000_000],
        'memory_kbs': [4, 8, 32, 64, 256, 1024],
        'epsilons': [1.0],
        'distributions': ['zipf']
    }
    # 生成并保存
    generate_and_save_calibration(test_cfg, "calib_matrix_eps1.pkl")