'''
Author: Jean jean_green@163.com
Date: 2026-06-17 02:45:25
LastEditors: Jean jean_green@163.com
LastEditTime: 2026-06-24 22:07:53
FilePath: /workspace/DPQE/src/scheduler.py
'''
# scheduler.py
import numpy as np
import logging
from DPQE.src.PrivKLL import PrivKLL
from DPQE.src.kll import KLL


class CheckpointScheduler:
    def __init__(self, n_min, N, kll_k=256, c=2.0/3.0, preset_cps=None):
        self.n_min = n_min
        self.N = N
        self.kll_k = kll_k
        self.c = c

        if preset_cps is not None:
            self.cps = sorted(list(set(preset_cps)))
        else:
            self.cps = []

    @classmethod
    def generate_kll_cps(cls, n_min, N, kll_k, c=2.0/3.0, data=None, domain=(0, 100000), verbose=False):
        """
        通过公共知识（Zipf, s=2）无隐私泄露地生成伪数据进行模拟探测，获取 KLL 的 Checkpoints
        """
        if data is None:
            # 1. 生成标准的 Zipf (s=2) 伪数据
            raw_zipf = np.random.zipf(a=2.0, size=N)
            low, high = domain
            simulated_data = np.clip(raw_zipf * (high / 10.0), low, high)
        else:
            simulated_data = data

        # 2. 启动探针模型推进流，捕捉树高（层数）跃迁
        probe_model = KLL(k=kll_k, c=c)  # 探针模型本身不消耗真实Epsilon，设为1.0即可
        kll_cps = []
        last_h = 0
        
        for i, val in enumerate(simulated_data):
            probe_model.update(val)
            n_curr = i + 1

            curr_h = len(probe_model.compactors)
            
            if n_curr >= n_min and curr_h > last_h:
                kll_cps.append(n_curr)
                last_h = curr_h
                
        if not kll_cps or kll_cps[-1] != N:
            kll_cps.append(N)

        kll_cps = sorted(list(set(kll_cps)))
        T_kll = len(kll_cps)

        return kll_cps
            
    def budgets(self, total_epsilon, strategy='uniform', custom_cps=None):
        cps = custom_cps if custom_cps is not None else self.cps
        T = len(cps)
        if T == 0:
            return []

        if strategy == 'uniform':
            return [total_epsilon / T] * T
        elif strategy == 'linear':
            weights = np.arange(1, T + 1)
            return (weights / weights.sum() * total_epsilon).tolist()
        elif strategy == 'exponential':
            weights = np.logspace(0, 2, num=T, base=2.0)
            return (weights / weights.sum() * total_epsilon).tolist()
        else:
            raise ValueError(f"Unknown budget strategy: {strategy}")