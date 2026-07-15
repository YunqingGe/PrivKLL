'''
Author: Jean jean_green@163.com
Date: 2026-04-09 18:06:51
LastEditors: Jean jean_green@163.com
LastEditTime: 2026-06-17 04:09:10
FilePath: /workspace/DPQE/privKLL.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import numpy as np 
from DPQE.src.kll import KLL

class PrivKLL:
    def __init__(self, epsilon, kll_k, c=2.0/3.0, domain_range=None):
            """
            确保参数名包含 kll_k
            """
            self.epsilon = epsilon
            self.kll_k = kll_k
            self.c = c
            self.domain_range = domain_range
            self.sketch = KLL(kll_k, c)

    @property
    def compactors(self):
        return self.sketch.compactors
    
    def ingest_data(self, val):
        self.sketch.update(val)

    def flatten_sketch(self):
        # 1. Flatten
        items_weights = []
        for h, compactor in enumerate(self.sketch.compactors):
            for item in compactor:
                items_weights.append((item, 2**h))
        
        if not items_weights:
            return None
            
        items_weights.sort()
        
        # 合并相同值的权重，确保 F(D) 的有序性
        f_d = []
        curr_v, curr_w = items_weights[0]
        for i in range(1, len(items_weights)):
            v, w = items_weights[i]
            if v == curr_v:
                curr_w += w
            else:
                f_d.append((curr_v, curr_w))
                curr_v, curr_w = v, w
        f_d.append((curr_v, curr_w))

        v_list = np.array([x[0] for x in f_d])
        w_list = np.array([x[1] for x in f_d])
        s = len(v_list)
        return v_list, w_list, s
        
    def release_quantile(self, q):
        v_list, w_list, s = self.flatten_sketch()
        
        # 2. 计算累积秩 R_j 和 目标秩 r*
        n_tilde = np.sum(w_list)
        r_star = q * n_tilde

        R = np.zeros(s)
        R[1:] = np.cumsum(w_list)[:-1] 

        # 3. 全局敏感度计算: Delta_u = 2^(H+1)
        # 这里的 H 为 KLL 的最大层级索引
        H_max_idx = len(self.sketch.compactors) - 1
        delta_u = 2**(H_max_idx + 1)

        # 4. 指数机制采样
        utilities = -np.abs(R - r_star)
        num_intervals = s - 1
        if num_intervals < 1:
            return v_list[0]

        # Gumbel-Max 实现采样
        gumbel_noise = np.random.gumbel(0, 1, size=num_intervals)
        v_list_intervals = v_list[1:] - v_list[:-1]  # 每个区间的宽度
        width_scores = np.log(np.maximum(v_list_intervals, 1e-10))
        scores = width_scores + (self.epsilon * utilities[:num_intervals]) / (2 * delta_u) + gumbel_noise
       
        j_star = np.argmax(scores)

        # 5. 区间内均匀采样
        v_start = v_list[j_star]
        v_end = v_list[j_star + 1]
        v_hat = np.random.uniform(v_start, v_end)
        
        return v_hat
    
    def release_multi_quantiles(self, quantiles):
        v_list, w_list, s = self.flatten_sketch()

        if self.domain_range is not None:
            a = self.domain_range[0]  # 使用已知的最小值作为左边界起点
            b = self.domain_range[1]  # 使用已知最大值
            # 确保当前的 v_list 不会越界
            v_list = np.clip(v_list, a, b)
        else:
            a = 0 # 缺省默认
            b = v_list[-1] if len(v_list) > 0 else 100
        # 拼入左边界
        v_arr = np.concatenate([[a], v_list])   # shape: (s+1,)
        
        # 如果右边界大于 v_list 的最大值，也可以考虑追加 b 作为最后一个区间的终点
        if self.domain_range is not None and v_arr[-1] < b:
            v_arr = np.concatenate([v_arr, [b]])
            w_list = np.concatenate([w_list, [0]])
            s += 1

        # a = 0
        # v_arr = np.concatenate([[a], v_list])   # shape: (s+1,)

        n_intervals = s                         # 区间总数
        w_arr = w_list.copy()                           # 区间 0..s-1 的 rank 宽度
       
        n_tilde = np.sum(w_list)

        # 累积 rank：R[i] = sum(w_arr[0..i-1])，R[0]=0
        R = np.zeros(n_intervals + 1)               # R[0..n_intervals]
        R[1:] = np.cumsum(w_arr)                    # R[i] = sum of w_arr[0..i-1]

        # 敏感度
        H_max_idx = len(self.sketch.compactors) - 1
        delta_u = 2 ** (H_max_idx + 1)

        qs = np.array(quantiles)                    # shape: (m,)
        m = len(qs)

        # ------------------------------------------------------------------ #
        # Phase 2: 预计算权重 phi[i, j]
        # phi(i, j) = exp(-eps/(2*Delta) * |R[i] - q_j * n_tilde|)
        # i in 0..n_intervals-1 (区间下标), j in 0..m-1 (分位数下标)
        # R[i] 是区间 i 的左端 rank
        # ------------------------------------------------------------------ #
        coeff = self.epsilon / (2.0 * delta_u)

        # R_left[i] = R[i]，即区间 i 左端的累积 rank
        R_left = R[:n_intervals]                    # shape: (n_intervals,)
        target_ranks = qs * n_tilde                 # shape: (m,)

        phi = np.exp(
            -coeff * np.abs(R_left[:, None] - target_ranks[None, :])
        )                                           # shape: (n_intervals, m)
 
        phi_tail = np.ones(n_intervals)

        # ------------------------------------------------------------------ #
        # Phase 3: 前向 DP
        # alpha[j, i, k]: 前 j+1 个分位数，最后连续 k+1 个都落在区间 i 的总权重
        # 使用 0-indexed：j in 0..m-1, i in 0..n_intervals-1, k in 0..j
        # ------------------------------------------------------------------ #
        # 用字典存储稀疏的 alpha，避免 m*n*m 的密集张量过大
        # alpha[(j, i, k)] = 权重值
        # 实际上 k <= j，且通常 k 很小，用二维数组 alpha[i, k] 滚动更新

        # 为数值稳定，在 log 域计算
        NEG_INF = -np.inf

        # log_alpha[i, k] 表示当前 j 层的 log(alpha(j, i, k))
        # k 维度最大为 j，用列表动态扩展

        # 初始化 j=0（对应第1个分位数）
        log_alpha_prev = {}                         # {(i, k): log_value}
        for i in range(n_intervals):
            log_val = np.log(max(w_arr[i], 1e-300)) + np.log(max(phi[i, 0], 1e-300))
            log_alpha_prev[(i, 0)] = log_val        # k=0 对应连续1个（0-indexed）

        def log_sum_exp(vals):
            """数值稳定的 log-sum-exp"""
            if not vals:
                return NEG_INF
            arr = np.array(vals)
            mx = np.max(arr)
            if mx == NEG_INF:
                return NEG_INF
            return mx + np.log(np.sum(np.exp(arr - mx)))

        # 存储所有层的 log_alpha，用于后向采样
        all_log_alpha = [log_alpha_prev]            # all_log_alpha[j] = {(i,k): log_val}

        for j in range(1, m):
            log_alpha_curr = {}

            # 计算 log_alpha_hat[i] = log(sum_k alpha(j-1, i, k))
            log_alpha_hat = np.full(n_intervals, NEG_INF)
            for (i, k), lv in log_alpha_prev.items():
                if lv > log_alpha_hat[i]:
                    # 先收集再做 log_sum_exp
                    pass
            # 重新按 i 分组
            from collections import defaultdict
            groups = defaultdict(list)
            for (i, k), lv in log_alpha_prev.items():
                groups[i].append(lv)
            for i, lvs in groups.items():
                log_alpha_hat[i] = log_sum_exp(lvs)

            # 前缀和 S[i] = log(sum_{i'=0}^{i} alpha_hat(j-1, i'))
            # 用 log-sum-exp 累积
            log_S = np.full(n_intervals + 1, NEG_INF)  # log_S[i] = log(S(j-1, i-1))
            # log_S[0] = log(0) = -inf（S(-1)=0 对应空前缀）
            for i in range(n_intervals):
                log_S[i + 1] = log_sum_exp([log_S[i], log_alpha_hat[i]])

            log_w = np.log(np.maximum(w_arr, 1e-300))
            log_phi_j = np.log(np.maximum(phi[:, j], 1e-300))

            # Case 1: k=0 (连续1个新区间 i，来自不同前驱)
            for i in range(n_intervals):
                # alpha(j, i, 1) = w_i * phi(i,j) * S(j-1, i-1)
                # log: log_w[i] + log_phi_j[i] + log_S[i]
                lv = log_w[i] + log_phi_j[i] + log_S[i]
                if lv > NEG_INF:
                    log_alpha_curr[(i, 0)] = lv

            # Case 2: k>=1 (延续区间 i，连续 k+1 个)
            for (i, k_prev), lv_prev in log_alpha_prev.items():
                k = k_prev + 1                      # 新的连续次数（0-indexed）
                # alpha(j, i, k+1) = w_i * phi(i,j) * alpha(j-1, i, k) / (k+1)
                lv = log_w[i] + log_phi_j[i] + lv_prev - np.log(k + 1)
                key = (i, k)
                if key not in log_alpha_curr or lv > log_alpha_curr[key]:
                    log_alpha_curr[key] = lv

            log_alpha_prev = log_alpha_curr
            all_log_alpha.append(log_alpha_curr)

        # ------------------------------------------------------------------ #
        # Phase 4: 后向采样
        # ------------------------------------------------------------------ #
        def sample_from_log_weights(keys, log_weights):
            """从 log 权重中采样一个 key"""
            log_weights = np.array(log_weights)
            mx = np.max(log_weights)
            weights = np.exp(log_weights - mx)
            weights /= weights.sum()
            idx = np.random.choice(len(keys), p=weights)
            return keys[idx]

        i_seq = np.zeros(m, dtype=int)             # 最终区间序列

        # 采样最后一个分位数的区间
        log_phi_tail = np.zeros(n_intervals)     # log(1) = 0
        last_alpha = all_log_alpha[m - 1]

        keys = list(last_alpha.keys())
        log_ws = [last_alpha[(i, k)] for (i, k) in keys]

        (i_star, k_star) = sample_from_log_weights(keys, log_ws)
        # k_star 是 0-indexed 连续次数，实际连续 k_star+1 个
        run_len = k_star + 1
        for jj in range(m - run_len, m):
            i_seq[jj] = i_star
        j_ptr = m - run_len - 1                    # 下一个待确定的分位数下标

        # 反向追踪
        while j_ptr >= 0:
            i_next = i_seq[j_ptr + 1]
            log_phi_next = np.log(max(phi[i_next, j_ptr + 1], 1e-300)) \
                        if j_ptr + 1 < m else log_phi_tail[i_next]

            cur_alpha = all_log_alpha[j_ptr]
            # 只考虑 i <= i_next 的区间（有序约束）
            keys = [(i, k) for (i, k) in cur_alpha if i <= i_next]
            if not keys:
                # 退化情形：强制选 i_next
                i_seq[j_ptr] = i_next
                j_ptr -= 1
                continue

            log_ws = [
                cur_alpha[(i, k)] + np.log(max(phi[i_next, j_ptr + 1], 1e-300))
                for (i, k) in keys
            ]
            (i_star, k_star) = sample_from_log_weights(keys, log_ws)
            run_len = k_star + 1
            for jj in range(j_ptr - run_len + 1, j_ptr + 1):
                if jj >= 0:
                    i_seq[jj] = i_star
            j_ptr = j_ptr - run_len

        # ------------------------------------------------------------------ #
        # Phase 5: 连续化输出
        # ------------------------------------------------------------------ #
        # print(f"i_seq = {i_seq}")
        # print(f"n_intervals = {n_intervals}, s = {s}")
        # print(f"v_arr shape = {v_arr.shape}, v_arr[-3:] = {v_arr[-3:]}")
        results = []
        max_valid_idx = len(v_arr) - 1 # 🌟 获取 v_arr 的最大安全索引边界
        for j in range(m):
            i = i_seq[j]

            # 🌟 防御性过滤：若索引溢出则安全向左平移，确保多进程在 10^9 极限流下不会因广播错位崩溃
            if i >= max_valid_idx: 
                i = max_valid_idx - 1

            lo = v_arr[i]                           # 区间左端点
            hi = v_arr[i + 1]                       # 区间右端点
            
            results.append(np.random.uniform(lo, hi))

        return results
        