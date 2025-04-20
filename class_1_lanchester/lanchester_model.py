# 兰彻斯特方程模型实现
# 用于SLG游戏战斗模拟与平衡性分析

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pandas as pd
import seaborn as sns

class LanchesterModel:
    """
    兰彻斯特方程模型基类
    实现基础的兰彻斯特方程（线性律和平方律）以及适合SLG游戏的扩展
    """
    
    def __init__(self, model_type='square'):
        """
        初始化兰彻斯特模型
        
        参数:
        model_type: 模型类型，可选 'linear'（线性律）或 'square'（平方律）
        """
        self.model_type = model_type
        
    def linear_law(self, t, y, params):
        """
        兰彻斯特线性律方程组
        适用于古代冷兵器或远程单位对单位的情况
        
        dA/dt = -b * B
        dB/dt = -a * A
        
        参数:
        t: 时间点
        y: 当前状态 [A, B]
        params: 参数字典，包含 'a' 和 'b'
        """
        A, B = y
        a, b = params['a'], params['b']
        
        # 确保部队数量不会变为负数
        if A <= 0:
            dA = 0
        else:
            dA = -b * B
            
        if B <= 0:
            dB = 0
        else:
            dB = -a * A
            
        return [dA, dB]
    
    def square_law(self, t, y, params):
        """
        兰彻斯特平方律方程组
        适用于现代战争或集中火力的情况
        
        dA/dt = -b * B
        dB/dt = -a * A
        
        参数:
        t: 时间点
        y: 当前状态 [A, B]
        params: 参数字典，包含 'a' 和 'b'
        """
        A, B = y
        a, b = params['a'], params['b']
        
        # 确保部队数量不会变为负数
        if A <= 0:
            dA = 0
        else:
            dA = -b * B
            
        if B <= 0:
            dB = 0
        else:
            dB = -a * A
            
        return [dA, dB]
    
    def simulate(self, A0, B0, params, t_max=100, t_points=1000):
        """
        模拟战斗过程
        
        参数:
        A0: A方初始兵力
        B0: B方初始兵力
        params: 模型参数
        t_max: 最大模拟时间
        t_points: 时间点数量
        
        返回:
        t: 时间点数组
        y: 状态数组，每行是一个时间点的 [A, B]
        """
        y0 = [A0, B0]
        t_span = (0, t_max)
        t_eval = np.linspace(0, t_max, t_points)
        
        if self.model_type == 'linear':
            sol = solve_ivp(lambda t, y: self.linear_law(t, y, params), 
                           t_span, y0, t_eval=t_eval, method='RK45')
        else:  # 默认使用平方律
            sol = solve_ivp(lambda t, y: self.square_law(t, y, params), 
                           t_span, y0, t_eval=t_eval, method='RK45')
        
        return sol.t, sol.y.T
    
    def plot_simulation(self, t, y, title=None):
        """
        绘制模拟结果
        
        参数:
        t: 时间点数组
        y: 状态数组
        title: 图表标题
        """
        plt.figure(figsize=(10, 6))
        plt.plot(t, y[:, 0], 'b-', label='A方兵力')
        plt.plot(t, y[:, 1], 'r-', label='B方兵力')
        plt.xlabel('时间')
        plt.ylabel('兵力')
        plt.grid(True)
        plt.legend()
        
        if title:
            plt.title(title)
        else:
            plt.title(f'兰彻斯特{"线性律" if self.model_type == "linear" else "平方律"}模拟结果')
        
        plt.show()


class SLGLanchesterModel(LanchesterModel):
    """
    扩展的兰彻斯特模型，适用于SLG游戏
    包含兵种克制、地形加成、士气等因素
    """
    
    def __init__(self, model_type='square'):
        super().__init__(model_type)
    
    def extended_law(self, t, y, params):
        """
        扩展的兰彻斯特方程，考虑SLG特有因素
        
        参数:
        t: 时间点
        y: 当前状态 [A1, A2, ..., An, B1, B2, ..., Bm]
            其中A1...An是A方不同兵种的兵力
            B1...Bm是B方不同兵种的兵力
        params: 参数字典，包含：
            'effectiveness_matrix': 效能矩阵，表示兵种间的克制关系
            'terrain_factor_A': A方地形加成
            'terrain_factor_B': B方地形加成
            'morale_A': A方士气
            'morale_B': B方士气
        """
        # 解析参数
        n_units_A = params['n_units_A']
        n_units_B = params['n_units_B']
        effectiveness_matrix = params['effectiveness_matrix']
        terrain_factor_A = params.get('terrain_factor_A', 1.0)
        terrain_factor_B = params.get('terrain_factor_B', 1.0)
        morale_A = params.get('morale_A', 1.0)
        morale_B = params.get('morale_B', 1.0)
        
        # 分离A方和B方兵力
        A_forces = y[:n_units_A]
        B_forces = y[n_units_A:n_units_A+n_units_B]
        
        # 初始化变化率
        dA = np.zeros(n_units_A)
        dB = np.zeros(n_units_B)
        
        # 计算A方各兵种的损失率
        for i in range(n_units_A):
            if A_forces[i] <= 0:
                dA[i] = 0
                continue
                
            loss_rate = 0
            for j in range(n_units_B):
                if B_forces[j] <= 0:
                    continue
                # B方j兵种对A方i兵种的效能
                effectiveness = effectiveness_matrix[n_units_A+j, i]
                loss_rate += effectiveness * B_forces[j] * terrain_factor_B * morale_B
            
            dA[i] = -loss_rate
        
        # 计算B方各兵种的损失率
        for j in range(n_units_B):
            if B_forces[j] <= 0:
                dB[j] = 0
                continue
                
            loss_rate = 0
            for i in range(n_units_A):
                if A_forces[i] <= 0:
                    continue
                # A方i兵种对B方j兵种的效能
                effectiveness = effectiveness_matrix[i, n_units_A+j]
                loss_rate += effectiveness * A_forces[i] * terrain_factor_A * morale_A
            
            dB[j] = -loss_rate
        
        return np.concatenate([dA, dB])
    
    def simulate_extended(self, A_forces, B_forces, params, t_max=100, t_points=1000):
        """
        模拟扩展模型的战斗过程
        
        参数:
        A_forces: A方各兵种初始兵力数组
        B_forces: B方各兵种初始兵力数组
        params: 模型参数
        t_max: 最大模拟时间
        t_points: 时间点数量
        
        返回:
        t: 时间点数组
        y: 状态数组
        """
        # 设置初始状态和参数
        y0 = np.concatenate([A_forces, B_forces])
        params['n_units_A'] = len(A_forces)
        params['n_units_B'] = len(B_forces)
        
        t_span = (0, t_max)
        t_eval = np.linspace(0, t_max, t_points)
        
        # 求解微分方程
        sol = solve_ivp(lambda t, y: self.extended_law(t, y, params), 
                       t_span, y0, t_eval=t_eval, method='RK45')
        
        return sol.t, sol.y.T
    
    def plot_extended_simulation(self, t, y, unit_names=None, title=None):
        """
        绘制扩展模型的模拟结果
        
        参数:
        t: 时间点数组
        y: 状态数组
        unit_names: 兵种名称列表
        title: 图表标题
        """
        n_units = y.shape[1]
        n_units_A = n_units // 2  # 假设A和B有相同数量的兵种
        
        if unit_names is None:
            unit_names = [f'A{i+1}' for i in range(n_units_A)] + [f'B{i+1}' for i in range(n_units - n_units_A)]
        
        plt.figure(figsize=(12, 8))
        
        # 绘制A方兵力变化
        for i in range(n_units_A):
            plt.plot(t, y[:, i], '-', label=unit_names[i])
        
        # 绘制B方兵力变化
        for i in range(n_units_A, n_units):
            plt.plot(t, y[:, i], '--', label=unit_names[i])
        
        plt.xlabel('时间')
        plt.ylabel('兵力')
        plt.grid(True)
        plt.legend()
        
        if title:
            plt.title(title)
        else:
            plt.title('SLG战斗模拟结果')
        
        plt.show()


class SensitivityAnalysis:
    """
    敏感性分析工具，用于评估参数变化对战斗结果的影响
    """
    
    def __init__(self, model):
        """
        初始化敏感性分析工具
        
        参数:
        model: 兰彻斯特模型实例
        """
        self.model = model
    
    def parameter_sweep(self, param_name, param_values, A0, B0, fixed_params, t_max=100):
        """
        参数扫描，分析单一参数变化对结果的影响
        
        参数:
        param_name: 要扫描的参数名称
        param_values: 参数值数组
        A0: A方初始兵力
        B0: B方初始兵力
        fixed_params: 其他固定参数
        t_max: 最大模拟时间
        
        返回:
        results: 结果字典，包含每个参数值对应的模拟结果
        """
        results = {}
        
        for value in param_values:
            # 创建参数副本并更新要扫描的参数
            params = fixed_params.copy()
            params[param_name] = value
            
            # 运行模拟
            t, y = self.model.simulate(A0, B0, params, t_max=t_max)
            
            # 存储结果
            results[value] = {
                't': t,
                'y': y,
                'winner': 'A' if y[-1, 0] > 0 and y[-1, 1] <= 0 else 
                         'B' if y[-1, 1] > 0 and y[-1, 0] <= 0 else 'Draw',
                'remaining_A': max(0, y[-1, 0]),
                'remaining_B': max(0, y[-1, 1]),
                'battle_duration': t[np.argmax((y[:, 0] <= 0) | (y[:, 1] <= 0))] if np.any((y[:, 0] <= 0) | (y[:, 1] <= 0)) else t[-1]
            }
        
        return results
    
    def plot_parameter_sweep_results(self, results, param_name, metric='winner'):
        """
        绘制参数扫描结果
        
        参数:
        results: parameter_sweep返回的结果字典
        param_name: 扫描的参数名称
        metric: 要绘制的指标，可选 'winner', 'remaining_A', 'remaining_B', 'battle_duration'
        """
        param_values = list(results.keys())
        
        if metric == 'winner':
            winners = [results[v]['winner'] for v in param_values]
            plt.figure(figsize=(10, 6))
            
            # 为不同的获胜方设置不同的颜色
            colors = {'A': 'blue', 'B': 'red', 'Draw': 'gray'}
            color_values = [colors[w] for w in winners]
            
            plt.scatter(param_values, [1] * len(param_values), c=color_values, s=100)
            plt.yticks([])
            plt.xlabel(param_name)
            plt.title(f'{param_name}对战斗结果的影响')
            
            # 添加图例
            from matplotlib.lines import Line2D
            legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=10, label=l)
                              for l, c in colors.items()]
            plt.legend(handles=legend_elements)
            
        else:
            metric_values = [results[v][metric] for v in param_values]
            plt.figure(figsize=(10, 6))
            plt.plot(param_values, metric_values, 'o-')
            plt.xlabel(param_name)
            plt.ylabel(metric)
            plt.title(f'{param_name}对{metric}的影响')
            plt.grid(True)
        
        plt.show()
    
    def heatmap_analysis(self, param1_name, param1_values, param2_name, param2_values, 
                        A0, B0, fixed_params, metric='winner', t_max=100):
        """
        热图分析，分析两个参数同时变化对结果的影响
        
        参数:
        param1_name, param2_name: 要分析的两个参数名称
        param1_values, param2_values: 两个参数的值数组
        A0, B0: 初始兵力
        fixed_params: 其他固定参数
        metric: 要分析的指标
        t_max: 最大模拟时间
        """
        # 创建结果矩阵
        n1, n2 = len(param1_values), len(param2_values)
        result_matrix = np.zeros((n1, n2))
        
        for i, v1 in enumerate(param1_values):
            for j, v2 in enumerate(param2_values):
                # 创建参数副本并更新要分析的参数
                params = fixed_params.copy()
                params[param1_name] = v1
                params[param2_name] = v2
                
                # 运行模拟
                t, y = self.model.simulate(A0, B0, params, t_max=t_max)
                
                # 根据指标存储结果
                if metric == 'winner':
                    # 1表示A获胜，-1表示B获胜，0表示平局
                    if y[-1, 0] > 0 and y[-1, 1] <= 0:
                        result_matrix[i, j] = 1  # A获胜
                    elif y[-1, 1] > 0 and y[-1, 0] <= 0:
                        result_matrix[i, j] = -1  # B获胜
                    else:
                        result_matrix[i, j] = 0  # 平局
                elif metric == 'remaining_A':
                    result_matrix[i, j] = max(0, y[-1, 0])
                elif metric == 'remaining_B':
                    result_matrix[i, j] = max(0, y[-1, 1])
                elif metric == 'battle_duration':
                    end_idx = np.argmax((y[:, 0] <= 0) | (y[:, 1] <= 0))
                    result_matrix[i, j] = t[end_idx] if end_idx > 0 else t[-1]
        
        # 绘制热图
        plt.figure(figsize=(10, 8))
        
        if metric == 'winner':
            # 自定义颜色映射
            cmap = plt.cm.RdBu
            sns.heatmap(result_matrix, cmap=cmap, center=0,
                      xticklabels=param2_values, yticklabels=param1_values)
            plt.title(f'{param1_name}和{param2_name}对战斗结果的影响')
            plt.xlabel(param2_name)
            plt.ylabel(param1_name)
            
            # 添加自定义图例
            from matplotlib.colors import LinearSegmentedColormap
            colors = [(0, 0, 1), (1, 1, 1), (1, 0, 0)]  # 蓝色-白色-红色
            cmap_custom = LinearSegmentedColormap.from_list('custom', colors, N=3)
            sm = plt.cm.ScalarMappable(cmap=cmap_custom, norm=plt.Normalize(vmin=-1, vmax=1))
            sm.set_array([])
            cbar = plt.colorbar(sm)
            cbar.set_ticks([-1, 0, 1])
            cbar.set_ticklabels(['B获胜', '平局', 'A获胜'])
        else:
            sns.heatmap(result_matrix, cmap='viridis',
                      xticklabels=param2_values, yticklabels=param1_values)
            plt.title(f'{param1_name}和{param2_name}对{metric}的影响')
            plt.xlabel(param2_name)
            plt.ylabel(param1_name)
            plt.colorbar(label=metric)
        
        plt.tight_layout()
        plt.show()