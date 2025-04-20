# 高级SLG战斗模型
# 扩展兰彻斯特方程，加入更多SLG游戏特有的战斗机制

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pandas as pd
import seaborn as sns
from lanchester_model import SLGLanchesterModel, SensitivityAnalysis

class AdvancedSLGModel(SLGLanchesterModel):
    """
    高级SLG战斗模型
    扩展基础兰彻斯特模型，加入更多SLG游戏特有的战斗机制：
    - 集火策略
    - 远程/近战差异
    - 技能和特殊能力
    - 疲劳和补给
    - 随机因素
    """
    
    def __init__(self, model_type='square'):
        super().__init__(model_type)
    
    def advanced_combat_law(self, t, y, params):
        """
        高级战斗方程，考虑更多SLG特有因素
        
        参数:
        t: 时间点
        y: 当前状态 [A1, A2, ..., An, B1, B2, ..., Bm, morale_A, morale_B, fatigue_A, fatigue_B]
            其中A1...An是A方不同兵种的兵力
            B1...Bm是B方不同兵种的兵力
            morale_A, morale_B是双方士气
            fatigue_A, fatigue_B是双方疲劳度
        params: 参数字典
        """
        # 解析参数
        n_units_A = params['n_units_A']
        n_units_B = params['n_units_B']
        effectiveness_matrix = params['effectiveness_matrix']
        terrain_factor_A = params.get('terrain_factor_A', 1.0)
        terrain_factor_B = params.get('terrain_factor_B', 1.0)
        
        # 从状态向量中提取各部分
        A_forces = y[:n_units_A]
        B_forces = y[n_units_A:n_units_A+n_units_B]
        morale_A = y[n_units_A+n_units_B]
        morale_B = y[n_units_A+n_units_B+1]
        fatigue_A = y[n_units_A+n_units_B+2]
        fatigue_B = y[n_units_A+n_units_B+3]
        
        # 初始化变化率
        dA = np.zeros(n_units_A)
        dB = np.zeros(n_units_B)
        dMorale_A = 0
        dMorale_B = 0
        dFatigue_A = 0
        dFatigue_B = 0
        
        # 应用疲劳效果（降低战斗效能）
        fatigue_effect_A = max(0.5, 1.0 - fatigue_A * 0.5)  # 疲劳最多降低50%效能
        fatigue_effect_B = max(0.5, 1.0 - fatigue_B * 0.5)
        
        # 计算有效兵力（考虑士气）
        effective_A_forces = A_forces * morale_A
        effective_B_forces = B_forces * morale_B
        
        # 获取远程单位标记
        ranged_units_A = params.get('ranged_units_A', np.zeros(n_units_A, dtype=bool))
        ranged_units_B = params.get('ranged_units_B', np.zeros(n_units_B, dtype=bool))
        
        # 获取集火策略
        focus_fire_A = params.get('focus_fire_A', False)
        focus_fire_B = params.get('focus_fire_B', False)
        focus_target_A = params.get('focus_target_A', 0)  # B方的目标单位索引
        focus_target_B = params.get('focus_target_B', 0)  # A方的目标单位索引
        
        # 计算A方各兵种的损失率
        for i in range(n_units_A):
            if A_forces[i] <= 0:
                dA[i] = 0
                continue
                
            loss_rate = 0
            
            # 如果B方使用集火策略且目标是当前单位
            if focus_fire_B and focus_target_B == i and np.sum(B_forces > 0) > 0:
                # 所有B方单位集中攻击A方的这个单位
                for j in range(n_units_B):
                    if B_forces[j] <= 0:
                        continue
                    
                    # 远程单位在前期有优势
                    range_factor = 1.2 if ranged_units_B[j] and t < params.get('ranged_advantage_time', 30) else 1.0
                    
                    # B方j兵种对A方i兵种的效能（考虑疲劳）
                    effectiveness = effectiveness_matrix[n_units_A+j, i] * fatigue_effect_B * range_factor
                    
                    # 集火时提高效能
                    effectiveness *= 1.3
                    
                    loss_rate += effectiveness * effective_B_forces[j] * terrain_factor_B
            else:
                # 常规攻击
                for j in range(n_units_B):
                    if B_forces[j] <= 0:
                        continue
                    
                    # 远程单位在前期有优势
                    range_factor = 1.2 if ranged_units_B[j] and t < params.get('ranged_advantage_time', 30) else 1.0
                    
                    # B方j兵种对A方i兵种的效能（考虑疲劳）
                    effectiveness = effectiveness_matrix[n_units_A+j, i] * fatigue_effect_B * range_factor
                    
                    loss_rate += effectiveness * effective_B_forces[j] * terrain_factor_B
            
            dA[i] = -loss_rate
        
        # 计算B方各兵种的损失率
        for j in range(n_units_B):
            if B_forces[j] <= 0:
                dB[j] = 0
                continue
                
            loss_rate = 0
            
            # 如果A方使用集火策略且目标是当前单位
            if focus_fire_A and focus_target_A == j and np.sum(A_forces > 0) > 0:
                # 所有A方单位集中攻击B方的这个单位
                for i in range(n_units_A):
                    if A_forces[i] <= 0:
                        continue
                    
                    # 远程单位在前期有优势
                    range_factor = 1.2 if ranged_units_A[i] and t < params.get('ranged_advantage_time', 30) else 1.0
                    
                    # A方i兵种对B方j兵种的效能（考虑疲劳）
                    effectiveness = effectiveness_matrix[i, n_units_A+j] * fatigue_effect_A * range_factor
                    
                    # 集火时提高效能
                    effectiveness *= 1.3
                    
                    loss_rate += effectiveness * effective_A_forces[i] * terrain_factor_A
            else:
                # 常规攻击
                for i in range(n_units_A):
                    if A_forces[i] <= 0:
                        continue
                    
                    # 远程单位在前期有优势
                    range_factor = 1.2 if ranged_units_A[i] and t < params.get('ranged_advantage_time', 30) else 1.0
                    
                    # A方i兵种对B方j兵种的效能（考虑疲劳）
                    effectiveness = effectiveness_matrix[i, n_units_A+j] * fatigue_effect_A * range_factor
                    
                    loss_rate += effectiveness * effective_A_forces[i] * terrain_factor_A
            
            dB[j] = -loss_rate
        
        # 计算士气变化
        # 士气受损失率影响：损失越大，士气下降越快
        A_total_initial = params.get('A_total_initial', np.sum(A_forces))
        B_total_initial = params.get('B_total_initial', np.sum(B_forces))
        
        A_current_total = np.sum(A_forces)
        B_current_total = np.sum(B_forces)
        
        A_loss_ratio = 1 - A_current_total / A_total_initial if A_total_initial > 0 else 0
        B_loss_ratio = 1 - B_current_total / B_total_initial if B_total_initial > 0 else 0
        
        # 士气变化率与损失率成正比
        morale_decay_rate = 0.01  # 基础士气衰减率
        dMorale_A = -morale_decay_rate * A_loss_ratio * morale_A
        dMorale_B = -morale_decay_rate * B_loss_ratio * morale_B
        
        # 如果一方损失过大，士气会急剧下降（溃败效应）
        rout_threshold = 0.7  # 溃败阈值
        if A_loss_ratio > rout_threshold:
            dMorale_A -= 0.05 * morale_A
        if B_loss_ratio > rout_threshold:
            dMorale_B -= 0.05 * morale_B
        
        # 计算疲劳变化
        # 疲劳随时间增加，但增长率会随时间减缓
        fatigue_increase_rate = 0.005  # 基础疲劳增长率
        dFatigue_A = fatigue_increase_rate * (1 - fatigue_A)  # 疲劳增长率随疲劳程度增加而减小
        dFatigue_B = fatigue_increase_rate * (1 - fatigue_B)
        
        # 应用补给效果（如果有）
        supply_rate_A = params.get('supply_rate_A', 0)
        supply_rate_B = params.get('supply_rate_B', 0)
        
        if supply_rate_A > 0:
            dFatigue_A -= supply_rate_A * fatigue_A  # 补给减少疲劳
        if supply_rate_B > 0:
            dFatigue_B -= supply_rate_B * fatigue_B
        
        # 合并所有变化率
        return np.concatenate([dA, dB, [dMorale_A, dMorale_B, dFatigue_A, dFatigue_B]])
    
    def simulate_advanced(self, A_forces, B_forces, params, t_max=100, t_points=1000):
        """
        模拟高级战斗模型
        
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
        # 设置初始状态
        initial_morale_A = params.get('initial_morale_A', 1.0)
        initial_morale_B = params.get('initial_morale_B', 1.0)
        initial_fatigue_A = params.get('initial_fatigue_A', 0.0)
        initial_fatigue_B = params.get('initial_fatigue_B', 0.0)
        
        y0 = np.concatenate([A_forces, B_forces, 
                            [initial_morale_A, initial_morale_B, 
                             initial_fatigue_A, initial_fatigue_B]])
        
        # 设置参数
        params['n_units_A'] = len(A_forces)
        params['n_units_B'] = len(B_forces)
        params['A_total_initial'] = np.sum(A_forces)
        params['B_total_initial'] = np.sum(B_forces)
        
        t_span = (0, t_max)
        t_eval = np.linspace(0, t_max, t_points)
        
        # 求解微分方程
        sol = solve_ivp(lambda t, y: self.advanced_combat_law(t, y, params), 
                       t_span, y0, t_eval=t_eval, method='RK45')
        
        return sol.t, sol.y.T
    
    def plot_advanced_simulation(self, t, y, params, title=None):
        """
        绘制高级模型的模拟结果
        
        参数:
        t: 时间点数组
        y: 状态数组
        params: 模型参数
        title: 图表标题
        """
        n_units_A = params['n_units_A']
        n_units_B = params['n_units_B']
        unit_names = params.get('unit_names', None)
        
        if unit_names is None:
            unit_names = [f'A{i+1}' for i in range(n_units_A)] + [f'B{i+1}' for i in range(n_units_B)]
        
        # 创建一个2x2的子图布局
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 兵力变化图
        ax1 = axes[0, 0]
        
        # 绘制A方兵力变化
        for i in range(n_units_A):
            ax1.plot(t, y[:, i], '-', label=unit_names[i])
        
        # 绘制B方兵力变化
        for i in range(n_units_A, n_units_A + n_units_B):
            ax1.plot(t, y[:, i], '--', label=unit_names[i])
        
        ax1.set_xlabel('时间')
        ax1.set_ylabel('兵力')
        ax1.set_title('各兵种兵力变化')
        ax1.grid(True)
        ax1.legend()
        
        # 2. 总兵力对比图
        ax2 = axes[0, 1]
        
        # 计算双方总兵力
        A_total = np.sum(y[:, :n_units_A], axis=1)
        B_total = np.sum(y[:, n_units_A:n_units_A+n_units_B], axis=1)
        
        ax2.plot(t, A_total, 'b-', linewidth=2, label='A方总兵力')
        ax2.plot(t, B_total, 'r-', linewidth=2, label='B方总兵力')
        ax2.set_xlabel('时间')
        ax2.set_ylabel('总兵力')
        ax2.set_title('双方总兵力对比')
        ax2.grid(True)
        ax2.legend()
        
        # 3. 士气变化图
        ax3 = axes[1, 0]
        
        morale_A_idx = n_units_A + n_units_B
        morale_B_idx = n_units_A + n_units_B + 1
        
        ax3.plot(t, y[:, morale_A_idx], 'b-', linewidth=2, label='A方士气')
        ax3.plot(t, y[:, morale_B_idx], 'r-', linewidth=2, label='B方士气')
        ax3.set_xlabel('时间')
        ax3.set_ylabel('士气')
        ax3.set_title('双方士气变化')
        ax3.set_ylim(0, 1.1)
        ax3.grid(True)
        ax3.legend()
        
        # 4. 疲劳变化图
        ax4 = axes[1, 1]
        
        fatigue_A_idx = n_units_A + n_units_B + 2
        fatigue_B_idx = n_units_A + n_units_B + 3
        
        ax4.plot(t, y[:, fatigue_A_idx], 'b-', linewidth=2, label='A方疲劳')
        ax4.plot(t, y[:, fatigue_B_idx], 'r-', linewidth=2, label='B方疲劳')
        ax4.set_xlabel('时间')
        ax4.set_ylabel('疲劳度')
        ax4.set_title('双方疲劳变化')
        ax4.set_ylim(0, 1.1)
        ax4.grid(True)
        ax4.legend()
        
        # 设置总标题
        if title:
            fig.suptitle(title, fontsize=16)
        else:
            fig.suptitle('高级SLG战斗模拟结果', fontsize=16)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.show()
    
    def analyze_battle_result(self, t, y, params):
        """
        分析战斗结果
        
        参数:
        t: 时间点数组
        y: 状态数组
        params: 模型参数
        
        返回:
        result: 结果字典
        """
        n_units_A = params['n_units_A']
        n_units_B = params['n_units_B']
        unit_names = params.get('unit_names', None)
        
        if unit_names is None:
            unit_names = [f'A{i+1}' for i in range(n_units_A)] + [f'B{i+1}' for i in range(n_units_B)]
        
        # 获取最终状态
        final_state = y[-1]
        
        # 分离各部分数据
        A_final = final_state[:n_units_A]
        B_final = final_state[n_units_A:n_units_A+n_units_B]
        morale_A_final = final_state[n_units_A+n_units_B]
        morale_B_final = final_state[n_units_A+n_units_B+1]
        fatigue_A_final = final_state[n_units_A+n_units_B+2]
        fatigue_B_final = final_state[n_units_A+n_units_B+3]
        
        # 计算总剩余兵力
        A_total_final = np.sum(np.maximum(A_final, 0))
        B_total_final = np.sum(np.maximum(B_final, 0))
        
        # 计算损失率
        A_total_initial = params['A_total_initial']
        B_total_initial = params['B_total_initial']
        
        A_loss_ratio = 1 - A_total_final / A_total_initial if A_total_initial > 0 else 1
        B_loss_ratio = 1 - B_total_final / B_total_initial if B_total_initial > 0 else 1
        
        # 判断胜负
        if A_total_final > 0 and B_total_final <= 0:
            winner = 'A'
        elif B_total_final > 0 and A_total_final <= 0:
            winner = 'B'
        else:
            # 如果双方都有剩余兵力，比较剩余比例
            if A_loss_ratio < B_loss_ratio:
                winner = 'A (优势)'
            elif B_loss_ratio < A_loss_ratio:
                winner = 'B (优势)'
            else:
                winner = '平局'
        
        # 计算战斗持续时间
        battle_end_idx = np.argmax((np.sum(y[:, :n_units_A], axis=1) <= 0) | 
                                  (np.sum(y[:, n_units_A:n_units_A+n_units_B], axis=1) <= 0))
        
        if battle_end_idx > 0:
            battle_duration = t[battle_end_idx]
        else:
            battle_duration = t[-1]  # 战斗持续到模拟结束
        
        # 构建结果字典
        result = {
            'winner': winner,
            'battle_duration': battle_duration,
            'A_total_final': A_total_final,
            'B_total_final': B_total_final,
            'A_loss_ratio': A_loss_ratio,
            'B_loss_ratio': B_loss_ratio,
            'morale_A_final': morale_A_final,
            'morale_B_final': morale_B_final,
            'fatigue_A_final': fatigue_A_final,
            'fatigue_B_final': fatigue_B_final,
            'unit_results': {}
        }
        
        # 添加各兵种结果
        for i in range(n_units_A + n_units_B):
            if i < n_units_A:
                initial = params.get('A_forces', [0] * n_units_A)[i]
                final = max(0, A_final[i])
                loss_ratio = 1 - final / initial if initial > 0 else 1
            else:
                j = i - n_units_A
                initial = params.get('B_forces', [0] * n_units_B)[j]
                final = max(0, B_final[j])
                loss_ratio = 1 - final / initial if initial > 0 else 1
            
            result['unit_results'][unit_names[i]] = {
                'initial': initial,
                'final': final,
                'loss_ratio': loss_ratio
            }
        
        return result


class StrategyAnalysis:
    """
    战略分析工具，用于评估不同战略选择的效果
    """
    
    def __init__(self, model):
        """
        初始化战略分析工具
        
        参数:
        model: 高级SLG模型实例
        """
        self.model = model
    
    def compare_strategies(self, strategies, A_forces, B_forces, base_params, t_max=100):
        """
        比较不同战略的效果
        
        参数:
        strategies: 战略列表，每个元素是一个字典，包含战略名称和参数修改
        A_forces: A方各兵种初始兵力数组
        B_forces: B方各兵种初始兵力数组
        base_params: 基础参数
        t_max: 最大模拟时间
        
        返回:
        results: 结果字典，包含每个战略的模拟结果
        """
        results = {}
        
        for strategy in strategies:
            strategy_name = strategy['name']
            strategy_params = strategy['params']
            
            # 创建参数副本并应用战略参数
            params = base_params.copy()
            for key, value in strategy_params.items():
                params[key] = value
            
            # 保存初始兵力信息
            params['A_forces'] = A_forces.copy()
            params['B_forces'] = B_forces.copy()
            
            # 运行模拟
            t, y = self.model.simulate_advanced(A_forces, B_forces, params, t_max=t_max)
            
            # 分析结果
            battle_result = self.model.analyze_battle_result(t, y, params)
            
            # 存储结果
            results[strategy_name] = {
                't': t,
                'y': y,
                'params': params,
                'battle_result': battle_result
            }
        
        return results
    
    def plot_strategy_comparison(self, results):
        """
        绘制不同战略的比较结果
        
        参数:
        results: compare_strategies返回的结果字典
        """
        # 提取战略名称和结果
        strategy_names = list(results.keys())
        n_strategies = len(strategy_names)
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 总兵力损失比较
        ax1 = axes[0, 0]
        
        A_loss_ratios = [results[name]['battle_result']['A_loss_ratio'] for name in strategy_names]
        B_loss_ratios = [results[name]['battle_result']['B_loss_ratio'] for name in strategy_names]
        
        x = np.arange(n_strategies)
        width = 0.35
        
        ax1.bar(x - width/2, A_loss_ratios, width, label='A方损失率')
        ax1.bar(x + width/2, B_loss_ratios, width, label='B方损失率')
        
        ax1.set_ylabel('损失率')
        ax1.set_title('不同战略下的兵力损失率比较')
        ax1.set_xticks(x)
        ax1.set_xticklabels(strategy_names)
        ax1.set_ylim(0, 1.1)
        ax1.legend()
        
        # 2. 战斗持续时间比较
        ax2 = axes[0, 1]
        
        durations = [results[name]['battle_result']['battle_duration'] for name in strategy_names]
        
        ax2.bar(x, durations, width)
        ax2.set_ylabel('时间')
        ax2.set_title('不同战略下的战斗持续时间比较')
        ax2.set_xticks(x)
        ax2.set_xticklabels(strategy_names)
        
        # 3. 士气最终值比较
        ax3 = axes[1, 0]
        
        morale_A_finals = [results[name]['battle_result']['morale_A_final'] for name in strategy_names]
        morale_B_finals = [results[name]['battle_result']['morale_B_final'] for name in strategy_names]
        
        ax3.bar(x - width/2, morale_A_finals, width, label='A方士气')
        ax3.bar(x + width/2, morale_B_finals, width, label='B方士气')
        
        ax3.set_ylabel('士气')
        ax3.set_title('不同战略下的最终士气比较')
        ax3.set_xticks(x)
        ax3.set_xticklabels(strategy_names)
        ax3.set_ylim(0, 1.1)
        ax3.legend()
        
        # 4. 胜负结果
        ax4 = axes[1, 1]
        
        winners = [results[name]['battle_result']['winner'] for name in strategy_names]
        
        # 为不同的获胜方设置不同的颜色
        colors = []
        for winner in winners:
            if winner.startswith('A'):
                colors.append('blue')
            elif winner.startswith('B'):
                colors.append('red')
            else:
                colors.append('gray')
        
        ax4.bar(x, [1] * n_strategies, color=colors)
        
        # 在柱状图上添加获胜方标签
        for i, winner in enumerate(winners):
            ax4.text(i, 0.5, winner, ha='center', va='center', color='white', fontweight='bold')
        
        ax4.set_title('不同战略下的战斗结果')
        ax4.set_yticks([])
        ax4.set_xticks(x)
        ax4.set_xticklabels(strategy_names)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        fig.suptitle('战略效果比较', fontsize=16)
        plt.show()
    
    def plot_unit_performance(self, results, strategy_name):
        """
        绘制特定战略下各兵种的表现
        
        参数:
        results: compare_strategies返回的结果字典
        strategy_name: 要分析的战略名称
        """
        if strategy_name not in results:
            print(f"错误：找不到战略 '{strategy_name}'")
            return
        
        result = results[strategy_name]
        battle_result = result['battle_result']
        unit_results = battle_result['unit_results']
        
        # 提取单位名称和损失率
        unit_names = list(unit_results.keys())
        loss_ratios = [unit_results[name]['loss_ratio'] for name in unit_names]
        
        # 区分A方和B方单位
        A_units = [name for name in unit_names if name.startswith('A')]
        B_units = [name for name in unit_names if name.startswith('B')]
        
        # 创建图表
        plt.figure(figsize=(12, 6))
        
        # 绘制损失率条形图
        x = np.arange(len(unit_names))
        colors = ['blue' if name.startswith('A') else 'red' for name in unit_names]
        
        plt.bar(x, loss_ratios, color=colors)
        plt.axhline(y=1.0, color='gray', linestyle='--')
        
        plt.ylabel('损失率')
        plt.title(f'战略 "{strategy_name}" 下各兵种的损失率')
        plt.xticks(x, unit_names, rotation=45)
        plt.ylim(0, 1.1)
        plt.grid(axis='y')
        
        plt.tight_layout()
        plt.show()