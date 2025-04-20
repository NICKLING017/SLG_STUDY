# 高级SLG战斗模型示例
# 展示如何使用高级SLG战斗模型进行战略分析

import numpy as np
import matplotlib.pyplot as plt
from advanced_slg_model import AdvancedSLGModel, StrategyAnalysis

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def focus_fire_strategy_example():
    """
    集火策略示例
    比较不同集火策略的效果
    """
    print("\n=== 集火策略分析示例 ===")
    
    # 创建高级模型
    model = AdvancedSLGModel()
    
    # 设置兵种和初始兵力
    # A方有3种兵种：步兵、骑兵、弓箭手
    A_forces = np.array([500, 300, 200])  # [步兵, 骑兵, 弓箭手]
    
    # B方有3种兵种：步兵、骑兵、弓箭手
    B_forces = np.array([600, 250, 150])  # [步兵, 骑兵, 弓箭手]
    
    # 兵种名称
    unit_names = ['A步兵', 'A骑兵', 'A弓箭手', 'B步兵', 'B骑兵', 'B弓箭手']
    
    # 设置效能矩阵（兵种克制关系）
    n_units = len(A_forces) + len(B_forces)
    effectiveness_matrix = np.zeros((n_units, n_units))
    
    # 设置A方兵种对B方兵种的效能
    # 步兵对步兵
    effectiveness_matrix[0, 3] = 0.05
    # 步兵对骑兵（克制）
    effectiveness_matrix[0, 4] = 0.08
    # 步兵对弓箭手
    effectiveness_matrix[0, 5] = 0.04
    
    # 骑兵对步兵
    effectiveness_matrix[1, 3] = 0.04
    # 骑兵对骑兵
    effectiveness_matrix[1, 4] = 0.06
    # 骑兵对弓箭手（克制）
    effectiveness_matrix[1, 5] = 0.09
    
    # 弓箭手对步兵
    effectiveness_matrix[2, 3] = 0.06
    # 弓箭手对骑兵
    effectiveness_matrix[2, 4] = 0.03
    # 弓箭手对弓箭手
    effectiveness_matrix[2, 5] = 0.05
    
    # 设置B方兵种对A方兵种的效能
    # 步兵对步兵
    effectiveness_matrix[3, 0] = 0.05
    # 步兵对骑兵（克制）
    effectiveness_matrix[3, 1] = 0.08
    # 步兵对弓箭手
    effectiveness_matrix[3, 2] = 0.04
    
    # 骑兵对步兵
    effectiveness_matrix[4, 0] = 0.04
    # 骑兵对骑兵
    effectiveness_matrix[4, 1] = 0.06
    # 骑兵对弓箭手（克制）
    effectiveness_matrix[4, 2] = 0.09
    
    # 弓箭手对步兵
    effectiveness_matrix[5, 0] = 0.06
    # 弓箭手对骑兵
    effectiveness_matrix[5, 1] = 0.03
    # 弓箭手对弓箭手
    effectiveness_matrix[5, 2] = 0.05
    
    # 设置基础参数
    base_params = {
        'effectiveness_matrix': effectiveness_matrix,
        'terrain_factor_A': 1.0,
        'terrain_factor_B': 1.0,
        'initial_morale_A': 1.0,
        'initial_morale_B': 1.0,
        'initial_fatigue_A': 0.0,
        'initial_fatigue_B': 0.0,
        'unit_names': unit_names,
        'ranged_units_A': [False, False, True],  # 只有弓箭手是远程单位
        'ranged_units_B': [False, False, True],
        'ranged_advantage_time': 30  # 远程单位在前30个时间单位有优势
    }
    
    # 创建战略分析工具
    analyzer = StrategyAnalysis(model)
    
    # 定义不同的集火策略
    strategies = [
        {
            'name': '无集火策略',
            'params': {
                'focus_fire_A': False,
                'focus_fire_B': False
            }
        },
        {
            'name': 'A集火B弓箭手',
            'params': {
                'focus_fire_A': True,
                'focus_target_A': 2,  # B方弓箭手的索引
                'focus_fire_B': False
            }
        },
        {
            'name': 'B集火A弓箭手',
            'params': {
                'focus_fire_A': False,
                'focus_fire_B': True,
                'focus_target_B': 2  # A方弓箭手的索引
            }
        },
        {
            'name': '双方集火对方弓箭手',
            'params': {
                'focus_fire_A': True,
                'focus_target_A': 2,
                'focus_fire_B': True,
                'focus_target_B': 2
            }
        }
    ]
    
    # 比较不同策略
    results = analyzer.compare_strategies(strategies, A_forces, B_forces, base_params, t_max=150)
    
    # 绘制策略比较结果
    analyzer.plot_strategy_comparison(results)
    
    # 分析特定策略下各兵种的表现
    analyzer.plot_unit_performance(results, '双方集火对方弓箭手')
    
    # 打印详细结果
    print("\n各策略战斗结果:")
    for strategy_name, result in results.items():
        battle_result = result['battle_result']
        print(f"\n策略: {strategy_name}")
        print(f"胜方: {battle_result['winner']}")
        print(f"战斗持续时间: {battle_result['battle_duration']:.2f}")
        print(f"A方损失率: {battle_result['A_loss_ratio']:.2%}")
        print(f"B方损失率: {battle_result['B_loss_ratio']:.2%}")


def terrain_morale_example():
    """
    地形和士气影响示例
    分析地形加成和士气对战斗结果的影响
    """
    print("\n=== 地形和士气影响分析示例 ===")
    
    # 创建高级模型
    model = AdvancedSLGModel()
    
    # 设置兵种和初始兵力
    A_forces = np.array([400, 300, 300])  # [步兵, 骑兵, 弓箭手]
    B_forces = np.array([400, 300, 300])  # [步兵, 骑兵, 弓箭手]
    
    # 兵种名称
    unit_names = ['A步兵', 'A骑兵', 'A弓箭手', 'B步兵', 'B骑兵', 'B弓箭手']
    
    # 设置效能矩阵（与前例相同）
    n_units = len(A_forces) + len(B_forces)
    effectiveness_matrix = np.zeros((n_units, n_units))
    
    # 设置A方兵种对B方兵种的效能
    effectiveness_matrix[0, 3] = 0.05  # 步兵对步兵
    effectiveness_matrix[0, 4] = 0.08  # 步兵对骑兵（克制）
    effectiveness_matrix[0, 5] = 0.04  # 步兵对弓箭手
    effectiveness_matrix[1, 3] = 0.04  # 骑兵对步兵
    effectiveness_matrix[1, 4] = 0.06  # 骑兵对骑兵
    effectiveness_matrix[1, 5] = 0.09  # 骑兵对弓箭手（克制）
    effectiveness_matrix[2, 3] = 0.06  # 弓箭手对步兵
    effectiveness_matrix[2, 4] = 0.03  # 弓箭手对骑兵
    effectiveness_matrix[2, 5] = 0.05  # 弓箭手对弓箭手
    
    # 设置B方兵种对A方兵种的效能
    effectiveness_matrix[3, 0] = 0.05  # 步兵对步兵
    effectiveness_matrix[3, 1] = 0.08  # 步兵对骑兵（克制）
    effectiveness_matrix[3, 2] = 0.04  # 步兵对弓箭手
    effectiveness_matrix[4, 0] = 0.04  # 骑兵对步兵
    effectiveness_matrix[4, 1] = 0.06  # 骑兵对骑兵
    effectiveness_matrix[4, 2] = 0.09  # 骑兵对弓箭手（克制）
    effectiveness_matrix[5, 0] = 0.06  # 弓箭手对步兵
    effectiveness_matrix[5, 1] = 0.03  # 弓箭手对骑兵
    effectiveness_matrix[5, 2] = 0.05  # 弓箭手对弓箭手
    
    # 设置基础参数
    base_params = {
        'effectiveness_matrix': effectiveness_matrix,
        'unit_names': unit_names,
        'ranged_units_A': [False, False, True],
        'ranged_units_B': [False, False, True],
        'ranged_advantage_time': 30
    }
    
    # 创建战略分析工具
    analyzer = StrategyAnalysis(model)
    
    # 定义不同的地形和士气组合
    strategies = [
        {
            'name': '无地形加成，士气相等',
            'params': {
                'terrain_factor_A': 1.0,
                'terrain_factor_B': 1.0,
                'initial_morale_A': 1.0,
                'initial_morale_B': 1.0
            }
        },
        {
            'name': 'A方地形优势',
            'params': {
                'terrain_factor_A': 1.3,
                'terrain_factor_B': 1.0,
                'initial_morale_A': 1.0,
                'initial_morale_B': 1.0
            }
        },
        {
            'name': 'A方士气优势',
            'params': {
                'terrain_factor_A': 1.0,
                'terrain_factor_B': 1.0,
                'initial_morale_A': 1.2,
                'initial_morale_B': 0.9
            }
        },
        {
            'name': 'A方地形和士气双优势',
            'params': {
                'terrain_factor_A': 1.3,
                'terrain_factor_B': 1.0,
                'initial_morale_A': 1.2,
                'initial_morale_B': 0.9
            }
        }
    ]
    
    # 比较不同策略
    results = analyzer.compare_strategies(strategies, A_forces, B_forces, base_params, t_max=150)
    
    # 绘制策略比较结果
    analyzer.plot_strategy_comparison(results)
    
    # 打印详细结果
    print("\n各场景战斗结果:")
    for strategy_name, result in results.items():
        battle_result = result['battle_result']
        print(f"\n场景: {strategy_name}")
        print(f"胜方: {battle_result['winner']}")
        print(f"战斗持续时间: {battle_result['battle_duration']:.2f}")
        print(f"A方损失率: {battle_result['A_loss_ratio']:.2%}")
        print(f"B方损失率: {battle_result['B_loss_ratio']:.2%}")
        print(f"A方最终士气: {battle_result['morale_A_final']:.2f}")
        print(f"B方最终士气: {battle_result['morale_B_final']:.2f}")


def fatigue_supply_example():
    """
    疲劳和补给示例
    分析疲劳和补给对长时间战斗的影响
    """
    print("\n=== 疲劳和补给影响分析示例 ===")
    
    # 创建高级模型
    model = AdvancedSLGModel()
    
    # 设置兵种和初始兵力（较大的兵力以延长战斗时间）
    A_forces = np.array([800, 600, 400])  # [步兵, 骑兵, 弓箭手]
    B_forces = np.array([800, 600, 400])  # [步兵, 骑兵, 弓箭手]
    
    # 兵种名称
    unit_names = ['A步兵', 'A骑兵', 'A弓箭手', 'B步兵', 'B骑兵', 'B弓箭手']
    
    # 设置效能矩阵（较低的效能以延长战斗时间）
    n_units = len(A_forces) + len(B_forces)
    effectiveness_matrix = np.zeros((n_units, n_units))
    
    # 设置较低的战斗效能
    for i in range(3):
        for j in range(3, 6):
            effectiveness_matrix[i, j] = 0.03
    
    for i in range(3, 6):
        for j in range(3):
            effectiveness_matrix[i, j] = 0.03
    
    # 设置基础参数
    base_params = {
        'effectiveness_matrix': effectiveness_matrix,
        'terrain_factor_A': 1.0,
        'terrain_factor_B': 1.0,
        'initial_morale_A': 1.0,
        'initial_morale_B': 1.0,
        'unit_names': unit_names,
        'ranged_units_A': [False, False, True],
        'ranged_units_B': [False, False, True]
    }
    
    # 创建战略分析工具
    analyzer = StrategyAnalysis(model)
    
    # 定义不同的疲劳和补给组合
    strategies = [
        {
            'name': '无疲劳效果',
            'params': {
                'initial_fatigue_A': 0.0,
                'initial_fatigue_B': 0.0,
                'supply_rate_A': 0.0,
                'supply_rate_B': 0.0
            }
        },
        {
            'name': '有疲劳无补给',
            'params': {
                'initial_fatigue_A': 0.0,
                'initial_fatigue_B': 0.0,
                'supply_rate_A': 0.0,
                'supply_rate_B': 0.0
            }
        },
        {
            'name': 'A方有补给',
            'params': {
                'initial_fatigue_A': 0.0,
                'initial_fatigue_B': 0.0,
                'supply_rate_A': 0.02,
                'supply_rate_B': 0.0
            }
        },
        {
            'name': 'B方有补给',
            'params': {
                'initial_fatigue_A': 0.0,
                'initial_fatigue_B': 0.0,
                'supply_rate_A': 0.0,
                'supply_rate_B': 0.02
            }
        }
    ]
    
    # 比较不同策略
    results = analyzer.compare_strategies(strategies, A_forces, B_forces, base_params, t_max=300)
    
    # 绘制策略比较结果
    analyzer.plot_strategy_comparison(results)
    
    # 为特定策略绘制详细的战斗过程
    for strategy_name in ['有疲劳无补给', 'A方有补给']:
        result = results[strategy_name]
        t, y = result['t'], result['y']
        params = result['params']
        
        model.plot_advanced_simulation(t, y, params, title=f'战略 "{strategy_name}" 的战斗过程')
    
    # 打印详细结果
    print("\n各场景战斗结果:")
    for strategy_name, result in results.items():
        battle_result = result['battle_result']
        print(f"\n场景: {strategy_name}")
        print(f"胜方: {battle_result['winner']}")
        print(f"战斗持续时间: {battle_result['battle_duration']:.2f}")
        print(f"A方损失率: {battle_result['A_loss_ratio']:.2%}")
        print(f"B方损失率: {battle_result['B_loss_ratio']:.2%}")
        print(f"A方最终疲劳: {battle_result['fatigue_A_final']:.2f}")
        print(f"B方最终疲劳: {battle_result['fatigue_B_final']:.2f}")


def main():
    """
    主函数，运行所有示例
    """
    print("高级SLG战斗模型示例")
    print("====================")
    
    # 运行集火策略示例
    focus_fire_strategy_example()
    
    # 运行地形和士气影响示例
    terrain_morale_example()
    
    # 运行疲劳和补给示例
    fatigue_supply_example()


if __name__ == "__main__":
    main()