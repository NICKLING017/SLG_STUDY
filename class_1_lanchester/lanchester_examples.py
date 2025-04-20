# 兰彻斯特方程模型示例
# 展示如何使用兰彻斯特模型进行SLG战斗模拟与平衡性分析

import numpy as np
import matplotlib.pyplot as plt
from lanchester_model import LanchesterModel, SLGLanchesterModel, SensitivityAnalysis


# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def basic_model_example():
    """
    基础兰彻斯特模型示例
    比较线性律和平方律的差异
    """
    print("\n=== 基础兰彻斯特模型示例 ===")
    
    # 初始参数设置
    A0 = 1000  # A方初始兵力
    B0 = 1200  # B方初始兵力
    params = {'a': 0.05, 'b': 0.04}  # 战斗效能系数
    
    # 创建线性律模型
    linear_model = LanchesterModel(model_type='linear')
    t_linear, y_linear = linear_model.simulate(A0, B0, params)
    
    # 创建平方律模型
    square_model = LanchesterModel(model_type='square')
    t_square, y_square = square_model.simulate(A0, B0, params)
    
    # 绘制结果比较
    plt.figure(figsize=(12, 10))
    
    # 线性律结果
    plt.subplot(2, 1, 1)
    plt.plot(t_linear, y_linear[:, 0], 'b-', label='A方兵力')
    plt.plot(t_linear, y_linear[:, 1], 'r-', label='B方兵力')
    plt.title('兰彻斯特线性律模拟结果')
    plt.xlabel('时间')
    plt.ylabel('兵力')
    plt.grid(True)
    plt.legend()
    
    # 平方律结果
    plt.subplot(2, 1, 2)
    plt.plot(t_square, y_square[:, 0], 'b-', label='A方兵力')
    plt.plot(t_square, y_square[:, 1], 'r-', label='B方兵力')
    plt.title('兰彻斯特平方律模拟结果')
    plt.xlabel('时间')
    plt.ylabel('兵力')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # 分析结果
    print(f"线性律模型结果: A方剩余兵力 = {max(0, y_linear[-1, 0]):.2f}, B方剩余兵力 = {max(0, y_linear[-1, 1]):.2f}")
    print(f"平方律模型结果: A方剩余兵力 = {max(0, y_square[-1, 0]):.2f}, B方剩余兵力 = {max(0, y_square[-1, 1]):.2f}")
    
    # 兰彻斯特平方律的理论预测
    # 根据平方律，如果 A0²/B0² > b/a，则A获胜，否则B获胜
    theoretical_ratio = (A0**2) / (B0**2)
    combat_power_ratio = params['b'] / params['a']
    
    print(f"\n理论分析:")
    print(f"初始兵力平方比 A0²/B0² = {theoretical_ratio:.4f}")
    print(f"战斗力比值 b/a = {combat_power_ratio:.4f}")
    
    if theoretical_ratio > combat_power_ratio:
        print("根据兰彻斯特平方律理论，A方应该获胜")
    else:
        print("根据兰彻斯特平方律理论，B方应该获胜")


def slg_extended_model_example():
    """
    SLG扩展兰彻斯特模型示例
    模拟包含多兵种、克制关系、地形和士气的战斗
    """
    print("\n=== SLG扩展兰彻斯特模型示例 ===")
    
    # 创建扩展模型
    slg_model = SLGLanchesterModel()
    
    # 设置兵种和初始兵力
    # A方有3种兵种：步兵、骑兵、弓箭手
    A_forces = np.array([500, 300, 200])  # [步兵, 骑兵, 弓箭手]
    
    # B方有3种兵种：步兵、骑兵、弓箭手
    B_forces = np.array([600, 250, 150])  # [步兵, 骑兵, 弓箭手]
    
    # 兵种名称
    unit_names = ['A步兵', 'A骑兵', 'A弓箭手', 'B步兵', 'B骑兵', 'B弓箭手']
    
    # 设置效能矩阵（兵种克制关系）
    # 矩阵大小为 (n_units_A + n_units_B) x (n_units_A + n_units_B)
    # 矩阵元素 (i,j) 表示i兵种对j兵种的效能
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
    
    # 设置模拟参数
    params = {
        'effectiveness_matrix': effectiveness_matrix,
        'terrain_factor_A': 1.2,  # A方地形加成
        'terrain_factor_B': 1.0,  # B方无地形加成
        'morale_A': 1.1,          # A方士气较高
        'morale_B': 0.9           # B方士气较低
    }
    
    # 运行模拟
    t, y = slg_model.simulate_extended(A_forces, B_forces, params, t_max=150)
    
    # 绘制结果
    slg_model.plot_extended_simulation(t, y, unit_names, title='SLG多兵种战斗模拟')
    
    # 分析结果
    print("战斗结束时各兵种剩余兵力:")
    for i, name in enumerate(unit_names):
        print(f"{name}: {max(0, y[-1, i]):.2f}")
    
    # 计算总剩余兵力
    A_remaining = sum(max(0, y[-1, i]) for i in range(len(A_forces)))
    B_remaining = sum(max(0, y[-1, i]) for i in range(len(A_forces), len(A_forces) + len(B_forces)))
    
    print(f"\nA方总剩余兵力: {A_remaining:.2f}")
    print(f"B方总剩余兵力: {B_remaining:.2f}")
    
    if A_remaining > 0 and B_remaining <= 0:
        print("A方获胜")
    elif B_remaining > 0 and A_remaining <= 0:
        print("B方获胜")
    else:
        if A_remaining > B_remaining:
            print(f"A方占优 (剩余兵力比: {A_remaining/B_remaining:.2f})")
        elif B_remaining > A_remaining:
            print(f"B方占优 (剩余兵力比: {B_remaining/A_remaining:.2f})")
        else:
            print("战斗势均力敌")


def sensitivity_analysis_example():
    """
    敏感性分析示例
    分析参数变化对战斗结果的影响
    """
    print("\n=== 敏感性分析示例 ===")
    
    # 创建基础模型和敏感性分析工具
    model = LanchesterModel(model_type='square')
    analyzer = SensitivityAnalysis(model)
    
    # 初始参数设置
    A0 = 1000  # A方初始兵力
    B0 = 1200  # B方初始兵力
    fixed_params = {'a': 0.05, 'b': 0.04}  # 基础战斗效能系数
    
    # 1. 分析A方战斗效能系数变化的影响
    print("\n分析A方战斗效能系数(a)变化的影响...")
    a_values = np.linspace(0.03, 0.07, 9)  # 从0.03到0.07的9个值
    a_results = analyzer.parameter_sweep('a', a_values, A0, B0, fixed_params)
    
    # 绘制a参数扫描结果
    analyzer.plot_parameter_sweep_results(a_results, 'a', metric='winner')
    analyzer.plot_parameter_sweep_results(a_results, 'a', metric='remaining_A')
    
    # 2. 分析B方战斗效能系数变化的影响
    print("\n分析B方战斗效能系数(b)变化的影响...")
    b_values = np.linspace(0.03, 0.07, 9)  # 从0.03到0.07的9个值
    b_results = analyzer.parameter_sweep('b', b_values, A0, B0, fixed_params)
    
    # 绘制b参数扫描结果
    analyzer.plot_parameter_sweep_results(b_results, 'b', metric='winner')
    analyzer.plot_parameter_sweep_results(b_results, 'b', metric='remaining_B')
    
    # 3. 热图分析：同时分析a和b的变化
    print("\n热图分析：同时分析a和b的变化...")
    a_values_heatmap = np.linspace(0.03, 0.07, 10)
    b_values_heatmap = np.linspace(0.03, 0.07, 10)
    
    # 绘制热图
    analyzer.heatmap_analysis('a', a_values_heatmap, 'b', b_values_heatmap, 
                            A0, B0, {}, metric='winner')
    
    # 4. 分析初始兵力比例的影响
    print("\n分析初始兵力比例的影响...")
    # 保持总兵力不变，改变比例
    total_forces = A0 + B0
    A_ratios = np.linspace(0.3, 0.7, 9)  # A方占总兵力的比例从30%到70%
    
    ratio_results = {}
    for ratio in A_ratios:
        A0_new = int(total_forces * ratio)
        B0_new = total_forces - A0_new
        t, y = model.simulate(A0_new, B0_new, fixed_params)
        
        ratio_results[ratio] = {
            't': t,
            'y': y,
            'winner': 'A' if y[-1, 0] > 0 and y[-1, 1] <= 0 else 
                     'B' if y[-1, 1] > 0 and y[-1, 0] <= 0 else 'Draw',
            'remaining_A': max(0, y[-1, 0]),
            'remaining_B': max(0, y[-1, 1])
        }
    
    # 绘制兵力比例影响结果
    plt.figure(figsize=(10, 6))
    
    # 为不同的获胜方设置不同的颜色
    winners = [ratio_results[r]['winner'] for r in A_ratios]
    colors = {'A': 'blue', 'B': 'red', 'Draw': 'gray'}
    color_values = [colors[w] for w in winners]
    
    plt.scatter(A_ratios, [1] * len(A_ratios), c=color_values, s=100)
    plt.yticks([])
    plt.xlabel('A方兵力占比')
    plt.title('初始兵力比例对战斗结果的影响')
    
    # 添加图例
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=10, label=l)
                      for l, c in colors.items()]
    plt.legend(handles=legend_elements)
    
    plt.show()
    
    # 绘制剩余兵力
    plt.figure(figsize=(10, 6))
    plt.plot(A_ratios, [ratio_results[r]['remaining_A'] for r in A_ratios], 'bo-', label='A方剩余兵力')
    plt.plot(A_ratios, [ratio_results[r]['remaining_B'] for r in A_ratios], 'ro-', label='B方剩余兵力')
    plt.xlabel('A方兵力占比')
    plt.ylabel('剩余兵力')
    plt.title('初始兵力比例对剩余兵力的影响')
    plt.grid(True)
    plt.legend()
    plt.show()


def main():
    """
    主函数，运行所有示例
    """
    print("兰彻斯特方程SLG战斗模拟示例")
    print("=============================")
    
    # 运行基础模型示例
    basic_model_example()
    
    # 运行SLG扩展模型示例
    slg_extended_model_example()
    
    # 运行敏感性分析示例
    sensitivity_analysis_example()


if __name__ == "__main__":
    main()