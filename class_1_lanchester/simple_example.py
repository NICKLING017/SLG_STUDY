# 兰彻斯特方程模型简单示例
# 这个脚本展示了如何使用兰彻斯特模型进行基本的战斗模拟

import numpy as np
import matplotlib.pyplot as plt
from lanchester_model import LanchesterModel

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def run_basic_example():
    # 创建一个基本的兰彻斯特模型（默认使用平方律）
    model = LanchesterModel(model_type='square')
    
    # 设置初始参数
    A0 = 1000  # A方初始兵力
    B0 = 1200  # B方初始兵力
    params = {'a': 0.05, 'b': 0.04}  # 战斗效能系数
    
    # 运行模拟
    t, y = model.simulate(A0, B0, params)
    
    # 绘制结果
    model.plot_simulation(t, y, title='兰彻斯特平方律模拟示例')
    
    # 输出结果
    print(f'模拟结束时：')
    print(f'A方剩余兵力: {max(0, y[-1, 0]):.2f}')
    print(f'B方剩余兵力: {max(0, y[-1, 1]):.2f}')
    
    if y[-1, 0] > 0 and y[-1, 1] <= 0:
        print('A方获胜')
    elif y[-1, 1] > 0 and y[-1, 0] <= 0:
        print('B方获胜')
    else:
        print('战斗未分出胜负')

# 如果您想尝试线性律模型，可以取消下面的注释并运行
def run_linear_example():
    # 创建一个线性律兰彻斯特模型
    model = LanchesterModel(model_type='linear')
    
    # 设置初始参数
    A0 = 1000  # A方初始兵力
    B0 = 1200  # B方初始兵力
    params = {'a': 0.05, 'b': 0.04}  # 战斗效能系数
    
    # 运行模拟
    t, y = model.simulate(A0, B0, params)
    
    # 绘制结果
    model.plot_simulation(t, y, title='兰彻斯特线性律模拟示例')
    
    # 输出结果
    print(f'\n模拟结束时：')
    print(f'A方剩余兵力: {max(0, y[-1, 0]):.2f}')
    print(f'B方剩余兵力: {max(0, y[-1, 1]):.2f}')

# 主函数
if __name__ == "__main__":
    print("运行兰彻斯特方程模型示例...\n")
    
    # 运行平方律示例
    run_basic_example()
    
    # 如果您想同时查看线性律示例，取消下面的注释
    # run_linear_example()
    
    print("\n示例运行完成。")