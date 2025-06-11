import numpy as np
import matplotlib.pyplot as plt
import os

# 文件列表
file_list = [
    './multi_cifar10_result/multi_cifar10_rmattack_result_test_accuracy_50_1.npy',
    # './multi_cifar10_result/multi_cifar10_rmattack_result_test_accuracy_10_50_10.npy',
    # './multi_cifar10_result/multi_cifar10_rmattack_result_test_accuracy_10_50_30.npy',
    # './multi_cifar10_result/multi_cifar10_result_test_accuracy_10_50_50.npy',
]

plt.figure(figsize=(12, 8)) # 创建一个图并设置大小

# 循环读取文件并绘图
for file_path in file_list:
    if not os.path.exists(file_path):
        print(f"错误：文件未找到 - {file_path}")
        continue 

    try:
        # 加载数据
        data = np.load(file_path)

        # 假设数据是测试精度随时间步长或 epoch 变化的序列
        # x轴通常是步长或 epoch，我们使用数据的索引作为x轴
        x_values = range(len(data))

        # 从文件名中提取用于图例的标签
        # 提取 '50_1', '50_10', '50_30', '50_50' 部分
        label = file_path.replace('multi_cifar10_result_test_accuracy_', '').replace('.npy', '')

        # 绘图，matplotlib 自动使用不同的颜色和线型
        plt.plot(x_values, data, label=label)

    except Exception as e:
        print(f"处理文件时发生错误 {file_path}: {e}")

# 添加图表标题和轴标签
plt.title('Test Accuracy Comparison')
plt.xlabel('Epoch') 
plt.ylabel('Test Accuracy')

# 添加图例，显示每个线条代表哪个文件/设置
plt.legend()

# 添加网格线
plt.grid(True)

# 显示图表
plt.savefig('./multi_cifar10_result/multi_cifar10_test_accuracy_result.png')