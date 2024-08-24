# import matplotlib.pyplot as plt
# import numpy as np

# # shuju
# stages = ['10', '20', '30', '40', '50', '70', '80']

# models = ['10', '20', '30', '40', '50', '70', '80']

# CO_ResNetRS50 = [98.98, 91.49, 83.90, 93.15, 86.89, 94.44, 97.02]
# ShuffleNet = [97.34, 89.79, 80.68, 91.02, 80.33, 91.98, 96.42]
# VGG16 = [96.11, 85.63, 71.83, 85.24, 70.08, 85.49, 93.44]
# EfficientNet = [97.14, 90.93, 77.06, 90.41, 88.52, 92.90, 95.83]
# ConvNext = [96.73, 88.28, 70.22, 87.52, 65.16, 89.51, 94.83]
# AlexNet = [96.73, 87.15, 74.85, 83.87, 63.93, 89.20, 94.43]


# bar_width = 0.2
# group_spacing = 0.4
# index = np.arange(len(models)) * (bar_width+group_spacing)

# plt.bar(index, CO_ResNetRS50, bar_width, label='CO-ResNetRS50')
# plt.bar(index + bar_width, ShuffleNet, bar_width, label='ShuffleNet')
# plt.bar(index + 2 * bar_width, VGG16, bar_width, label='VGG16')
# plt.bar(index + 3 * bar_width, EfficientNet, bar_width, label='EfficientNet')
# plt.bar(index + 4 * bar_width, ConvNext, bar_width, label='ConvNext')
# plt.bar(index + 5 * bar_width, AlexNet, bar_width, label='AlexNet')


# plt.xlabel('The growth stage')
# plt.ylabel('Value of accuracy')
# plt.title('duibi')

# plt.legend()
# plt.xticks([i + bar_width + group_spacing / 2 for  i in index], models)



# plt.grid(True)
# plt.tight_layout()
# plt.show()

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams.update({'font.size': 10})

# 数据
categories = ['1x', '2x', '3x', '4x', '5x', '7x', '8x']  # 类别


ShuffleNet = [97.34, 89.79, 80.68, 91.02, 80.33, 91.98, 96.42]   # 第一组数据
VGG16 = [96.11, 85.63, 71.83, 85.24, 70.08, 85.49, 93.44]        # 第二组数据
EfficientNet = [97.14, 90.93, 77.06, 90.41, 88.52, 92.90, 95.83]
ConvNext = [96.73, 88.28, 70.22, 87.52, 65.16, 89.51, 94.83]
AlexNet = [96.73, 87.15, 74.85, 83.87, 63.93, 89.20, 94.43]
CO_ResNetRS50 = [98.98, 91.49, 83.90, 93.15, 86.89, 94.44, 97.02]


# 设置条形图的宽度
bar_width = 0.13

# 创建图表并设置尺寸
plt.figure(figsize=(7, 4))

# 创建图表
# plt.bar(categories, CO_ResNetRS50, width=bar_width, color='#7FABD1', label='CO-ResNetRS50')
# plt.bar([x + bar_width for x in range(len(categories))], ShuffleNet, width=bar_width, color='#EC6E66', label='ShuffleNet')
# plt.bar([x + bar_width*2 for x in range(len(categories))], VGG16, width=bar_width, color='#91CCC0', label='VGG16')
# plt.bar([x + bar_width*3 for x in range(len(categories))], EfficientNet, width=bar_width, color='#F7AC53', label='EfficientNet')
# plt.bar([x + bar_width*4 for x in range(len(categories))], ConvNext, width=bar_width, color='#B5CE4E', label='ConvNext')
# plt.bar([x + bar_width*5 for x in range(len(categories))], AlexNet, width=bar_width, color='#BD7795', label='AlexNet')

plt.bar(categories, ShuffleNet, width=bar_width, color='#EC6E66', label='ShuffleNet')
plt.bar([x + bar_width for x in range(len(categories))], VGG16, width=bar_width, color='#91CCC0', label='VGG16')
plt.bar([x + bar_width*2 for x in range(len(categories))], EfficientNet, width=bar_width, color='#F7AC53', label='EfficientNet')
plt.bar([x + bar_width*3 for x in range(len(categories))], ConvNext, width=bar_width, color='#B5CE4E', label='ConvNext')
plt.bar([x + bar_width*4 for x in range(len(categories))], AlexNet, width=bar_width, color='#BD7795', label='AlexNet')
plt.bar([x + bar_width*5 for x in range(len(categories))], CO_ResNetRS50, width=bar_width, color='#7FABD1', label='CO-ResNetRS50-SSL')

# 添加标题和标签
plt.title('')
plt.xlabel('BBCH')
plt.ylabel('Accuracy(%)')

# 添加图例
plt.legend()

# 调整X轴标签的位置
plt.xticks([x + bar_width*2.5 for x in range(len(categories))], categories)
# 设置Y轴范围从50%开始
plt.ylim(50, 100)
# 设置Y轴刻度间隔为5
plt.yticks(range(50, 101, 5))

# 添加图例
plt.legend(bbox_to_anchor=(0.5, 1.2), loc='upper center', ncol=3, columnspacing=5.2, frameon=False)  # ncol表示图例的列数

# 调整图的位置，使其向上移动
plt.subplots_adjust(bottom=0.18)


# plt.tight_layout()

save_path = "D:/Desktop/huatu2/tiaoxingtu333.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')

# 显示图表
plt.show()
