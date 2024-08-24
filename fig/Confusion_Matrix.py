# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns



# plt.rcParams['font.family'] = 'Times New Roman'
# plt.rcParams.update({'font.size': 10})

# # CO-ResNetRS50
# # cm = np.array([[484, 5, 0, 0, 0, 0, 0],
# #                [19, 484, 24, 2, 0, 0, 0],
# #                [0, 28, 417, 43, 7, 2, 0],
# #                [0, 0, 37, 612, 3, 4, 1],
# #                [0, 0, 6, 19, 212, 7, 0],
# #                [0, 2, 1, 1, 3, 306, 11],
# #                [0, 0, 0, 3, 1, 11, 488]])

# # ShuffleNet
# # cm = np.array([[476, 13, 0, 0, 0, 0, 0],
# #                [19, 475, 28, 5, 0, 1, 1],
# #                [0, 41, 401, 48, 4, 3, 0],
# #                [0, 3, 39, 598, 13, 3, 1],
# #                [0, 0, 9, 31, 196, 8, 0],
# #                [0, 0, 3, 1, 5, 298, 17],
# #                [0, 0, 1, 5, 0, 12, 485]])

# # EfficientNet 
# # cm = np.array([[475, 11, 1, 1, 0, 0, 1],
# #                [17, 481, 25, 3, 0, 1, 2],
# #                [0, 41, 383, 61, 9, 2, 1],
# #                [0, 1, 46, 594, 8, 4, 4],
# #                [0, 0, 5, 17, 216, 6, 0],
# #                [0, 0, 2, 2, 5, 301, 14],
# #                [0, 0, 0, 5, 0, 16, 482]])

# # AlexNet 
# # cm = np.array([[473, 12, 0, 0, 0, 0, 4],
# #                [21, 461, 39, 4, 1, 0, 3],
# #                [0, 37, 372, 61, 20, 5, 2],
# #                [0, 0, 64, 551, 34, 4, 4],
# #                [0, 0, 16, 62, 156, 10, 0],
# #                [0, 0, 4, 5, 9, 289, 17],
# #                [0, 2, 0, 5, 0, 21, 475]])

# # VGG 
# # cm = np.array([[470, 16, 0, 1, 0, 0, 2],
# #                [25, 453, 39, 5, 1, 2, 4],
# #                [0, 46, 357, 68, 14, 6, 6],
# #                [0, 2, 68, 560, 15, 5, 7],
# #                [0, 1, 21, 36, 171, 14, 1],
# #                [1, 2, 6, 5, 15, 277, 18],
# #                [1, 1, 4, 6, 1, 20, 470]])

# # ConvNext 
# cm = np.array([[473, 16, 0, 0, 0, 0, 0],
#                [19, 467, 33, 4, 1, 2, 3],
#                [0, 42, 349, 78, 18, 7, 3],
#                [0, 1, 57, 575, 19, 4, 1],
#                [0, 0, 13, 53, 159, 18, 1],
#                [1, 2, 2, 5, 4, 290, 20],
#                [4, 3, 0, 2, 3, 14, 477]])


# new_labels = ['1', '2', '3', '4', '5', '7', '8']

# plt.figure(figsize=(3.2, 2.9))

# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=new_labels, yticklabels=new_labels, square=True, cbar_kws={"shrink": 0.80})

# plt.xlabel('Predicted Label')
# plt.ylabel('True Label')

# # 添加下标(a)并加粗字体
# # plt.text(0.50, -0.15, '(a)', transform=plt.gca().transAxes, fontsize=10, fontweight='bold')


# # 调整图像边界，增加右侧边界的空间
# plt.subplots_adjust(left=0.15, right=0.95, top=0.99)


# # save_path = "D:/Users/Yang/Desktop/huatu2/hunxiaojuzhen/COResNetRS50.png"
# # save_path = "D:/Users/Yang/Desktop/huatu2/hunxiaojuzhen/ShuffleNet.png"
# # save_path = "D:/Users/Yang/Desktop/huatu2/hunxiaojuzhen/EfficientNet.png"
# # save_path = "D:/Users/Yang/Desktop/huatu2/hunxiaojuzhen/AlexNet.png"
# # save_path = "D:/Users/Yang/Desktop/huatu2/hunxiaojuzhen/VGG.png"
# save_path = "D:/Users/Yang/Desktop/huatu2/hunxiaojuzhen/ConvNext.png"
# plt.savefig(save_path, dpi=500)

# # plt.show()





# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns


# plt.rcParams['font.family'] = 'Times New Roman'
# plt.rcParams.update({'font.size': 10})


# ConvNext = np.array([[473, 16, 0, 0, 0, 0, 0],
#                [19, 467, 33, 4, 1, 2, 3],
#                [0, 42, 349, 78, 18, 7, 3],
#                [0, 1, 57, 575, 19, 4, 1],
#                [0, 0, 13, 53, 159, 18, 1],
#                [1, 2, 2, 5, 4, 290, 20],
#                [4, 3, 0, 2, 3, 14, 477]])

# ShuffleNet = np.array([[476, 13, 0, 0, 0, 0, 0],
#                [19, 475, 28, 5, 0, 1, 1],
#                [0, 41, 401, 48, 4, 3, 0],
#                [0, 3, 39, 598, 13, 3, 1],
#                [0, 0, 9, 31, 196, 8, 0],
#                [0, 0, 3, 1, 5, 298, 17],
#                [0, 0, 1, 5, 0, 12, 485]])

# EfficientNet = np.array([[475, 11, 1, 1, 0, 0, 1],
#                [17, 481, 25, 3, 0, 1, 2],
#                [0, 41, 383, 61, 9, 2, 1],
#                [0, 1, 46, 594, 8, 4, 4],
#                [0, 0, 5, 17, 216, 6, 0],
#                [0, 0, 2, 2, 5, 301, 14],
#                [0, 0, 0, 5, 0, 16, 482]])

# AlexNet = np.array([[473, 12, 0, 0, 0, 0, 4],
#                [21, 461, 39, 4, 1, 0, 3],
#                [0, 37, 372, 61, 20, 5, 2],
#                [0, 0, 64, 551, 34, 4, 4],
#                [0, 0, 16, 62, 156, 10, 0],
#                [0, 0, 4, 5, 9, 289, 17],
#                [0, 2, 0, 5, 0, 21, 475]])

# VGG = np.array([[470, 16, 0, 1, 0, 0, 2],
#                [25, 453, 39, 5, 1, 2, 4],
#                [0, 46, 357, 68, 14, 6, 6],
#                [0, 2, 68, 560, 15, 5, 7],
#                [0, 1, 21, 36, 171, 14, 1],
#                [1, 2, 6, 5, 15, 277, 18],
#                [1, 1, 4, 6, 1, 20, 470]])

# CO_ResNetRS50 = np.array([[484, 5, 0, 0, 0, 0, 0],
#                [19, 484, 24, 2, 0, 0, 0],
#                [0, 28, 417, 43, 7, 2, 0],
#                [0, 0, 37, 612, 3, 4, 1],
#                [0, 0, 6, 19, 212, 7, 0],
#                [0, 2, 1, 1, 3, 306, 11],
#                [0, 0, 0, 3, 1, 11, 488]])

# fig, axs = plt.subplots(3, 2, figsize=(7, 10))

# # 替换标签
# new_labels = ['1x', '2x', '3x', '4x', '5x', '7x', '8x']

# # 绘制模型1的混淆矩阵
# sns.heatmap(ConvNext, annot=True, fmt='d', cmap='Blues', xticklabels=new_labels, yticklabels=new_labels, ax=axs[0, 0])
# axs[0, 0].set_xlabel('Predicted Label')
# axs[0, 0].set_ylabel('True Label')
# # axs[0, 0].set_title('ConvNext-base')
# axs[0, 0].text(0.5, -0.3, "(a)", ha='center', va='center', transform=axs[0, 0].transAxes,
#                fontdict={'fontname': 'Times New Roman', 'fontsize': 10, 'weight': 'bold'})

# # 绘制模型2的混淆矩阵
# sns.heatmap(ShuffleNet, annot=True, fmt='d', cmap='Blues', xticklabels=new_labels, yticklabels=new_labels, ax=axs[0, 1])
# axs[0, 1].set_xlabel('Predicted Label')
# axs[0, 1].set_ylabel('True Label')
# # axs[0, 1].set_title('ShuffleNet-v2')
# axs[0, 1].text(0.5, -0.3, "(b)", ha='center', va='center', transform=axs[0, 1].transAxes,
#                fontdict={'fontname': 'Times New Roman', 'fontsize': 10, 'weight': 'bold'})

# # 绘制模型3的混淆矩阵
# sns.heatmap(EfficientNet, annot=True, fmt='d', cmap='Blues', xticklabels=new_labels, yticklabels=new_labels, ax=axs[1, 0])
# axs[1, 0].set_xlabel('Predicted Label')
# axs[1, 0].set_ylabel('True Label')
# # axs[1, 0].set_title('EfficientNet-b0')
# axs[1, 0].text(0.5, -0.3, "(c)", ha='center', va='center', transform=axs[1, 0].transAxes,
#                fontdict={'fontname': 'Times New Roman', 'fontsize': 10, 'weight': 'bold'})

# # 绘制模型4的混淆矩阵
# sns.heatmap(AlexNet, annot=True, fmt='d', cmap='Blues', xticklabels=new_labels, yticklabels=new_labels, ax=axs[1, 1])
# axs[1, 1].set_xlabel('Predicted Label')
# axs[1, 1].set_ylabel('True Label')
# # axs[1, 1].set_title('AlexNet')
# axs[1, 1].text(0.5, -0.3, "(d)", ha='center', va='center', transform=axs[1, 1].transAxes,
#                fontdict={'fontname': 'Times New Roman', 'fontsize': 10, 'weight': 'bold'})

# # 绘制模型5的混淆矩阵
# sns.heatmap(VGG, annot=True, fmt='d', cmap='Blues', xticklabels=new_labels, yticklabels=new_labels, ax=axs[2, 0])
# axs[2, 0].set_xlabel('Predicted Label')
# axs[2, 0].set_ylabel('True Label')
# # axs[2, 0].set_title('VGG-16')
# axs[2, 0].text(0.5, -0.3, "(e)", ha='center', va='center', transform=axs[2, 0].transAxes,
#                fontdict={'fontname': 'Times New Roman', 'fontsize': 10, 'weight': 'bold'})

# # 绘制模型6的混淆矩阵
# sns.heatmap(CO_ResNetRS50, annot=True, fmt='d', cmap='Blues', xticklabels=new_labels, yticklabels=new_labels, ax=axs[2, 1])
# axs[2, 1].set_xlabel('Predicted Label')
# axs[2, 1].set_ylabel('True Label')
# # axs[2, 1].set_title('CO-ResNetRS50')
# axs[2, 1].text(0.5, -0.3, "(f)", ha='center', va='center', transform=axs[2, 1].transAxes,
#                fontdict={'fontname': 'Times New Roman', 'fontsize': 10, 'weight': 'bold'})

# # 调整子图之间的间距
# plt.subplots_adjust(hspace=0.4, wspace=0.35)

# save_path = "D:/Desktop/huatu2/hunxiaojuzhen2.png"
# plt.savefig(save_path, dpi=300, bbox_inches='tight')

# plt.show()

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 混淆矩阵数据
ConvNext = np.array([[473, 16, 0, 0, 0, 0, 0],
               [19, 467, 33, 4, 1, 2, 3],
               [0, 42, 349, 78, 18, 7, 3],
               [0, 1, 57, 575, 19, 4, 1],
               [0, 0, 13, 53, 159, 18, 1],
               [1, 2, 2, 5, 4, 290, 20],
               [4, 3, 0, 2, 3, 14, 477]])

ShuffleNet = np.array([[476, 13, 0, 0, 0, 0, 0],
               [19, 475, 28, 5, 0, 1, 1],
               [0, 41, 401, 48, 4, 3, 0],
               [0, 3, 39, 598, 13, 3, 1],
               [0, 0, 9, 31, 196, 8, 0],
               [0, 0, 3, 1, 5, 298, 17],
               [0, 0, 1, 5, 0, 12, 485]])

EfficientNet = np.array([[475, 11, 1, 1, 0, 0, 1],
               [17, 481, 25, 3, 0, 1, 2],
               [0, 41, 383, 61, 9, 2, 1],
               [0, 1, 46, 594, 8, 4, 4],
               [0, 0, 5, 17, 216, 6, 0],
               [0, 0, 2, 2, 5, 301, 14],
               [0, 0, 0, 5, 0, 16, 482]])

AlexNet = np.array([[473, 12, 0, 0, 0, 0, 4],
               [21, 461, 39, 4, 1, 0, 3],
               [0, 37, 372, 61, 20, 5, 2],
               [0, 0, 64, 551, 34, 4, 4],
               [0, 0, 16, 62, 156, 10, 0],
               [0, 0, 4, 5, 9, 289, 17],
               [0, 2, 0, 5, 0, 21, 475]])

VGG = np.array([[470, 16, 0, 1, 0, 0, 2],
               [25, 453, 39, 5, 1, 2, 4],
               [0, 46, 357, 68, 14, 6, 6],
               [0, 2, 68, 560, 15, 5, 7],
               [0, 1, 21, 36, 171, 14, 1],
               [1, 2, 6, 5, 15, 277, 18],
               [1, 1, 4, 6, 1, 20, 470]])

CO_ResNetRS50 = np.array([[484, 5, 0, 0, 0, 0, 0],
               [19, 484, 24, 2, 0, 0, 0],
               [0, 28, 417, 43, 7, 2, 0],
               [0, 0, 37, 612, 3, 4, 1],
               [0, 0, 6, 19, 212, 7, 0],
               [0, 2, 1, 1, 3, 306, 11],
               [0, 0, 0, 3, 1, 11, 488]])

# 模型名称
model_names = ['ConvNext', 'ShuffleNet', 'EfficientNet', 'AlexNet', 'VGG', 'CO_ResNetRS50']
matrices = [ConvNext, ShuffleNet, EfficientNet, AlexNet, VGG, CO_ResNetRS50]

# 创建图形
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

for ax, matrix, name in zip(axes.flatten(), matrices, model_names):
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
    ax.set_title(name)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')

plt.tight_layout()
plt.show()