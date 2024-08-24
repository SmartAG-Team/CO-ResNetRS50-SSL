import matplotlib.pyplot as plt
import re

def extract_values(file_path, keyword):
    values = []
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
        for line in lines:
            if keyword in line:
                value = re.findall(rf'{keyword}: (\d+\.\d+)', line)
                if value:
                    values.append(float(value[0]))
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return values

# 文件路径
file_paths = {
    "CO-ResNetRS50-SSL": './reporting/ResNetRS_Coord_ODConv/best_model_7.txt',
    "AlexNet": './reporting/AlexNet/best_model_6.txt',
    "EfficientNet-b0": './reporting/EfficientNet/best_model_6.txt',
    "ConvNext-base": './reporting/ConvNext/best_model_6.txt',
    "ShuffleNetV2": './reporting/ShuffleNet/best_model_6.txt',
    "VGG16": './reporting/VGG/best_model_6.txt'
}

# 提取数据
train_acc_data = {label: extract_values(path, 'Train Acc') for label, path in file_paths.items()}
val_acc_data = {label: extract_values(path, 'Val Acc') for label, path in file_paths.items()}
train_loss_data = {label: extract_values(path, 'Train Loss') for label, path in file_paths.items()}
val_loss_data = {label: extract_values(path, 'Val Loss') for label, path in file_paths.items()}

# 设置字体
plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 10})

# 创建子图
fig, axs = plt.subplots(2, 2, figsize=(7, 5))  # 调整图的大小为(7, 5)

# 绘制Train Accuracy
for label, values in train_acc_data.items():
    epochs = range(1, len(values) + 1)
    axs[0, 0].plot(epochs, values, label=label, linewidth=1.0)
axs[0, 0].set_ylabel('Accuracy')
axs[0, 0].grid(True, alpha=0.3)
axs[0, 0].text(0.50, 0.50, 'Training', transform=axs[0, 0].transAxes, fontsize=10, ha='center', va='center')
axs[0, 0].text(0.04, 0.95, '(a)', transform=axs[0, 0].transAxes, fontsize=10, ha='left', va='top', fontweight='bold')
axs[0, 0].set_ylim(0.3, 1.0)
axs[0, 0].set_yticks([i/10 for i in range(3, 11)])


# 绘制Val Accuracy
for label, values in val_acc_data.items():
    epochs = range(1, len(values) + 1)
    axs[0, 1].plot(epochs, values, label=label, linewidth=1.0)
axs[0, 1].grid(True, alpha=0.3)
axs[0, 1].text(0.50, 0.50, 'Validation', transform=axs[0, 1].transAxes, fontsize=10, ha='center', va='center')
axs[0, 1].text(0.04, 0.95, '(b)', transform=axs[0, 1].transAxes, fontsize=10, ha='left', va='top', fontweight='bold')
axs[0, 1].set_ylim(0.3, 1.0)
axs[0, 1].set_yticks([i/10 for i in range(3, 11)])


# 绘制Train Loss
for label, values in train_loss_data.items():
    epochs = range(1, len(values) + 1)
    axs[1, 0].plot(epochs, values, label=label, linewidth=1.0)
axs[1, 0].set_ylabel('Loss')
axs[1, 0].set_xlabel('Epoch')
axs[1, 0].grid(True, alpha=0.3)
axs[1, 0].text(0.50, 0.50, 'Training', transform=axs[1, 0].transAxes, fontsize=10, ha='center', va='center')
axs[1, 0].text(0.04, 0.05, '(c)', transform=axs[1, 0].transAxes, fontsize=10, ha='left', va='bottom', fontweight='bold')
axs[1, 0].set_ylim(0.0, 2.5)

# 绘制Val Loss
for label, values in val_loss_data.items():
    epochs = range(1, len(values) + 1)
    axs[1, 1].plot(epochs, values, label=label, linewidth=1.0)
axs[1, 1].set_xlabel('Epoch')
axs[1, 1].grid(True, alpha=0.3)
axs[1, 1].text(0.50, 0.50, 'Validation', transform=axs[1, 1].transAxes, fontsize=10, ha='center', va='center')
axs[1, 1].text(0.04, 0.05, '(d)', transform=axs[1, 1].transAxes, fontsize=10, ha='left', va='bottom', fontweight='bold')
axs[1, 1].set_ylim(0.0, 2.5)

# 绘制图例
handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.10), ncol=3, columnspacing=9.0)


# 调整布局
plt.tight_layout(pad=0.4)
save_path = "D:/Desktop/huatu2/composite111111.png"
plt.savefig(save_path, dpi=600, bbox_inches='tight')  # 提高dpi以保持字体清晰
plt.show()





# import matplotlib.pyplot as plt
# import re

# def extract_values(file_path, keyword):
#     values = []
#     try:
#         with open(file_path, 'r') as file:
#             lines = file.readlines()
#         for line in lines:
#             if keyword in line:
#                 value = re.findall(rf'{keyword}: (\d+\.\d+)', line)
#                 if value:
#                     values.append(float(value[0]))
#     except Exception as e:
#         print(f"Error reading {file_path}: {e}")
#     return values

# # 文件路径
# file_paths = {
#     "ResNetRS50": './reporting/ResNetRS2/best_model_5.txt',
#     "ResNetRS50-SSL": './reporting/ResNetRS2/best_model_6.txt',
#     "C-ResNetRS50-SSL": './reporting/ResNetRS_Coord/best_model_6.txt',
#     "CO-ResNetRS50-SSL": './reporting/ResNetRS_Coord_ODConv/best_model_7.txt',
# }


# # 提取数据
# train_acc_data = {label: extract_values(path, 'Train Acc') for label, path in file_paths.items()}
# val_acc_data = {label: extract_values(path, 'Val Acc') for label, path in file_paths.items()}
# train_loss_data = {label: extract_values(path, 'Train Loss') for label, path in file_paths.items()}
# val_loss_data = {label: extract_values(path, 'Val Loss') for label, path in file_paths.items()}

# # 设置字体
# plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 10})

# # 创建子图
# fig, axs = plt.subplots(2, 2, figsize=(7, 5))  # 调整图的大小为(7, 5)

# # 绘制Train Accuracy
# for label, values in train_acc_data.items():
#     epochs = range(1, len(values) + 1)
#     axs[0, 0].plot(epochs, values, label=label, linewidth=1.0)
# axs[0, 0].set_ylabel('Accuracy')
# axs[0, 0].grid(True, alpha=0.3)
# axs[0, 0].text(0.50, 0.50, 'Training', transform=axs[0, 0].transAxes, fontsize=10, ha='center', va='center')
# axs[0, 0].text(0.04, 0.95, '(a)', transform=axs[0, 0].transAxes, fontsize=10, ha='left', va='top', fontweight='bold')
# axs[0, 0].set_ylim(0.5, 1.0)


# # 绘制Val Accuracy
# for label, values in val_acc_data.items():
#     epochs = range(1, len(values) + 1)
#     axs[0, 1].plot(epochs, values, label=label, linewidth=1.0)
# axs[0, 1].grid(True, alpha=0.3)
# axs[0, 1].text(0.50, 0.50, 'Validation', transform=axs[0, 1].transAxes, fontsize=10, ha='center', va='center')
# axs[0, 1].text(0.04, 0.95, '(b)', transform=axs[0, 1].transAxes, fontsize=10, ha='left', va='top', fontweight='bold')
# axs[0, 1].set_ylim(0.5, 1.0)


# # 绘制Train Loss
# for label, values in train_loss_data.items():
#     epochs = range(1, len(values) + 1)
#     axs[1, 0].plot(epochs, values, label=label, linewidth=1.0)
# axs[1, 0].set_ylabel('Loss')
# axs[1, 0].set_xlabel('Epoch')
# axs[1, 0].grid(True, alpha=0.3)
# axs[1, 0].text(0.50, 0.50, 'Training', transform=axs[1, 0].transAxes, fontsize=10, ha='center', va='center')
# axs[1, 0].text(0.04, 0.05, '(c)', transform=axs[1, 0].transAxes, fontsize=10, ha='left', va='bottom', fontweight='bold')
# axs[1, 0].set_ylim(0.0, 1.25)

# # 绘制Val Loss
# for label, values in val_loss_data.items():
#     epochs = range(1, len(values) + 1)
#     axs[1, 1].plot(epochs, values, label=label, linewidth=1.0)
# axs[1, 1].set_xlabel('Epoch')
# axs[1, 1].grid(True, alpha=0.3)
# axs[1, 1].text(0.50, 0.50, 'Validation', transform=axs[1, 1].transAxes, fontsize=10, ha='center', va='center')
# axs[1, 1].text(0.04, 0.05, '(d)', transform=axs[1, 1].transAxes, fontsize=10, ha='left', va='bottom', fontweight='bold')
# axs[1, 1].set_ylim(0.0, 1.25)

# # 绘制图例
# handles, labels = axs[0, 0].get_legend_handles_labels()
# fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.06), ncol=len(labels))


# # 调整布局
# plt.tight_layout(pad=0.4)
# save_path = "D:/Desktop/huatu2/composite.png"
# plt.savefig(save_path, dpi=600, bbox_inches='tight')  # 提高dpi以保持字体清晰
# plt.show()