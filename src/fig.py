import matplotlib.pyplot as plt
import re
import seaborn as sns
import numpy as np

# 方法一：画半监督置信度阈值的图
def fig1():
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
        "ResNetRS50": './data/ResNetRS50/yuanshi/result.txt',
        "ResNetRS50-SSL-90": './data/ResNetRS50/ssl_90/result.txt',
        "ResNetRS50-SSL-85": './data/ResNetRS50/ssl_85/result.txt',
        "ResNetRS50-SSL-925": './data/ResNetRS50/ssl_925/result.txt',
        "ResNetRS50-SSL-875": './data/ResNetRS50/ssl_875/result.txt',
        "ResNetRS50-SSL-95": './data/ResNetRS50/ssl_95/result.txt',
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
    axs[0, 0].set_ylim(0.5, 1.0)
    axs[0, 0].set_yticks([i/10 for i in range(5, 11)])


    # 绘制Val Accuracy
    for label, values in val_acc_data.items():
        epochs = range(1, len(values) + 1)
        axs[0, 1].plot(epochs, values, label=label, linewidth=1.0)
    axs[0, 1].grid(True, alpha=0.3)
    axs[0, 1].text(0.50, 0.50, 'Validation', transform=axs[0, 1].transAxes, fontsize=10, ha='center', va='center')
    axs[0, 1].text(0.04, 0.95, '(b)', transform=axs[0, 1].transAxes, fontsize=10, ha='left', va='top', fontweight='bold')
    axs[0, 1].set_ylim(0.5, 1.0)
    axs[0, 1].set_yticks([i/10 for i in range(5, 11)])


    # 绘制Train Loss
    for label, values in train_loss_data.items():
        epochs = range(1, len(values) + 1)
        axs[1, 0].plot(epochs, values, label=label, linewidth=1.0)
    axs[1, 0].set_ylabel('Loss')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].grid(True, alpha=0.3)
    axs[1, 0].text(0.50, 0.50, 'Training', transform=axs[1, 0].transAxes, fontsize=10, ha='center', va='center')
    axs[1, 0].text(0.04, 0.05, '(c)', transform=axs[1, 0].transAxes, fontsize=10, ha='left', va='bottom', fontweight='bold')
    axs[1, 0].set_ylim(0.0, 2.0)

    # 绘制Val Loss
    for label, values in val_loss_data.items():
        epochs = range(1, len(values) + 1)
        axs[1, 1].plot(epochs, values, label=label, linewidth=1.0)
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].grid(True, alpha=0.3)
    axs[1, 1].text(0.50, 0.50, 'Validation', transform=axs[1, 1].transAxes, fontsize=10, ha='center', va='center')
    axs[1, 1].text(0.04, 0.05, '(d)', transform=axs[1, 1].transAxes, fontsize=10, ha='left', va='bottom', fontweight='bold')
    axs[1, 1].set_ylim(0.0, 2.0)

    # 绘制图例
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.10), ncol=3, columnspacing=7.0)


    # 调整布局
    plt.tight_layout(pad=0.4)
    save_path = "D:/Desktop/huatu2/SSLConfidenceThreshold.png"
    plt.savefig(save_path, dpi=600, bbox_inches='tight')  # 提高dpi以保持字体清晰
    plt.show()

# 方法二：画添加图片的数量的图
def fig2():
    def extract_values(file_path, keyword):
        values = []
        try:
            with open(file_path, 'r') as file:
                lines = file.readlines()
            for line in lines:
                if keyword in line:
                    value = re.findall(rf'{keyword}: ([\d\.]+)', line)
                    if value:
                        values.append(float(value[0]))
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
        return values

    # 文件路径
    file_paths = {
        "ResNetRS50-SSL-85": './data/ResNetRS50/ssl_85/result.txt',
        "ResNetRS50-SSL-925": './data/ResNetRS50/ssl_925/result.txt',
        "ResNetRS50-SSL-875": './data/ResNetRS50/ssl_875/result.txt',
        "ResNetRS50-SSL-95": './data/ResNetRS50/ssl_95/result.txt',
        "ResNetRS50-SSL-90": './data/ResNetRS50/ssl_90/result.txt',
    }

    # 提取数据
    total_difference = {label: extract_values(path, 'Total Difference') for label, path in file_paths.items()}

    # 设置字体
    plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 10})

    # 创建子图
    fig, ax = plt.subplots(figsize=(7, 5))  # 只创建一个子图

    # 绘制total_difference
    for label, values in total_difference.items():
        epochs = range(2, len(values) + 2)
        ax.plot(epochs, values, label=label, linewidth=1.0)
    ax.set_ylabel('Number')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 7000)
    yticks = range(0, 7001, 500)  # 每隔1000设置一个刻度
    ax.set_yticks(yticks)
    ax.set_xlabel('Epoch')  # 添加x轴标签

    # 绘制图例
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.10), ncol=3, columnspacing=7.0)

    # 调整布局
    plt.tight_layout(pad=0.4)
    save_path = "D:/Desktop/huatu2/NumberOfImages.png"
    plt.savefig(save_path, dpi=600, bbox_inches='tight')  # 提高dpi以保持字体清晰
    plt.show()

# 方法三：画数据增强的图
def fig3():
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
        "ResNetRS50": './data/ResNetRS50/yuanshi/result.txt',
        "Aug-SSL": './data/ResNetRS50/aug_ssl_90/result.txt',
        "ResNetRS50-Aug": './data/ResNetRS50/yuanshi_aug/result.txt',
        "All-Aug-SSL": './data/ResNetRS50/all_aug_ssl_90/result.txt',
        "ResNetRS50-SSL": './data/ResNetRS50/ssl_90/result.txt',
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
    axs[0, 0].set_ylim(0.5, 1.0)
    axs[0, 0].set_yticks([i/10 for i in range(5, 11)])


    # 绘制Val Accuracy
    for label, values in val_acc_data.items():
        epochs = range(1, len(values) + 1)
        axs[0, 1].plot(epochs, values, label=label, linewidth=1.0)
    axs[0, 1].grid(True, alpha=0.3)
    axs[0, 1].text(0.50, 0.50, 'Validation', transform=axs[0, 1].transAxes, fontsize=10, ha='center', va='center')
    axs[0, 1].text(0.04, 0.95, '(b)', transform=axs[0, 1].transAxes, fontsize=10, ha='left', va='top', fontweight='bold')
    axs[0, 1].set_ylim(0.5, 1.0)
    axs[0, 1].set_yticks([i/10 for i in range(5, 11)])


    # 绘制Train Loss
    for label, values in train_loss_data.items():
        epochs = range(1, len(values) + 1)
        axs[1, 0].plot(epochs, values, label=label, linewidth=1.0)
    axs[1, 0].set_ylabel('Loss')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].grid(True, alpha=0.3)
    axs[1, 0].text(0.50, 0.50, 'Training', transform=axs[1, 0].transAxes, fontsize=10, ha='center', va='center')
    axs[1, 0].text(0.04, 0.05, '(c)', transform=axs[1, 0].transAxes, fontsize=10, ha='left', va='bottom', fontweight='bold')
    axs[1, 0].set_ylim(0.0, 2.0)

    # 绘制Val Loss
    for label, values in val_loss_data.items():
        epochs = range(1, len(values) + 1)
        axs[1, 1].plot(epochs, values, label=label, linewidth=1.0)
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].grid(True, alpha=0.3)
    axs[1, 1].text(0.50, 0.50, 'Validation', transform=axs[1, 1].transAxes, fontsize=10, ha='center', va='center')
    axs[1, 1].text(0.04, 0.05, '(d)', transform=axs[1, 1].transAxes, fontsize=10, ha='left', va='bottom', fontweight='bold')
    axs[1, 1].set_ylim(0.0, 2.0)

    # 绘制图例
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.10), ncol=3, columnspacing=10.0)


    # 调整布局
    plt.tight_layout(pad=0.4)
    save_path = "D:/Desktop/huatu2/DataAugmentation.png"
    plt.savefig(save_path, dpi=600, bbox_inches='tight')  # 提高dpi以保持字体清晰
    plt.show()

# 方法四：画消融实验的图
def fig4():
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
        "ResNetRS50": './data/ResNetRS50/yuanshi/result.txt',
        "ResNetRS50-SSL": './data/ResNetRS50/ssl_90/result.txt',
        "C-ResNetRS50-SSL": './data/C_ResNetRS50/result.txt',
        "CO-ResNetRS50-SSL": './data/CO_ResNetRS50/result.txt',
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
    axs[0, 0].set_ylim(0.5, 1.0)
    axs[0, 0].set_yticks([i/10 for i in range(5, 11)])


    # 绘制Val Accuracy
    for label, values in val_acc_data.items():
        epochs = range(1, len(values) + 1)
        axs[0, 1].plot(epochs, values, label=label, linewidth=1.0)
    axs[0, 1].grid(True, alpha=0.3)
    axs[0, 1].text(0.50, 0.50, 'Validation', transform=axs[0, 1].transAxes, fontsize=10, ha='center', va='center')
    axs[0, 1].text(0.04, 0.95, '(b)', transform=axs[0, 1].transAxes, fontsize=10, ha='left', va='top', fontweight='bold')
    axs[0, 1].set_ylim(0.5, 1.0)
    axs[0, 1].set_yticks([i/10 for i in range(5, 11)])


    # 绘制Train Loss
    for label, values in train_loss_data.items():
        epochs = range(1, len(values) + 1)
        axs[1, 0].plot(epochs, values, label=label, linewidth=1.0)
    axs[1, 0].set_ylabel('Loss')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].grid(True, alpha=0.3)
    axs[1, 0].text(0.50, 0.50, 'Training', transform=axs[1, 0].transAxes, fontsize=10, ha='center', va='center')
    axs[1, 0].text(0.04, 0.05, '(c)', transform=axs[1, 0].transAxes, fontsize=10, ha='left', va='bottom', fontweight='bold')
    axs[1, 0].set_ylim(0.0, 2.0)

    # 绘制Val Loss
    for label, values in val_loss_data.items():
        epochs = range(1, len(values) + 1)
        axs[1, 1].plot(epochs, values, label=label, linewidth=1.0)
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].grid(True, alpha=0.3)
    axs[1, 1].text(0.50, 0.50, 'Validation', transform=axs[1, 1].transAxes, fontsize=10, ha='center', va='center')
    axs[1, 1].text(0.04, 0.05, '(d)', transform=axs[1, 1].transAxes, fontsize=10, ha='left', va='bottom', fontweight='bold')
    axs[1, 1].set_ylim(0.0, 2.0)

    # 绘制图例
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.07), ncol=4, columnspacing=2.7)


    # 调整布局
    plt.tight_layout(pad=0.4)
    save_path = "D:/Desktop/huatu2/AblationExperiment.png"
    plt.savefig(save_path, dpi=600, bbox_inches='tight')  # 提高dpi以保持字体清晰
    plt.show()

# 方法五：画对比实验的图
def fig5():
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
        "EfficientNet-b0": './data/EfficientNet_b0/result.txt',
        "ShuffleNetV2": './data/ShuffleNetV2/result.txt',
        "AlexNet": './data/AlexNet/result.txt',
        "ConvNext-base": './data/ConvNext/result.txt',
        "VGG16": './data/VGG16/result.txt',
        "CO-ResNetRS50-SSL": './data/CO_ResNetRS50/result.txt',
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
    axs[1, 0].set_ylim(0.0, 2.7)

    # 绘制Val Loss
    for label, values in val_loss_data.items():
        epochs = range(1, len(values) + 1)
        axs[1, 1].plot(epochs, values, label=label, linewidth=1.0)
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].grid(True, alpha=0.3)
    axs[1, 1].text(0.50, 0.50, 'Validation', transform=axs[1, 1].transAxes, fontsize=10, ha='center', va='center')
    axs[1, 1].text(0.04, 0.05, '(d)', transform=axs[1, 1].transAxes, fontsize=10, ha='left', va='bottom', fontweight='bold')
    axs[1, 1].set_ylim(0.0, 2.7)

    # 绘制图例
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.10), ncol=3, columnspacing=9.8)


    # 调整布局
    plt.tight_layout(pad=0.4)
    save_path = "D:/Desktop/huatu2/ComparisonExperiment.png"
    plt.savefig(save_path, dpi=600, bbox_inches='tight')  # 提高dpi以保持字体清晰
    plt.show()

# 方法六：画条形图
def fig6():
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams.update({'font.size': 10})

    # 数据
    categories = ['1x', '2x', '3x', '4x', '5x', '7x', '8x']  # 类别

    ShuffleNet = [95.36, 87.92, 77.47, 89.06, 65.12, 88.70, 93.02]   # 第一组数据
    VGG16 = [93.91, 81.23, 69.75, 83.15, 61.05, 84.52, 89.83]        # 第二组数据
    EfficientNet = [96.52, 89.72, 83.64, 89.06, 79.07, 91.63, 95.35]
    ConvNext = [96.52, 86.63, 80.56, 83.59, 65.70, 87.03, 94.19]
    AlexNet = [94.20, 85.35, 71.60, 80.31, 63.37, 85.36, 90.41]
    CO_ResNetRS50 = [96.52, 88.43, 80.86, 91.90, 84.30, 92.89, 94.19]

    # 设置条形图的宽度
    bar_width = 0.13

    # 创建图表并设置尺寸
    plt.figure(figsize=(7, 4))

    # 创建图表
    plt.bar(categories, ShuffleNet, width=bar_width, color='#EC6E66', label='ShuffleNetV2')
    plt.bar([x + bar_width for x in range(len(categories))], VGG16, width=bar_width, color='#91CCC0', label='VGG16')
    plt.bar([x + bar_width*2 for x in range(len(categories))], EfficientNet, width=bar_width, color='#F7AC53', label='EfficientNet-b0')
    plt.bar([x + bar_width*3 for x in range(len(categories))], ConvNext, width=bar_width, color='#B5CE4E', label='ConvNext-base')
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

    save_path = "D:/Desktop/huatu2/BarChart.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    # 显示图表
    plt.show()

# 方法七：画混淆矩阵图
def fig7():
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams.update({'font.size': 10})


    ConvNext = np.array([[333, 11, 0, 1, 0, 0, 0],
                        [19, 337, 30, 2, 0, 0, 1],
                        [0, 18, 261, 37, 7, 1, 0],
                        [0, 3, 44, 382, 21, 5, 2],
                        [0, 2, 7, 42, 113, 7, 1],
                        [0, 1, 3, 4, 9, 208, 14],
                        [0, 2, 2, 4, 0, 12, 324]])

    ShuffleNet = np.array([[329, 13, 0, 0, 0, 2, 1],
                        [11, 342, 29, 2, 0, 3, 2],
                        [0, 17, 251, 47, 5, 4, 0],
                        [0, 3, 37, 407, 6, 1, 3],
                        [0, 2, 13, 37, 112, 7, 1],
                        [1, 2, 5, 3, 4, 212, 12],
                        [1, 1, 2, 3, 0, 17, 320]])

    EfficientNet = np.array([[333, 10, 1, 0, 0, 0, 1],
                            [13, 349, 24, 1, 0, 2, 0],
                            [0, 20, 271, 28, 4, 1, 0],
                            [0, 6, 33, 407, 9, 0, 2],
                            [0, 0, 5, 23, 136, 8, 0],
                            [1, 0, 2, 1, 6, 219, 10],
                            [0, 1, 3, 3, 0, 9, 328]])

    AlexNet = np.array([[325, 17, 1, 0, 0, 1, 1],
                        [15, 332, 34, 2, 0, 1, 5],
                        [0, 24, 232, 48, 12, 7, 1],
                        [0, 1, 43, 367, 34, 8, 4],
                        [0, 1, 17, 39, 109, 6, 0],
                        [0, 4, 5, 3, 9, 204, 14],
                        [1, 3, 7, 4, 1, 17, 331]])

    VGG = np.array([[324, 19, 1, 0, 0, 0, 1],
                    [18, 316, 43, 5, 0, 2, 5],
                    [1, 18, 226, 73, 6, 0, 0],
                    [0, 4, 42, 380, 21, 4, 6],
                    [0, 2, 17, 38,105, 8, 2],
                    [2, 2, 6, 5, 7, 202, 15],
                    [1, 6, 1, 10, 1, 16, 309]])

    CO_ResNetRS50 = np.array([[333, 10, 1, 0, 0, 0, 1],
                            [19, 344, 24, 0, 0, 1, 1],
                            [0, 17, 262, 39, 6, 0, 0],
                            [0, 2, 27, 420, 5, 1, 2],
                            [0, 0, 7, 16, 145, 3, 1],
                            [0, 1, 4, 1, 3, 222, 8],
                            [0, 1, 3, 3, 0, 13, 324]])

    fig, axs = plt.subplots(3, 2, figsize=(7, 10))

    vmin, vmax = 0, 420  # 根据你的数据范围进行调整

    # 替换标签
    new_labels = ['1x', '2x', '3x', '4x', '5x', '7x', '8x']

    # 绘制模型1的混淆矩阵
    sns.heatmap(ConvNext, annot=True, fmt='d', cmap='Blues', xticklabels=new_labels, yticklabels=new_labels, ax=axs[0, 0], vmin=vmin, vmax=vmax)
    axs[0, 0].set_xlabel('Predicted Label')
    axs[0, 0].set_ylabel('True Label')
    # axs[0, 0].set_title('ConvNext-base')
    axs[0, 0].text(0.5, -0.3, "(a)", ha='center', va='center', transform=axs[0, 0].transAxes,
                fontdict={'fontname': 'Times New Roman', 'fontsize': 10, 'weight': 'bold'})

    # 绘制模型2的混淆矩阵
    sns.heatmap(ShuffleNet, annot=True, fmt='d', cmap='Blues', xticklabels=new_labels, yticklabels=new_labels, ax=axs[0, 1], vmin=vmin, vmax=vmax)
    axs[0, 1].set_xlabel('Predicted Label')
    axs[0, 1].set_ylabel('True Label')
    # axs[0, 1].set_title('ShuffleNet-v2')
    axs[0, 1].text(0.5, -0.3, "(b)", ha='center', va='center', transform=axs[0, 1].transAxes,
                fontdict={'fontname': 'Times New Roman', 'fontsize': 10, 'weight': 'bold'})

    # 绘制模型3的混淆矩阵
    sns.heatmap(EfficientNet, annot=True, fmt='d', cmap='Blues', xticklabels=new_labels, yticklabels=new_labels, ax=axs[1, 0], vmin=vmin, vmax=vmax)
    axs[1, 0].set_xlabel('Predicted Label')
    axs[1, 0].set_ylabel('True Label')
    # axs[1, 0].set_title('EfficientNet-b0')
    axs[1, 0].text(0.5, -0.3, "(c)", ha='center', va='center', transform=axs[1, 0].transAxes,
                fontdict={'fontname': 'Times New Roman', 'fontsize': 10, 'weight': 'bold'})

    # 绘制模型4的混淆矩阵
    sns.heatmap(AlexNet, annot=True, fmt='d', cmap='Blues', xticklabels=new_labels, yticklabels=new_labels, ax=axs[1, 1], vmin=vmin, vmax=vmax)
    axs[1, 1].set_xlabel('Predicted Label')
    axs[1, 1].set_ylabel('True Label')
    # axs[1, 1].set_title('AlexNet')
    axs[1, 1].text(0.5, -0.3, "(d)", ha='center', va='center', transform=axs[1, 1].transAxes,
                fontdict={'fontname': 'Times New Roman', 'fontsize': 10, 'weight': 'bold'})

    # 绘制模型5的混淆矩阵
    sns.heatmap(VGG, annot=True, fmt='d', cmap='Blues', xticklabels=new_labels, yticklabels=new_labels, ax=axs[2, 0], vmin=vmin, vmax=vmax)
    axs[2, 0].set_xlabel('Predicted Label')
    axs[2, 0].set_ylabel('True Label')
    # axs[2, 0].set_title('VGG-16')
    axs[2, 0].text(0.5, -0.3, "(e)", ha='center', va='center', transform=axs[2, 0].transAxes,
                fontdict={'fontname': 'Times New Roman', 'fontsize': 10, 'weight': 'bold'})

    # 绘制模型6的混淆矩阵
    sns.heatmap(CO_ResNetRS50, annot=True, fmt='d', cmap='Blues', xticklabels=new_labels, yticklabels=new_labels, ax=axs[2, 1], vmin=vmin, vmax=vmax)
    axs[2, 1].set_xlabel('Predicted Label')
    axs[2, 1].set_ylabel('True Label')
    # axs[2, 1].set_title('CO-ResNetRS50')
    axs[2, 1].text(0.5, -0.3, "(f)", ha='center', va='center', transform=axs[2, 1].transAxes,
                fontdict={'fontname': 'Times New Roman', 'fontsize': 10, 'weight': 'bold'})

    # 调整子图之间的间距
    plt.subplots_adjust(hspace=0.4, wspace=0.35)

    save_path = "D:/Desktop/huatu2/ConfusionMatrix.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()

# 主函数
def main():
    # 调用不同的方法
    fig4()

# 运行主函数
if __name__ == "__main__":
    main()