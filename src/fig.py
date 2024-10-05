import matplotlib.pyplot as plt
import re
import seaborn as sns
import numpy as np
from matplotlib import font_manager

# 生育期和对应的图片数量
def fig1():
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams.update({'font.size': 10})

    # 数据
    categories = ['1', '2', '3', '4', '5', '7', '8']  # 类别


    traing_dataset = [1610, 1812, 1515, 2131, 800, 1118, 1607]
    validation_dataset = [345, 388, 325, 457, 171, 240, 344]
    test_dataset = [345, 389, 324, 457, 172, 239, 344]

    # 设置条形图的宽度
    bar_width = 0.40
    spacing = 1.3  # 通过调整这个值增大类别之间的间距

    # 创建图表并设置尺寸
    plt.figure(figsize=(7, 4))  # 适当增大图表的宽度

    # 创建图表
    bars1 = plt.bar([x * spacing for x in range(len(categories))], traing_dataset, width=bar_width, color='#EC6E66', label='Training', zorder=2)
    bars2 = plt.bar([x * spacing + bar_width for x in range(len(categories))], validation_dataset, width=bar_width, color='#91CCC0', label='Validation', zorder=2)
    bars3 = plt.bar([x * spacing + bar_width*2 for x in range(len(categories))], test_dataset, width=bar_width, color='#F7AC53', label='Test', zorder=2)


    # 在每个条形上添加数值
    def add_values_to_bars(bars, values):
        for bar, value in zip(bars, values):
            offset = 30 if value == 2131 else 0  # 只对2131进行调整
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() - offset, f'{value}', 
                    ha='center', va='bottom', fontsize=10)

    add_values_to_bars(bars1, traing_dataset)
    add_values_to_bars(bars2, validation_dataset)
    add_values_to_bars(bars3, test_dataset)


    # 添加标题和标签
    # plt.xlabel('Principal growth stages (BBCH)')
    plt.xlabel('Principal BBCH Code')
    plt.ylabel('Number of images')

    # 调整X轴标签的位置，使其居中，并增加间距
    # plt.xticks([x * spacing + bar_width*1.5 for x in range(len(categories))], categories)
    plt.xticks([x * spacing + bar_width for x in range(len(categories))], categories)

    # 设置Y轴范围从75%开始
    plt.ylim(150, 2200)
    # 设置Y轴刻度间隔为5
    # plt.yticks(range(77, 99, 2))
    plt.grid(True, zorder=1)

    # 添加图例
    plt.legend(bbox_to_anchor=(0.5, 1.15), loc='upper center', ncol=3, columnspacing=7.0, frameon=False)

    # 调整图的位置，使其向上移动
    plt.subplots_adjust(bottom=0.18)
    

    # 保存图片
    save_path = "D:/Desktop/huatu2/GrowthStageImageCount.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    # 显示图表
    plt.show()

# 消融实验
def fig2():
        # 设置字体
    plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 10})

    # 模型名称
    models = ['ResNetRS50', 'ResNetRS50-SSL', 'C-ResNetRS50-SSL', 'CO-ResNetRS50-SSL']

    # 各个模型的性能指标数据
    metrics = ['Accuracy', 'Recall', 'Precision', 'F1 score', 'Parameter']
    performance = np.array([
        [88.59, 89.38, 89.91, 90.31],  # Accuracy
        [88.29, 88.73, 89.23, 89.87],  # Recall
        [88.71, 89.73, 90.34, 90.53],  # Precision
        [88.45, 89.17, 89.71, 90.17]   # F1 score
    ])

    # 模型的参数量 (单位: 百万)
    parameters = [48.19, 48.19, 50.11, 65.38]

    # 颜色设置
    colors = ['#EC6E66', '#91CCC0', '#F7AC53', '#B5CE4E']

    # 设置柱形图的位置和宽度
    x = np.arange(len(metrics))  # 指标的标签位置
    width = 0.2  # 每个柱子的宽度

    # 创建图形
    fig, ax1 = plt.subplots(figsize=(7, 5))

    # 绘制性能指标条形图，分别为每个模型绘制
    for i in range(len(models)):
        ax1.bar(x[:-1] + i * width - width * 1.5, performance[:, i], width, label=models[i], color=colors[i], zorder=2)

    # 添加数值标签
    for i in range(len(metrics) - 1):
        for j in range(len(models)):
            ax1.text(x[i] + j * width - width * 1.5, performance[i, j] + 0.1, f'{performance[i, j]:.2f}', 
                    ha='center', va='bottom', fontsize=10)

    # 设置左侧y轴标签和范围
    ax1.set_ylabel('Values of Accuracy, Recall, Precision, and F1 score. (%)')
    ax1.set_ylim(87, 92)

    # 设置x轴标签
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)

    # 设置右侧y轴，用于显示参数量
    ax2 = ax1.twinx()
    ax2.set_ylabel('Values of Parameters. (M)')  # 修改右侧y轴的颜色
    ax2.tick_params(axis='y', rotation=45)    # 让右侧y轴的标签也为蓝色
    ax2.set_ylim(10, 90)

    # 绘制参数量的条形图
    for i in range(len(models)):
        ax2.bar(x[-1] + i * width - width * 1.5, parameters[i], width, label=models[i], color=colors[i])

    # 添加参数量数值标签，并设置对应的颜色
    for i in range(len(models)):
        ax2.text(x[-1] + i * width - width * 1.5, parameters[i] + 0.5, f'{parameters[i]:.2f}', 
                ha='center', va='bottom', fontsize=10, rotation=45)

    # 显示图例
    ax1.legend(loc='upper left')

    # 显示网格线
    ax1.grid(axis='y', alpha=0.3, zorder=1)

    # 保存图像
    save_path = "D:/Desktop/huatu2/Ablation_Experiments.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')  # 提高dpi以保持字体清晰

    # 显示图形
    plt.show()

# 半监督置信度阈值的图
def fig3():
    # 设置字体
    plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 10})
    # Data for each model
    labels = ['Accuracy', 'Recall', 'Precision', 'F1score']

    ResNetRS50 = [88.59, 88.29, 88.71, 88.45]
    ResNetRS50_SSL_85 = [89.30, 88.50, 89.45, 88.93]
    ResNetRS50_SSL_875 = [88.28, 87.23, 88.91, 87.94]
    ResNetRS50_SSL_90 = [89.38, 88.73, 89.73, 89.17]
    ResNetRS50_SSL_925 = [89.34, 88.50, 89.95, 89.12]
    ResNetRS50_SSL_95 = [89.12, 88.25, 89.40, 88.77]

    # X axis for confidence thresholds
    confidence_thresholds = ['Baseline', '0.85', '0.875', '0.90', '0.925', '0.95']

    # Plot data for each metric
    plt.figure(figsize=(7, 5))

    # Plot lines for each metric
    for i, label in enumerate(labels):
        metric_values = [ResNetRS50[i], ResNetRS50_SSL_85[i], ResNetRS50_SSL_875[i], ResNetRS50_SSL_90[i], ResNetRS50_SSL_925[i], ResNetRS50_SSL_95[i]]
        plt.plot(confidence_thresholds, metric_values, marker='o', label=label)
        # Annotate each point with its value
        for j, value in enumerate(metric_values):
            plt.text(confidence_thresholds[j], value + 0.04, f'{value:.2f}', ha='center', va='bottom', fontsize=10)
        
    

    # Add labels and title
    plt.xlabel('Confidence Threshold')
    plt.ylabel('Metric Value (%)')
    plt.legend()
    plt.grid(True)

    # Show plot
    plt.tight_layout(pad=0.4)
    save_path = "D:/Desktop/huatu2/Confidence_Threshold.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')  # 提高dpi以保持字体清晰
    plt.show()

# 添加图片的数量
def fig4():
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
    ax.set_ylabel('Number of pseudo-labeled images added')
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

# 数据增强的图
def fig5():

    # 设置字体
    plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 10})

    # 模型名称
    models = ['ResNetRS50', 'ResNetRS50-Aug', 'ResNetRS50-SSL', 'ResNetRS50-SSL-LabelAug', 'ResNetRS50-SSL-AllAug']

    # 各个模型的指标数据
    metrics = ['Accuracy', 'Recall', 'Precision', 'F1 score']
    performance = np.array([
        [88.59, 88.94, 89.38, 89.12, 89.30],  # Accuracy
        [88.29, 88.11, 88.73, 87.90, 88.92],  # Recall
        [88.71, 89.28, 89.73, 89.29, 89.55],  # Precision
        [88.45, 88.63, 89.17, 88.50, 89.19]  # F1 score
    ])

    # 颜色设置
    colors = ['#EC6E66', '#91CCC0', '#F7AC53', '#B5CE4E', '#BD7795']

    # 设置柱形图的位置和宽度
    x = np.arange(len(metrics))  # 指标的标签位置
    width = 0.18  # 每个柱子的宽度

    # 创建图形
    plt.figure(figsize=(7, 5))

    # 绘制条形图，分别为每个模型绘制，并设置不同颜色
    for i in range(len(models)):
        plt.bar(x + i * width - width * (len(models) - 1) / 2, performance[:, i], width, label=models[i], color=colors[i], zorder=2)

    # 添加数值标签
    for i in range(len(metrics)):
        for j in range(len(models)):
            plt.text(x[i] + j * width - width * (len(models) - 1) / 2, performance[i, j] + 0.05, 
                    f'{performance[i, j]:.2f}', ha='center', va='bottom', fontsize=10)

    # 设置标签和标题
    plt.ylabel('Values (%)')
    plt.xticks(x, metrics)
    plt.ylim(87.50, 90)  # 根据数据范围设置Y轴

    plt.xlabel('Metrics')

    # 图例
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.16), ncol=3, handletextpad=0.5, columnspacing=0.6)

    # 显示网格线
    plt.grid(axis='y', alpha=0.3, zorder=1)

    # 保存图像
    save_path = "D:/Desktop/huatu2/AugmentationComparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')  # 提高dpi以保持字体清晰

    # 显示图形
    plt.show()

# 具体生育期的条形图
def fig6():
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams.update({'font.size': 10})

    # 数据
    categories = ['1', '2', '3', '4', '5', '7', '8']  # 类别

    ResNetRS50 = [95.65, 86.63, 81.79, 86.65, 80.81, 92.89, 93.60]   # 第一组数据
    ResNetRS50_SSL = [95.07, 89.97, 79.94, 89.72, 79.65, 92.89, 93.90]        # 第二组数据
    C_ResNetRS50_SSL = [95.65, 91.52, 78.09, 91.90, 81.40, 93.31, 92.73]
    CO_ResNetRS50_SSL = [96.52, 88.43, 80.86, 91.90, 84.30, 92.89, 94.19]

    # 设置条形图的宽度
    bar_width = 0.30
    spacing = 1.3  # 通过调整这个值增大类别之间的间距

    # 创建图表并设置尺寸
    plt.figure(figsize=(7, 4))  # 适当增大图表的宽度

    # 创建图表
    bars1 = plt.bar([x * spacing for x in range(len(categories))], ResNetRS50, width=bar_width, color='#EC6E66', label='ResNetRS50', zorder=2)
    bars2 = plt.bar([x * spacing + bar_width for x in range(len(categories))], ResNetRS50_SSL, width=bar_width, color='#91CCC0', label='ResNetRS50-SSL', zorder=2)
    bars3 = plt.bar([x * spacing + bar_width*2 for x in range(len(categories))], C_ResNetRS50_SSL, width=bar_width, color='#F7AC53', label='C-ResNetRS50-SSL', zorder=2)
    bars4 = plt.bar([x * spacing + bar_width*3 for x in range(len(categories))], CO_ResNetRS50_SSL, width=bar_width, color='#B5CE4E', label='CO-ResNetRS50-SSL', zorder=2)

    # 添加标题和标签
    # plt.xlabel('Principal growth stages (BBCH)')
    plt.xlabel('Principal BBCH Code')
    plt.ylabel('Accuracy(%)')

    # 调整X轴标签的位置，使其居中，并增加间距
    plt.xticks([x * spacing + bar_width*1.5 for x in range(len(categories))], categories)

    # 设置Y轴范围从75%开始
    plt.ylim(77, 98)
    # 设置Y轴刻度间隔为5
    plt.yticks(range(77, 99, 2))

    # 添加图例
    plt.legend(bbox_to_anchor=(0.5, 1.2), loc='upper center', ncol=2, columnspacing=11.0, frameon=False)

    # 调整图的位置，使其向上移动
    plt.subplots_adjust(bottom=0.18)
    plt.grid(True, zorder=1)

    # 保存图片
    save_path = "D:/Desktop/huatu2/BarChart.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    # 显示图表
    plt.show()

# 混淆矩阵
def fig7():
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams.update({'font.size': 10})


    ResNetRS50 = np.array([[330, 12, 1, 0, 0, 1, 1],
                        [17, 337, 28, 3, 0, 1, 3],
                        [1, 16, 265, 38, 4, 0, 0],
                        [0, 2, 46, 396, 10, 0, 3],
                        [0, 1, 8, 16, 139, 6, 2],
                        [0, 1, 3, 1, 6, 222, 6],
                        [1, 0, 4, 3, 1, 13, 322]])
    

    ResNetRS50_SSL = np.array([[328, 14, 1, 0, 0, 1, 1],
                        [14, 350, 19, 1, 0, 1, 4],
                        [0, 12, 259, 50, 3, 0, 0],
                        [0, 3, 34, 410, 7, 0, 3],
                        [0, 2, 6, 20, 137, 7, 0],
                        [0, 1, 2, 1, 3, 222, 10],
                        [0, 1, 4, 2, 1, 13, 323]])

    C_ResNetRS50_SSL = np.array([[330, 13, 0, 1, 0, 0, 1],
                            [13, 356, 20, 0, 0, 0, 0],
                            [0, 18, 253, 47, 4, 2, 0],
                            [0, 1, 31, 420, 3, 0, 2],
                            [0, 2, 6, 20, 140, 3, 1],
                            [0, 0, 2, 3, 3, 223, 8],
                            [0, 2, 1, 3, 2, 17, 319]])


    CO_ResNetRS50_SSL = np.array([[333, 10, 1, 0, 0, 0, 1],
                            [19, 344, 24, 0, 0, 1, 1],
                            [0, 17, 262, 39, 6, 0, 0],
                            [0, 2, 27, 420, 5, 1, 2],
                            [0, 0, 7, 16, 145, 3, 1],
                            [0, 1, 4, 1, 3, 222, 8],
                            [0, 1, 3, 3, 0, 13, 324]])

    fig, axs = plt.subplots(2, 2, figsize=(7, 6))

    vmin, vmax = 0, 420  # 根据你的数据范围进行调整

    # 替换标签
    new_labels = ['1', '2', '3', '4', '5', '7', '8']

    # 绘制模型1的混淆矩阵
    sns.heatmap(ResNetRS50, annot=True, fmt='d', cmap='Blues', xticklabels=new_labels, yticklabels=new_labels, ax=axs[0, 0], vmin=vmin, vmax=vmax)
    axs[0, 0].set_xlabel('Predicted Principle BBCH Code')
    axs[0, 0].set_ylabel('True Principle BBCH Code')
    # axs[0, 0].set_title('ResNetRS50')
    axs[0, 0].text(0.5, -0.3, "(a) ResNetRS50", ha='center', va='center', transform=axs[0, 0].transAxes,
                fontdict={'fontname': 'Times New Roman', 'fontsize': 10, 'weight': 'bold'})

    # 绘制模型2的混淆矩阵
    sns.heatmap(ResNetRS50_SSL, annot=True, fmt='d', cmap='Blues', xticklabels=new_labels, yticklabels=new_labels, ax=axs[0, 1], vmin=vmin, vmax=vmax)
    axs[0, 1].set_xlabel('Predicted Principle BBCH Code')
    axs[0, 1].set_ylabel('True Principle BBCH Code')
    # axs[0, 1].set_title('ResNetRS50-SSL')
    axs[0, 1].text(0.5, -0.3, "(b) ResNetRS50-SSL", ha='center', va='center', transform=axs[0, 1].transAxes,
                fontdict={'fontname': 'Times New Roman', 'fontsize': 10, 'weight': 'bold'})

    # 绘制模型3的混淆矩阵
    sns.heatmap(C_ResNetRS50_SSL, annot=True, fmt='d', cmap='Blues', xticklabels=new_labels, yticklabels=new_labels, ax=axs[1, 0], vmin=vmin, vmax=vmax)
    axs[1, 0].set_xlabel('Predicted Principle BBCH Code')
    axs[1, 0].set_ylabel('True Principle BBCH Code')
    # axs[1, 0].set_title('C-ResNetRS50-SSL')
    axs[1, 0].text(0.5, -0.3, "(c) C-ResNetRS50-SSL", ha='center', va='center', transform=axs[1, 0].transAxes,
                fontdict={'fontname': 'Times New Roman', 'fontsize': 10, 'weight': 'bold'})

    # 绘制模型4的混淆矩阵
    sns.heatmap(CO_ResNetRS50_SSL, annot=True, fmt='d', cmap='Blues', xticklabels=new_labels, yticklabels=new_labels, ax=axs[1, 1], vmin=vmin, vmax=vmax)
    axs[1, 1].set_xlabel('Predicted Principle BBCH Code')
    axs[1, 1].set_ylabel('True Principle BBCH Code')
    # axs[1, 1].set_title('CO-ResNetRS50-SSL')
    axs[1, 1].text(0.5, -0.3, "(d) CO-ResNetRS50-SSL", ha='center', va='center', transform=axs[1, 1].transAxes,
                fontdict={'fontname': 'Times New Roman', 'fontsize': 10, 'weight': 'bold'})

    # 调整子图之间的间距
    plt.subplots_adjust(hspace=0.4, wspace=0.35)

    save_path = "D:/Desktop/huatu2/ConfusionMatrix.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


# 主函数
def main():
    # 调用不同的方法
    fig1()

# 运行主函数
if __name__ == "__main__":
    main()