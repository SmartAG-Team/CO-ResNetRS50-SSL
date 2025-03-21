import matplotlib.pyplot as plt
import re
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib import font_manager
import ast
from scipy.stats import ttest_rel
from scipy.stats import ttest_ind
from scipy import stats
# 生育期和对应的图片数量
def fig3():

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams.update({'font.size': 10})

    # 读取数据
    training_data = pd.read_csv('./data/dataset_count/train_label_count.csv') 
    validation_data = pd.read_csv('./data/dataset_count/val_label_count.csv')  
    test_data = pd.read_csv('./data/dataset_count/test_label_count.csv')  

    # 提取类别和计数
    categories = training_data['Label'].astype(str).tolist()  # 转换为字符串类型
    traing_dataset = training_data['Count'].tolist()
    validation_dataset = validation_data['Count'].tolist()
    test_dataset = test_data['Count'].tolist()

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
            offset = 30 if value == max(values) else 0  # 只对最大值进行调整
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() - offset, f'{value}', 
                    ha='center', va='bottom', fontsize=10)

    add_values_to_bars(bars1, traing_dataset)
    add_values_to_bars(bars2, validation_dataset)
    add_values_to_bars(bars3, test_dataset)

    # 添加标题和标签
    plt.xlabel('Principal BBCH Code')
    plt.ylabel('Number of images')

    # 调整X轴标签的位置，使其居中，并增加间距
    plt.xticks([x * spacing + bar_width for x in range(len(categories))], categories)

    # 设置Y轴范围从150到2200
    plt.ylim(150, 2200)
    plt.grid(True, zorder=1)

    # 添加图例
    plt.legend(bbox_to_anchor=(0.5, 1.15), loc='upper center', ncol=3, columnspacing=7.0, frameon=False)

    # 调整图的位置，使其向上移动
    plt.subplots_adjust(bottom=0.18)

    # 保存图片
    save_path = "./fig/Fig3.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    # 显示图表
    plt.show()

# 消融实验
def fig9():

    # 设置字体
    plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 10})

    # 模型名称
    models = ['ResNetRS50', 'ResNetRS50-SSL', 'C-ResNetRS50-SSL', 'CO-ResNetRS50-SSL']

    # 每个模型的五次实验路径
    model_paths = {
        'ResNetRS50': [
            './data/ResNetRS50/result.csv',
            './data/ResNetRS50/result2.csv',
            './data/ResNetRS50/result3.csv',
            './data/ResNetRS50/result4.csv',
            './data/ResNetRS50/result5.csv'
        ],
        'ResNetRS50-SSL': [
            './data/ResNetRS50_SSL/ssl_90/result.csv',
            './data/ResNetRS50_SSL/ssl_90/result2.csv',
            './data/ResNetRS50_SSL/ssl_90/result3.csv',
            './data/ResNetRS50_SSL/ssl_90/result4.csv',
            './data/ResNetRS50_SSL/ssl_90/result5.csv'
        ],
        'C-ResNetRS50-SSL': [
            './data/C_ResNetRS50_SSL/result.csv',
            './data/C_ResNetRS50_SSL/result2.csv',
            './data/C_ResNetRS50_SSL/result3.csv',
            './data/C_ResNetRS50_SSL/result4.csv',
            './data/C_ResNetRS50_SSL/result5.csv'
        ],
        'CO-ResNetRS50-SSL': [
            './data/CO_ResNetRS50_SSL/result.csv',
            './data/CO_ResNetRS50_SSL/result2.csv',
            './data/CO_ResNetRS50_SSL/result3.csv',
            './data/CO_ResNetRS50_SSL/result4.csv',
            './data/CO_ResNetRS50_SSL/result5.csv'
        ]
    }

    # 初始化性能数据和参数列表
    performance = []
    parameters = []
    p_values = {'accuracy': [], 'precision': [], 'recall': [], 'f1_score': []}

    # 读取CO-ResNetRS50-SSL的数据（对比的基准模型）
    co_resnet_results = pd.read_csv(model_paths['CO-ResNetRS50-SSL'][0])
    co_resnet_performance = {
        'accuracy': co_resnet_results['Test_Acc'].dropna().values[-1] * 100,
        'precision': co_resnet_results['Test_Pre'].dropna().values[-1] * 100,
        'recall': co_resnet_results['Test_Recall'].dropna().values[-1] * 100,
        'f1_score': co_resnet_results['Test_F1'].dropna().values[-1] * 100,
    }
    co_resnet_params = co_resnet_results['Model_Parameter'].dropna().values[-1]

    # 修改模型参数的提取方式
    for model in models:
        model_performance = []
        model_params = []
        for file in model_paths[model]:
            df = pd.read_csv(file)
            accuracy = df['Test_Acc'].dropna().values[-1] * 100
            precision = df['Test_Pre'].dropna().values[-1] * 100
            recall = df['Test_Recall'].dropna().values[-1] * 100
            f1_score = df['Test_F1'].dropna().values[-1] * 100
            model_performance.append([accuracy, precision, recall, f1_score])
            
            # 提取模型参数
            model_param = df['Model_Parameter'].dropna().values[-1]  # 取参数量
            model_params.append(model_param)

        model_performance = np.array(model_performance)

        # 计算每个性能指标的平均值
        mean_performance = np.mean(model_performance, axis=0)

        # 计算每个性能指标的t检验（基于五次实验）
        for metric, idx in zip(['accuracy', 'precision', 'recall', 'f1_score'], range(4)):
            t_stat, p_value = stats.ttest_ind(model_performance[:, idx], [co_resnet_performance[metric]] * 5)
            p_values[metric].append(p_value)

        # 计算参数量的平均值
        mean_param = np.mean(model_params)

        # 将平均参数量添加到参数列表
        parameters.append(mean_param)

        # 将平均性能添加到性能列表
        performance.append(mean_performance)

    # 转换性能为numpy数组
    performance = np.array(performance).T

    # 颜色设置
    colors = ['#EC6E66', '#91CCC0', '#F7AC53', '#B5CE4E']

    # 设置柱形图的位置和宽度
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 score', 'Parameter']
    x = np.arange(len(metrics))
    width = 0.2
    offset = (len(models) - 1) * width / 2

    # 创建图形
    fig, ax1 = plt.subplots(figsize=(7, 5))

    # 绘制性能指标条形图
    for i in range(len(models)):
        ax1.bar(x[:-1] + i * width - offset, performance[:, i], width, label=models[i], color=colors[i], zorder=2)

    # 添加数值标签
    for i in range(len(metrics) - 1):
        for j in range(len(models)):
            text_offset = 0.1
            if models[j] == 'SwinTransformer':
                text_offset = 0.4
            ax1.text(x[i] + j * width - offset, performance[i, j] + text_offset, f'{performance[i, j]:.2f}',
                    ha='center', va='bottom', fontsize=10)

    # 设置y轴标签
    ax1.set_ylabel('Values of Accuracy, Recall, Precision, and F1 score (%)')
    ax1.set_ylim(87, 92)

    # 设置x轴标签
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)

    # 设置右侧y轴
    ax2 = ax1.twinx()
    ax2.set_ylabel('Size of Parameter (M)')
    ax2.tick_params(axis='y', rotation=45)
    ax2.set_ylim(10, 90)

    # 绘制参数量的条形图
    for i in range(len(models)):
        ax2.bar(x[-1] + i * width - offset, parameters[i], width, label=models[i], color=colors[i])

    # 添加参数量数值标签
    for i in range(len(models)):
        ax2.text(x[-1] + i * width - offset, parameters[i] + 0.5, f'{parameters[i]:.2f}',
                ha='center', va='bottom', fontsize=10, rotation=45)

    # 在图上标记显著性差异
    for i in range(len(models)):
        if models[i] != 'CO-ResNetRS50-SSL':
            for metric, idx in zip(['accuracy', 'precision', 'recall', 'f1_score'], range(4)):
                p_value = p_values[metric][i]
                if p_value < 0.001:  # p-value < 0.001 indicates significant difference
                    star = '***'
                elif p_value < 0.01:  # p-value < 0.01 indicates significant difference
                    star = '**'
                elif p_value < 0.05:  # p-value < 0.05 indicates significant difference
                    star = '*'
                else:
                    star = 'ns'

                # 增加递增的高度偏移量
                line_offset = 0.3 * i  # 每个模型递增的高度

                # 连接线的绘制
                if p_value < 0.05:
                    current_y = performance[idx, i] + 0.5 + line_offset  # 当前模型的y值，增加高度偏移
                    co_resnet_y = performance[idx, -1] + 0.5 + line_offset  # CO-ResNetRS50-SSL的y值，增加高度偏移

                    # 水平连接线
                    ax1.plot([x[idx] + i * width - offset, x[idx] + (len(models) - 1) * width - offset],
                            [max(current_y, co_resnet_y), max(current_y, co_resnet_y)], color='black', lw=1)

                    # 当前模型的竖直连接线
                    ax1.plot([x[idx] + i * width - offset, x[idx] + i * width - offset],
                            [performance[idx, i] + 0.3, max(current_y, co_resnet_y)], color='black', lw=1)

                    # 基准模型的竖直连接线
                    ax1.plot([x[idx] + (len(models) - 1) * width - offset, x[idx] + (len(models) - 1) * width - offset],
                            [performance[idx, -1] + 0.3, max(current_y, co_resnet_y)], color='black', lw=1)

                    # 在连接线的中间位置添加显著性标记
                    mid_x = (x[idx] + i * width - offset)
                    mid_y = max(current_y, co_resnet_y) + 0  # 上移一点，避免重叠
                    ax1.text(mid_x, mid_y, star, ha='center', va='bottom', fontsize=10, color='black')

    # 在右上角添加备注
    ax1.text(0.97, 0.99, '* denotes p ≤ 0.05', transform=ax1.transAxes, ha='right', va='top', fontsize=10, color='red')
    ax1.text(0.97, 0.94, '*** denotes p ≤ 0.001', transform=ax1.transAxes, ha='right', va='top', fontsize=10, color='red')
    # 显示图例
    # ax1.legend(loc='upper left')
    # 显示图例在正上方，设置两排两列
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.16), ncol=2, columnspacing=12)

    # 显示网格线
    ax1.grid(axis='y', alpha=0.3, zorder=1)

    # 保存图像
    save_path = "./fig/Fig9.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    # 显示图形
    plt.show()

def fig10():

    plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 10})

    # 模型名称
    models = ['ConvNext-base', 'FasterNet-t1', 'ShuffleNetV2', 'SwinTransformer', 'Vision Transformer', 'CO-ResNetRS50-SSL']

    # 模型对应的csv文件路径
    model_paths = {
        'ConvNext-base': [
            './data/ConvNext/result.csv',
            './data/ConvNext/result2.csv',
            './data/ConvNext/result3.csv',
            './data/ConvNext/result4.csv',
            './data/ConvNext/result5.csv'
        ],
        'ShuffleNetV2': [
            './data/ShuffleNetV2/result.csv',
            './data/ShuffleNetV2/result2.csv',
            './data/ShuffleNetV2/result3.csv',
            './data/ShuffleNetV2/result4.csv',
            './data/ShuffleNetV2/result5.csv'
        ],
        'FasterNet-t1': [
            './data/FasterNet_t1/result.csv',
            './data/FasterNet_t1/result2.csv',
            './data/FasterNet_t1/result3.csv',
            './data/FasterNet_t1/result4.csv',
            './data/FasterNet_t1/result5.csv'
        ],
        'Vision Transformer': [
            './data/vit/result.csv',
            './data/vit/result2.csv',
            './data/vit/result3.csv',
            './data/vit/result4.csv',
            './data/vit/result5.csv'
        ],
        'SwinTransformer': [
            './data/SwinTransformer/result.csv',
            './data/SwinTransformer/result2.csv',
            './data/SwinTransformer/result3.csv',
            './data/SwinTransformer/result4.csv',
            './data/SwinTransformer/result5.csv'
        ],
        'CO-ResNetRS50-SSL': [
            './data/CO_ResNetRS50_SSL/result.csv',
            './data/CO_ResNetRS50_SSL/result2.csv',
            './data/CO_ResNetRS50_SSL/result3.csv',
            './data/CO_ResNetRS50_SSL/result4.csv',
            './data/CO_ResNetRS50_SSL/result5.csv'
        ]
    }

    # 初始化性能数据
    performance = {metric: [] for metric in ['Accuracy', 'Precision', 'Recall', 'F1 score']}
    parameters = []

    # 循环读取每个模型的CSV文件
    for model in models:
        csv_files = model_paths[model]
        temp_results = {metric: [] for metric in performance.keys()}

        for csv_file in csv_files:
            df = pd.read_csv(csv_file)

            # 提取 Test_Acc, Test_Pre, Test_Recall, Test_F1 的所有试验数据
            temp_results['Accuracy'].append(df['Test_Acc'].dropna().values[0] * 100)
            temp_results['Precision'].append(df['Test_Pre'].dropna().values[0] * 100)
            temp_results['Recall'].append(df['Test_Recall'].dropna().values[0] * 100)
            temp_results['F1 score'].append(df['Test_F1'].dropna().values[0] * 100)

        # 将每个模型的五次试验的平均值添加到 performance 中
        for metric in performance.keys():
            performance[metric].append(np.mean(temp_results[metric]))  # 使用五次试验的平均值

        # 提取模型参数（只需从第一个文件中提取一次）
        param = df['Model_Parameter'].dropna().values[0]
        parameters.append(param)

    # CO-ResNetRS50-SSL 模型的性能数据（用于 t 检验对比，五次试验的数据）
    co_resnet_performance = {metric: temp_results[metric] for metric in performance.keys()}

    # t 检验
    ttest_results = {metric: [] for metric in performance.keys()}
    for metric in performance.keys():
        co_resnet_samples = co_resnet_performance[metric]  # CO-ResNetRS50-SSL 模型的所有样本数据
        for i in range(len(models) - 1):  # 忽略 CO-ResNetRS50-SSL 模型
            other_model_samples = performance[metric][i]  # 其他模型的五次试验数据
            t_stat, p_value = ttest_ind(other_model_samples, co_resnet_samples)  # 使用样本组进行 t 检验
            ttest_results[metric].append(p_value)

    # 创建子图布局
    metrics = ['Accuracy', 'Recall', 'Precision', 'F1 score', 'Parameter']
    fig, axes = plt.subplots(3, 2, figsize=(7, 8))
    axes = axes.flatten()
    colors = ['#EC6E66', '#F7AC53', '#B5CE4E', '#6A5ACD', '#FFA07A', '#4682B4']

    for i, metric in enumerate(metrics):
        ax = axes[i]

        if metric != 'Parameter':
            data = performance[metric]
            ylabel = f'Values of {metric} (%)'
            ylim = (80, 98)
        else:
            data = parameters
            ylabel = f'Values of {metric} (M)'
            ylim = (0, 100)

        # 绘制柱形图
        x = np.arange(len(models))
        width = 0.5
        bars = ax.bar(x, data, width, color=colors[:len(models)])

        # 添加显著性标注（*号）
        if metric != 'Parameter':
            # 初始化竖线和水平线的高度偏移量
            base_y = max(data) + 1.0  # 基础高度
            height_step = 1.2  # 每个模型之间的高度差
            
            # 添加连接线（折线），不同模型的竖线高度逐渐增高
            for j in range(len(models) - 1):  # 忽略 CO-ResNetRS50-SSL
                if ttest_results[metric][j] < 0.05:  # 如果显著性小于 0.05
                    current_y = base_y + j * height_step  # 根据模型编号增加高度偏移
                    ax.plot([j, j], [data[j] + 0.5, current_y], color='black', lw=1)  # 第一个模型的竖线
                    ax.plot([len(models) - 1, len(models) - 1], [data[-1] + 0.5, current_y], color='black', lw=1)  # CO-ResNetRS50-SSL 模型的竖线
                    ax.plot([j, len(models) - 1], [current_y, current_y], color='black', lw=1)  # 水平线
                    
                    # 添加显著性标记到水平线上方
                    if ttest_results[metric][j] < 0.001:
                        # ax.text(j + (len(models) - 1 - j) / 2, current_y - 0.2, '***', ha='center', va='bottom', fontsize=12, color='black')
                        ax.text(j, current_y - 0.2, '***', ha='center', va='bottom', fontsize=12, color='black')
                    elif ttest_results[metric][j] < 0.01:
                        ax.text(j, current_y - 0.2, '**', ha='center', va='bottom', fontsize=12, color='black')
                    elif ttest_results[metric][j] < 0.05:
                        ax.text(j, current_y - 0.2, '*', ha='center', va='bottom', fontsize=12, color='black')
                    else:
                        ax.text(j, current_y - 0.2, 'ns', ha='center', va='bottom', fontsize=12, color='black')


        # 设置子图标题和标签
        ax.set_ylabel(ylabel)
        if metric in ['Accuracy', 'Recall', 'Precision']:
            ax.set_xticks([])  # 不显示 x 轴刻度标签
        else:
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=45, fontsize=9)
        ax.set_ylim(ylim)

    # 删除多余的子图框架
    for i in range(len(metrics), len(axes)):
        fig.delaxes(axes[i])

    # 在右下角添加备注
    fig.text(0.83, 0.23, '** denotes p ≤ 0.01', ha='right', va='bottom', fontsize=10, color='red')
    fig.text(0.83, 0.20, '*** denotes p ≤ 0.001', ha='right', va='bottom', fontsize=10, color='red')

    # 调整布局并保存图像
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.1, wspace=0.2)
    save_path = "./fig/Fig10.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    # 显示图形
    plt.show()


# 半监督置信度阈值的图
def fig11():
    # 设置字体
    plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 10})

    # 读取CSV文件
    files = [
        "./data/ResNetRS50/result.csv",                                # baseline
        "./data/ResNetRS50_SSL/ssl_85/result.csv",                     # ResNetRS50-SSL-85
        "./data/ResNetRS50_SSL/ssl_875/result.csv",                    # ResNetRS50-SSL-85
        "./data/ResNetRS50_SSL/ssl_90/result.csv",                     # ResNetRS50-SSL-85
        "./data/ResNetRS50_SSL/ssl_925/result.csv",                    # ResNetRS50-SSL-85
        "./data/ResNetRS50_SSL/ssl_95/result.csv"                      # ResNetRS50-SSL-85
    ]

    # 初始化指标列表
    metrics = {'Accuracy': [], 'Recall': [], 'Precision': [], 'F1score': []}

    # 读取每个模型的CSV文件并提取指标
    for file in files:
        df = pd.read_csv(file)
        # 假设测试集的指标在最后一行，提取相应的值
        metrics['Accuracy'].append(df['Test_Acc'].iloc[-1] * 100)  # 转换为百分比
        metrics['Recall'].append(df['Test_Recall'].iloc[-1] * 100)
        metrics['Precision'].append(df['Test_Pre'].iloc[-1] * 100)
        metrics['F1score'].append(df['Test_F1'].iloc[-1] * 100)

    # X axis for confidence thresholds
    confidence_thresholds = ['Baseline', '0.85', '0.875', '0.90', '0.925', '0.95']

    # Plot data for each metric
    plt.figure(figsize=(7, 5))

    # Plot lines for each metric
    for label in metrics:
        plt.plot(confidence_thresholds, metrics[label], marker='o', label=label)
        # Annotate each point with its value
        for j, value in enumerate(metrics[label]):
            plt.text(confidence_thresholds[j], value + 0.04, f'{value:.2f}', ha='center', va='bottom', fontsize=10)

    # Add labels and title
    plt.xlabel('Confidence Threshold')
    plt.ylabel('Metric Value (%)')
    plt.legend()
    plt.grid(True)

    # Show plot
    plt.tight_layout(pad=0.4)
    save_path = "./fig/Fig11.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')  # 提高dpi以保持字体清晰
    plt.show()

# 添加图片的数量
def fig12():

    # 文件路径
    file_paths = {
        "ResNetRS50-SSL-85": './data/ResNetRS50_SSL/ssl_85/result.csv',
        "ResNetRS50-SSL-925": './data/ResNetRS50_SSL/ssl_925/result.csv',
        "ResNetRS50-SSL-875": './data/ResNetRS50_SSL/ssl_875/result.csv',
        "ResNetRS50-SSL-95": './data/ResNetRS50_SSL/ssl_95/result.csv',
        "ResNetRS50-SSL-90": './data/ResNetRS50_SSL/ssl_90/result.csv',
    }

    # 提取数据
    total_difference = {label: [] for label in file_paths.keys()}
    epochs = {label: [] for label in file_paths.keys()}  # 存储每个模型的epochs

    for label, path in file_paths.items():
        try:
            # 读取CSV文件，并处理None或NaN值
            df = pd.read_csv(path)
            df = df.dropna(subset=['Number_of_Unlabed_Images'])  # 删除'Number_of_Unlabed_Images'为None或NaN的行
            df['Number_of_Unlabed_Images'] = df['Number_of_Unlabed_Images'].astype(int)  # 转换为整数类型

            # 提取Epoch和Number_of_Unlabed_Images列的值
            epochs[label] = df['Epoch'].tolist()
            total_difference[label] = df['Number_of_Unlabed_Images'].tolist()
        except Exception as e:
            print(f"Error reading {path}: {e}")

    # 设置字体
    plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 10})

    # 创建子图
    fig, ax = plt.subplots(figsize=(7, 5))  # 只创建一个子图

    # 绘制total_difference
    for label in file_paths.keys():
        ax.plot(epochs[label], total_difference[label], label=label, linewidth=1.0)

    ax.set_ylabel('Number of Unlabeled Images')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 7000)
    yticks = range(0, 7001, 500)  # 每隔500设置一个刻度
    ax.set_yticks(yticks)
    ax.set_xlabel('Epoch')  # 添加x轴标签

    # 绘制图例
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.10), ncol=3, columnspacing=7.0)

    # 调整布局
    plt.tight_layout(pad=0.4)
    save_path = "./fig/Fig12.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')  # 提高dpi以保持字体清晰
    plt.show()

# 具体生育期的热力图
def fig13():
    # 设置字体
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams.update({'font.size': 10})

    # 读取每个模型对应的CSV文件
    ResNetRS50_path = './data/ResNetRS50/result.csv' 
    ResNetRS50_SSL_path = './data/ResNetRS50_SSL/ssl_90/result.csv'  
    C_ResNetRS50_SSL_path = './data/C_ResNetRS50_SSL/result.csv'  
    CO_ResNetRS50_SSL_path = './data/CO_ResNetRS50_SSL/result.csv' 

    # 分别读取每个文件
    df_resnet = pd.read_csv(ResNetRS50_path)
    df_resnet_ssl = pd.read_csv(ResNetRS50_SSL_path)
    df_c_resnet_ssl = pd.read_csv(C_ResNetRS50_SSL_path)
    df_co_resnet_ssl = pd.read_csv(CO_ResNetRS50_SSL_path)

    # 提取每个模型的测试集类别准确率（确保列名一致）
    categories = ['1', '2', '3', '4', '5', '7', '8']  # 类别

    ResNetRS50 = df_resnet[['Test_1_Acc', 'Test_2_Acc', 'Test_3_Acc', 'Test_4_Acc', 'Test_5_Acc', 'Test_7_Acc', 'Test_8_Acc']].iloc[79].values * 100
    ResNetRS50_SSL = df_resnet_ssl[['Test_1_Acc', 'Test_2_Acc', 'Test_3_Acc', 'Test_4_Acc', 'Test_5_Acc', 'Test_7_Acc', 'Test_8_Acc']].iloc[79].values * 100
    C_ResNetRS50_SSL = df_c_resnet_ssl[['Test_1_Acc', 'Test_2_Acc', 'Test_3_Acc', 'Test_4_Acc', 'Test_5_Acc', 'Test_7_Acc', 'Test_8_Acc']].iloc[79].values * 100
    CO_ResNetRS50_SSL = df_co_resnet_ssl[['Test_1_Acc', 'Test_2_Acc', 'Test_3_Acc', 'Test_4_Acc', 'Test_5_Acc', 'Test_7_Acc', 'Test_8_Acc']].iloc[79].values * 100

    # 将数据组合成矩阵
    data = np.array([
        ResNetRS50,
        ResNetRS50_SSL,
        C_ResNetRS50_SSL,
        CO_ResNetRS50_SSL
    ])

    # 模型名称
    models = ['ResNetRS50', 'ResNetRS50-SSL', 'C-ResNetRS50-SSL', 'CO-ResNetRS50-SSL']

    # 创建热图
    fig, ax = plt.subplots(figsize=(7, 5))
    cax = ax.imshow(data, cmap="YlGn", aspect="auto")

    # 添加颜色条
    cbar = fig.colorbar(cax, ax=ax)
    cbar.set_label("Accuracy(%)", rotation=270, labelpad=15)

    # 设置轴标签和刻度
    ax.set_xticks(np.arange(len(categories)))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels(categories)
    ax.set_yticklabels(models, rotation=45)
    ax.set_xlabel("Principal BBCH Code")
    ax.set_ylabel("Models")


    # 显示每个单元格的值
    for i in range(len(models)):
        for j in range(len(categories)):
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", color="black")

    plt.tight_layout()

    # 保存图片
    save_path = "./fig/Fig13.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    # 显示图表
    plt.show()

# 混淆矩阵
def fig14():

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams.update({'font.size': 10})

    # 读取不同模型的混淆矩阵
    df_resnetrs50 = pd.read_csv('./data/ResNetRS50/result.csv')
    df_resnetrs50_ssl = pd.read_csv('./data/ResNetRS50_SSL/ssl_90/result.csv')
    df_c_resnetrs50_ssl = pd.read_csv('./data/C_ResNetRS50_SSL/result.csv')
    df_co_resnetrs50_ssl = pd.read_csv('./data/CO_ResNetRS50_SSL/result.csv')


    ResNetRS50 = df_resnetrs50['Test_Confusion_Matrix'][79]
    ResNetRS50_SSL = df_resnetrs50_ssl['Test_Confusion_Matrix'][79]
    C_ResNetRS50_SSL = df_c_resnetrs50_ssl['Test_Confusion_Matrix'][79]
    CO_ResNetRS50_SSL = df_co_resnetrs50_ssl['Test_Confusion_Matrix'][79]



    ResNetRS50 = ResNetRS50.replace('[', '').replace(']', '').replace('\n', '')
    ResNetRS50_values = list(map(int, ResNetRS50.split()))
    ResNetRS50 = np.array(ResNetRS50_values).reshape(7, 7)

    ResNetRS50_SSL = ResNetRS50_SSL.replace('[', '').replace(']', '').replace('\n', '')
    ResNetRS50_SSL_values = list(map(int, ResNetRS50_SSL.split()))
    ResNetRS50_SSL = np.array(ResNetRS50_SSL_values).reshape(7, 7)

    C_ResNetRS50_SSL = C_ResNetRS50_SSL.replace('[', '').replace(']', '').replace('\n', '')
    C_ResNetRS50_SSL_values = list(map(int, C_ResNetRS50_SSL.split()))
    C_ResNetRS50_SSL = np.array(C_ResNetRS50_SSL_values).reshape(7, 7)

    CO_ResNetRS50_SSL = CO_ResNetRS50_SSL.replace('[', '').replace(']', '').replace('\n', '')
    CO_ResNetRS50_SSL_values = list(map(int, CO_ResNetRS50_SSL.split()))
    CO_ResNetRS50_SSL = np.array(CO_ResNetRS50_SSL_values).reshape(7, 7)


    fig, axs = plt.subplots(2, 2, figsize=(7, 6))

    vmin, vmax = 0, 420  # 根据数据范围进行调整

    # 替换标签
    new_labels = ['1', '2', '3', '4', '5', '7', '8']

    # 绘制模型1的混淆矩阵
    sns.heatmap(ResNetRS50, annot=True, fmt='d', cmap='Blues', xticklabels=new_labels, yticklabels=new_labels, ax=axs[0, 0], vmin=vmin, vmax=vmax)
    axs[0, 0].set_xlabel('Predicted Principle BBCH Code')
    axs[0, 0].set_ylabel('True Principle BBCH Code')
    axs[0, 0].text(0.5, -0.3, "(a) ResNetRS50", ha='center', va='center', transform=axs[0, 0].transAxes,
                fontdict={'fontname': 'Times New Roman', 'fontsize': 10, 'weight': 'bold'})

    # 绘制模型2的混淆矩阵
    sns.heatmap(ResNetRS50_SSL, annot=True, fmt='d', cmap='Blues', xticklabels=new_labels, yticklabels=new_labels, ax=axs[0, 1], vmin=vmin, vmax=vmax)
    axs[0, 1].set_xlabel('Predicted Principle BBCH Code')
    axs[0, 1].set_ylabel('True Principle BBCH Code')
    axs[0, 1].text(0.5, -0.3, "(b) ResNetRS50-SSL", ha='center', va='center', transform=axs[0, 1].transAxes,
                fontdict={'fontname': 'Times New Roman', 'fontsize': 10, 'weight': 'bold'})

    # 绘制模型3的混淆矩阵
    sns.heatmap(C_ResNetRS50_SSL, annot=True, fmt='d', cmap='Blues', xticklabels=new_labels, yticklabels=new_labels, ax=axs[1, 0], vmin=vmin, vmax=vmax)
    axs[1, 0].set_xlabel('Predicted Principle BBCH Code')
    axs[1, 0].set_ylabel('True Principle BBCH Code')
    axs[1, 0].text(0.5, -0.3, "(c) C-ResNetRS50-SSL", ha='center', va='center', transform=axs[1, 0].transAxes,
                fontdict={'fontname': 'Times New Roman', 'fontsize': 10, 'weight': 'bold'})

    # 绘制模型4的混淆矩阵
    sns.heatmap(CO_ResNetRS50_SSL, annot=True, fmt='d', cmap='Blues', xticklabels=new_labels, yticklabels=new_labels, ax=axs[1, 1], vmin=vmin, vmax=vmax)
    axs[1, 1].set_xlabel('Predicted Principle BBCH Code')
    axs[1, 1].set_ylabel('True Principle BBCH Code')
    axs[1, 1].text(0.5, -0.3, "(d) CO-ResNetRS50-SSL", ha='center', va='center', transform=axs[1, 1].transAxes,
                fontdict={'fontname': 'Times New Roman', 'fontsize': 10, 'weight': 'bold'})

    # 调整子图之间的间距
    plt.subplots_adjust(hspace=0.4, wspace=0.35)
    save_path = "./fig/Fig14.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def fig15():
    # 设置字体
    plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 10})

    # 读取CSV文件
    files = [
        './data/CO_ResNetRS50_SSL/result_128.csv',
        './data/CO_ResNetRS50_SSL/result.csv',
        './data/CO_ResNetRS50_SSL/result_512.csv',
    ]

    # 初始化指标列表
    metrics = {'Accuracy': [], 'Recall': [], 'Precision': [], 'F1 score': []}
    hours = []  # 初始化训练时间列表
    latency = []  # 初始化延迟指标列表

    # 读取每个模型的CSV文件并提取指标
    for file in files:
        df = pd.read_csv(file)
        # 假设测试集的指标在最后一行，提取相应的值
        metrics['Accuracy'].append(df['Test_Acc'].iloc[-1] * 100)  # 转换为百分比
        metrics['Recall'].append(df['Test_Recall'].iloc[-1] * 100)
        metrics['Precision'].append(df['Test_Pre'].iloc[-1] * 100)
        metrics['F1 score'].append(df['Test_F1'].iloc[-1] * 100)
        hours.append(df['Hours'].iloc[-1])  # 提取训练时间
        latency.append(df['Latency'].iloc[-1])  # 提取延迟时间（单位：毫秒）

    size = ['128×128', '224×224', '512×512']

    # 创建图形
    fig, ax1 = plt.subplots(figsize=(7, 5))

    # 绘制左侧 y 轴的折线图
    for label in metrics:
        ax1.plot(size, metrics[label], marker='o', label=label)
        # 为每个点添加注释
        for j, value in enumerate(metrics[label]):
            ax1.text(size[j], value + 0.04, f'{value:.2f}', ha='center', va='bottom', fontsize=10)

    # 设置左侧 y 轴
    ax1.set_ylim(86, 91)  # 根据需要修改范围
    ax1.set_ylabel('Values of Accuracy, Recall, Precision, and F1 score. (%)')
    ax1.grid(True)

    # 创建右侧第一个 y 轴（Training time）
    ax2 = ax1.twinx()
    ax2.plot(size, hours, marker='s', color='mediumpurple', label='Training time')

    # 为 Training time 添加注释
    for j, value in enumerate(hours):
        ax2.text(size[j], value + 0.1, f'{value:.2f}', ha='center', va='bottom', fontsize=10, color='mediumpurple')

    # 设置 Training time 的轴标签和刻度
    ax2.set_ylabel('Training time (Hour)', color='mediumpurple')
    ax2.tick_params(axis='y', labelcolor='mediumpurple')  # 设置刻度颜色
    ax2.spines["right"].set_edgecolor("mediumpurple")  # 设置右侧边框颜色
    ax2.set_ylim(0, 14)  # 根据需要调整范围

    # 创建右侧第二个 y 轴（Latency）
    ax3 = ax1.twinx()  # 创建新 y 轴
    ax3.spines["right"].set_position(("axes", 1.1))  # 将新轴移到更右侧
    ax3.plot(size, latency, marker='^', color='teal', label='Latency')

    # 为 Latency 添加注释
    for j, value in enumerate(latency):
        ax3.text(size[j], value + 0.1, f'{value:.2f}', ha='center', va='bottom', fontsize=10, color='teal')

    # 设置 Latency 的轴标签和刻度
    ax3.set_ylabel('Latency (ms)', color='teal')
    ax3.tick_params(axis='y', labelcolor='teal')  # 设置刻度颜色
    ax3.spines["right"].set_edgecolor("teal")  # 设置右侧边框颜色
    ax3.set_ylim(18.3, 22)  # 根据需要调整范围

    # 合并图例
    lines_labels = [ax.get_legend_handles_labels() for ax in [ax1, ax2, ax3]]
    lines, labels = [], []
    for line, label in lines_labels:
        lines.extend(line)
        labels.extend(label)

    # 在顶部居中添加图例
    fig.legend(lines, labels, loc='upper left', bbox_to_anchor=(0.1, 0.97), fontsize=10)

    # 调整布局并保存图像
    plt.tight_layout(pad=0.4)
    save_path = "./fig/Fig15.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')  # 提高dpi以保持字体清晰
    plt.show()


# 主函数
def main():
    # 调用不同的方法
    fig10()

# 运行主函数
if __name__ == "__main__":
    main()