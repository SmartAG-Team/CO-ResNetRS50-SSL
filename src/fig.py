import matplotlib.pyplot as plt
import re
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib import font_manager
import ast
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
def fig8():
    # 设置字体
    plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 10})

    # 模型名称
    models = ['ResNetRS50', 'ResNetRS50-SSL', 'C-ResNetRS50-SSL', 'CO-ResNetRS50-SSL']

    # 对应每个模型的CSV文件路径
    csv_files = [
        './data/ResNetRS50/result.csv',
        './data/ResNetRS50_SSL/ssl_90/result.csv',
        './data/C_ResNetRS50_SSL/result.csv',
        './data/CO_ResNetRS50_SSL/result.csv'
    ]

    # 初始化性能数据和参数列表
    performance = []
    parameters = []

    # 循环读取每个模型的CSV文件并提取需要的值
    for csv_file in csv_files:
        # 读取CSV文件
        df = pd.read_csv(csv_file)
        
        # 提取Test_Acc, Test_Pre, Test_Recall, Test_F1的最后一个epoch数据
        accuracy = df['Test_Acc'].dropna().values[-1] * 100  # 转为百分比
        precision = df['Test_Pre'].dropna().values[-1] * 100
        recall = df['Test_Recall'].dropna().values[-1] * 100
        f1_score = df['Test_F1'].dropna().values[-1] * 100
        
        # 提取模型参数
        model_param = df['Model_Parameter'].dropna().values[-1]  # 假设需要将单位从百万(M)调整
        
        # 将数据添加到performance和parameters列表中
        performance.append([accuracy, recall, precision, f1_score])
        parameters.append(model_param)

    # 将performance转换为numpy数组，方便后续处理
    performance = np.array(performance).T  # 转置，使其与之前的代码结构保持一致

    # 颜色设置
    colors = ['#EC6E66', '#91CCC0', '#F7AC53', '#B5CE4E']

    # 设置柱形图的位置和宽度
    metrics = ['Accuracy', 'Recall', 'Precision', 'F1 score', 'Parameter']
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
    save_path = "./fig/Fig8.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')  # 提高dpi以保持字体清晰

    # 显示图形
    plt.show()

# 半监督置信度阈值的图
def fig9():
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
    save_path = "./fig/Fig9.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')  # 提高dpi以保持字体清晰
    plt.show()

# 添加图片的数量
def fig10():

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
    save_path = "./fig/Fig10.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')  # 提高dpi以保持字体清晰
    plt.show()

# 具体生育期的条形图
def fig11():

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

    # 设置条形图的宽度
    bar_width = 0.30
    spacing = 1.3  # 通过调整这个值增大类别之间的间距

    # 创建图表并设置尺寸
    plt.figure(figsize=(7, 4))  # 适当增大图表的宽度

    # 创建条形图
    bars1 = plt.bar([x * spacing for x in range(len(categories))], ResNetRS50, width=bar_width, color='#EC6E66', label='ResNetRS50', zorder=2)
    bars2 = plt.bar([x * spacing + bar_width for x in range(len(categories))], ResNetRS50_SSL, width=bar_width, color='#91CCC0', label='ResNetRS50-SSL', zorder=2)
    bars3 = plt.bar([x * spacing + bar_width*2 for x in range(len(categories))], C_ResNetRS50_SSL, width=bar_width, color='#F7AC53', label='C-ResNetRS50-SSL', zorder=2)
    bars4 = plt.bar([x * spacing + bar_width*3 for x in range(len(categories))], CO_ResNetRS50_SSL, width=bar_width, color='#B5CE4E', label='CO-ResNetRS50-SSL', zorder=2)

    # 添加标题和标签
    plt.xlabel('Principal BBCH Code')
    plt.ylabel('Accuracy(%)')

    # 调整X轴标签的位置，使其居中，并增加间距
    plt.xticks([x * spacing + bar_width*1.5 for x in range(len(categories))], categories)

    # 设置Y轴范围从77%开始
    plt.ylim(77, 98)
    # 设置Y轴刻度间隔为2
    plt.yticks(range(77, 99, 2))

    # 添加图例
    plt.legend(bbox_to_anchor=(0.5, 1.2), loc='upper center', ncol=2, columnspacing=11.0, frameon=False)

    # 调整图的位置，使其向上移动
    plt.subplots_adjust(bottom=0.18)
    plt.grid(True, zorder=1)

    # 保存图片
    save_path = "./fig/Fig11.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    # 显示图表
    plt.show()

# 混淆矩阵
def fig12():

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
    save_path = "./fig/Fig12.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


# 主函数
def main():
    # 调用不同的方法
    fig10()

# 运行主函数
if __name__ == "__main__":
    main()