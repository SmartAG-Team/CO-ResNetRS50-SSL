import os
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets.folder import default_loader


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = os.path.join('./RiceImagesData/', self.dataframe.iloc[idx, 0] + '.jpeg')
        image = default_loader(img_name)

        label = int(self.dataframe.iloc[idx, 1])
        label_mapping = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 7: 5, 8: 6}
        label = label_mapping[label]

        if self.transform:
            image = self.transform(image)

        return image, label
    

class UnlableDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = os.path.join('./RiceImagesData/', self.dataframe.iloc[idx, 0] + '.jpeg')
        image = default_loader(img_name)
        uuid = self.dataframe.iloc[idx, 0]  # 获取uuid
        if self.transform:
            image = self.transform(image)
        return image, uuid

def count_labels(dataset):
    label_count = {}
    for _, label in dataset:
        if label in label_count:
            label_count[label] += 1
        else:
            label_count[label] = 1
    return label_count

def save_label_count_to_csv(label_count, filename):
    # 标签映射字典
    label_mapping = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 7, 6: 8}
    # 使用映射更新标签
    mapped_label_count = {label_mapping[label]: count for label, count in label_count.items()}
    # 将字典转换为DataFrame并按标签排序
    df = pd.DataFrame(list(mapped_label_count.items()), columns=['Label', 'Count'])
    df.sort_values(by='Label', ascending=True, inplace=True)  # 按标签排序
    df.to_csv(filename, index=False)
    print(f"标签统计结果已保存到 {filename}")

def load_data(args):
    train_csv_file = './data/train_data.csv'
    val_csv_file = './data/val_data.csv'
    test_csv_file = './data/test_data.csv'
    unlabeled_csv_file = './data/unlabeled_data.csv'

    # 读取训练集、验证集和测试集数据
    train_df = pd.read_csv(train_csv_file)
    val_df = pd.read_csv(val_csv_file)
    test_df = pd.read_csv(test_csv_file)
    unlabeled_df = pd.read_csv(unlabeled_csv_file)

    transform_train = transforms.Compose([
        transforms.Resize(args.input_size),
        transforms.RandomCrop(224, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
    ])

    transform_unlabeled = transform_train

    transform_val = transforms.Compose([
        transforms.Resize(args.input_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
    ])

    transform_test = transform_val

    # 创建数据集
    train_dataset = CustomDataset(train_df, transform=transform_train)
    val_dataset = CustomDataset(val_df, transform=transform_val)
    test_dataset = CustomDataset(test_df, transform=transform_test)
    unlabeled_dataset = UnlableDataset(unlabeled_df, transform=transform_unlabeled)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=args.batch_size, shuffle=False)

    # 统计并保存训练集中各个标签的图像数量
    train_label_count = count_labels(train_loader.dataset)
    print("训练集中各个标签的图像数量：", train_label_count)
    save_label_count_to_csv(train_label_count, './data/dataset_count/train_label_count.csv')

    # 统计并保存验证集中各个标签的图像数量
    val_label_count = count_labels(val_loader.dataset)
    print("验证集中各个标签的图像数量：", val_label_count)
    save_label_count_to_csv(val_label_count, './data/dataset_count/val_label_count.csv')

    # 统计并保存测试集中各个标签的图像数量
    test_label_count = count_labels(test_loader.dataset)
    print("测试集中各个标签的图像数量：", test_label_count)
    save_label_count_to_csv(test_label_count, './data/dataset_count/test_label_count.csv')

    return train_loader, val_loader, test_loader, unlabeled_loader



