import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets.folder import default_loader
import os
from collections import Counter
import random

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = os.path.join('../../Rice2/gs_img/', self.dataframe.iloc[idx, 0] + '.jpeg')
        image = default_loader(img_name)

        label = int(self.dataframe.iloc[idx, 1])
        label_mapping = {10: 0, 20: 1, 30: 2, 40: 3, 50: 4, 70: 5, 80: 6}
        # label_mapping = {10: 0, 11: 1, 12: 2, 13: 3, 14: 4, 15: 5, 16: 6, 17: 7, 18: 8, 19: 9, 21: 10, 22: 11, 23: 12, 24: 13, 25: 14, 26: 15, 27: 16, 28: 17, 29: 18, 30: 19, 32: 20, 34: 21, 37: 22, 39: 23, 41: 24, 43: 25, 45: 26, 47: 27, 49: 28, 51: 29, 52: 30, 53: 31, 54: 32, 55: 33, 56: 34, 57: 35, 58: 36, 59: 37, 71: 38, 73: 39, 75: 40, 77: 41, 83: 42, 85: 43, 87: 44, 89: 45}
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
        img_name = os.path.join('../../Rice2/gs_img/', self.dataframe.iloc[idx, 0] + '.jpeg')
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


def augment_data(df, label_column=None, min_images=100, transform=None, is_unlabeled=False):
    augment_rows = []
    
    if is_unlabeled:
        for _, row in df.iterrows():
            augment_image_name = row[0]
            img_path = os.path.join('../../Rice2/gs_img/', augment_image_name + '.jpeg')
            for _ in range(4):
                image = default_loader(img_path)
                if transform:
                    image = transform(image)
                augment_rows.append((augment_image_name, None))  # 无标签数据集只需返回图像和UUID
    else:
        label_counts = Counter(df[label_column])
        for label, count in label_counts.items():
            if count < min_images:
                augment_count = min_images - count
                label_df = df[df[label_column] == label]

                for _ in range(augment_count):
                    sample = label_df.sample(1).iloc[0]
                    augment_image_name = sample.iloc[0]
                    img_path = os.path.join('../../Rice2/gs_img/', augment_image_name + '.jpeg')
                    image = default_loader(img_path)

                    if transform:
                        image = transform(image)

                    augment_rows.append((augment_image_name, label))

    augment_df = pd.DataFrame(augment_rows, columns=df.columns)
    df = pd.concat([df, augment_df], ignore_index=True)
    return df


def load_data(args):
    csv_file = './clean_data.csv'
    df = pd.read_csv(csv_file)


    # 确保每个类别都划分30%数据到无标签数据集中
    labeled_df, unlabeled_df = train_test_split(df, test_size=0.3, random_state=36, stratify=df['growth_stage_code'])
    # 将有标签的数据集划分为训练集、验证集和测试集
    train_df, test_val_df = train_test_split(labeled_df, test_size=0.3, random_state=36, stratify=labeled_df['growth_stage_code'])
    val_df, test_df = train_test_split(test_val_df, test_size=0.5, random_state=36, stratify=test_val_df['growth_stage_code'])

    transform_augment = transforms.Compose([
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))], p=0.5),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
        transforms.RandomApply([transforms.Grayscale(num_output_channels=3)], p=0.1),
        transforms.ToTensor()
    ])
    train_df = augment_data(train_df, label_column='growth_stage_code', min_images=1500, transform=transform_augment)
    # 对无标签数据进行扩增，不需要使用min_images
    unlabeled_df = augment_data(unlabeled_df, transform=transform_augment, is_unlabeled=True)

    transform_train = transforms.Compose([
        transforms.Resize(args.input_size),
        transforms.RandomCrop(224, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
    ])


    transform_unlabeled = transform_train  # 使用相同的数据增强策略


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
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=args.batch_size, shuffle=False)
    



    # 统计训练集中各个标签的图像数量
    train_label_count = count_labels(train_loader.dataset)
    print("训练集中各个标签的图像数量：", train_label_count)

    # 统计验证集中各个标签的图像数量
    val_label_count = count_labels(val_loader.dataset)
    print("验证集中各个标签的图像数量：", val_label_count)

    # 统计测试集中各个标签的图像数量
    test_label_count = count_labels(test_loader.dataset)
    print("测试集中各个标签的图像数量：", test_label_count)

    # 统计无标签数据集中图像的数量
    unlabeled_sample_count = len(unlabeled_loader.dataset)
    print("无标签数据集中图像的数量：", unlabeled_sample_count)

    return train_loader, val_loader, test_loader, unlabeled_loader