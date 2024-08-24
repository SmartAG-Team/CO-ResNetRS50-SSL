import torch
import torch.optim as optim
import torch.nn as nn
import cv2
import numpy as np
import os
import glob
import argparse
import json
import random
import codecs
import time

from model import resnet50
# from model.ResNetRS50.ResNetRS import ResNet
from model.CO_ResNetRS50.ResNetRS import ResNet
# from model.VGG import vgg
# from torchvision.models.shufflenetv2 import shufflenet_v2_x1_0


use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')


def get_args():
    parser = argparse.ArgumentParser('parameters')

    parser.add_argument('--epochs', type=int, default=100, help='number of epochs, (default: 100)')
    parser.add_argument('--learning-rate', type=float, default=0.01, help='learning rate, (default: 1e-1)')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size, (default: 100)')
    parser.add_argument('--input-size', type=tuple, default=(32, 32), help='input data size, (default: (32, 32))')
    args = parser.parse_args()

    return args


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.learning_rate * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def unlabeled_batch_iterator(batch_size=32, shape=(32, 32)):
    with codecs.open("./selected_image.json", "r", encoding="utf-8", errors="ignore") as f:
        json_data = json.load(f)

    image_list = json_data["all"]
    random.shuffle(image_list)
    while len(image_list) != 0:
        batch_keys = image_list[:batch_size]

        images = []
        labels = []

        for key in batch_keys:
            image = cv2.imread(key[0])
            image = cv2.resize(image, dsize=shape)

            images.append(image)
            labels.append(key[1])

        images = np.array(images)
        images = np.reshape(images, newshape=[-1, 3, 224, 224])
        labels = np.array(labels)
        yield images, labels

        del image_list[:batch_size]

def train(model, optimizer, criterion, epoch, args):
    model.train()  # 设置模型为训练模式
    train_loss = 0  # 用于累积训练损失
    train_acc = 0  # 用于累积训练准确率
    total_batches = 0  # 批次计数器，用于记录处理了多少批次

    # unlabeled_batch_iterator 是一个生成器函数，每次调用会返回一个批次的数据
    for batch_image, batch_label in unlabeled_batch_iterator(args.batch_size, args.input_size):
        adjust_learning_rate(optimizer, epoch, args)  # 根据epoch调整学习率
        data, target = torch.cuda.FloatTensor(batch_image), torch.cuda.LongTensor(batch_label)

        optimizer.zero_grad()  # 清空梯度
        output = model(data)  # 前向传播
        loss = criterion(output, target)  # 计算损失
        loss.backward()  # 反向传播，计算梯度
        optimizer.step()  # 更新参数

        train_loss += loss.item()  # 累加损失
        y_pred = output.data.max(1, keepdim=True)[1]  # 获取概率最高的预测结果

        acc = y_pred.eq(target.view_as(y_pred)).sum().item() / len(data)  # 计算批次准确率
        train_acc += acc  # 累加准确率
        total_batches += 1  # 更新批次计数器

    for param_group in optimizer.param_groups:
        print(f", 当前学习率为: {param_group['lr']}")

    avg_train_loss = train_loss / total_batches  # 计算平均损失
    avg_train_acc = train_acc / total_batches  # 计算平均准确率

    return avg_train_loss, avg_train_acc  # 返回平均损失和平均准确率


def main(args):
    # model = resnet50().to(device)
    model = ResNet.build_model("resnetrs50").to(device)
    # model = shufflenet_v2_x1_0(num_classes=7).to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=5e-4, momentum=0.9)
    criterion = nn.CrossEntropyLoss().to(device)

    start_time = time.time()
    max_train_acc = 0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train(model, optimizer, criterion, epoch, args)

        print('Epoch: {0:d},train set loss: {1:.4f}, accuracy: {2:.4f}, Best train accuracy: {3:.4f}'.format(epoch, train_loss, train_acc, max_train_acc))
        if max_train_acc < train_acc:
            print('Saving..')
            state = {
                'model': model.state_dict(),
                'acc': train_acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            filename = "student_network_Best_model_"
            torch.save(state, './checkpoint/' + filename + 'ckpt.t7')
            max_train_acc = train_acc

        time_interval = time.time() - start_time
        time_split = time.gmtime(time_interval)
        print("Training time: ", time_interval, "Hour: ", time_split.tm_hour, "Minute: ", time_split.tm_min, "Second: ",
              time_split.tm_sec)


if __name__ == "__main__":
    args = get_args()
    main(args)
