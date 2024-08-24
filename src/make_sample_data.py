import torch
import cv2
import numpy as np
import os
import glob
import argparse
import json
import codecs
import random
import pandas as pd
import torchvision.transforms as transforms
from PIL import Image

# from model import resnet50
from model.VGG import vgg
# from model.ResNetRS50.ResNetRS import ResNet
from model.CO_ResNetRS50.ResNetRS import ResNet
# from torchvision.models.shufflenetv2 import shufflenet_v2_x1_0


use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')


def get_args():
    parser = argparse.ArgumentParser("parameters")
    parser.add_argument('--k', type=int, default=3000, help=', there is an important trade-off on K that strongly depends on the ratio K/M. (default: 1000)')
    parser.add_argument('--load-pretrained', type=bool, default=True)
    args = parser.parse_args()
    return args



# 函数：处理图片并进行必要的转换
def process_image(image_path):
    # 定义转换操作 - 根据你的模型训练代码可能需要调整参数
    image_transforms = transforms.Compose([
        transforms.Resize((32, 32)),  # 尺寸应与模型训练时期望的输入尺寸匹配
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))# 归一化参数应与训练时使用的一致
    ])
    # 加载图片
    image = Image.open(image_path).convert('RGB')
    # 应用转换操作
    image_tensor = image_transforms(image).unsqueeze(0)
    return image_tensor
# 函数：加载预训练模型
def load_pretrained_model(model_path):
    # 初始化模型
    # model = resnet50().to(device)
    # model = convnext_base(num_classes=7).to(device)
    model = ResNet.build_model("resnetrs50").to(device)
    # model = shufflenet_v2_x1_0(num_classes=7).to(device)
    # 加载保存的权重
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    epoch = checkpoint['epoch']
    acc = checkpoint['acc']
    print("Load Model Accuracy: ", acc, "Load Model end epoch: ", epoch)
    model.eval()  # 设置模型为评估模式
    return model

def predict_image(model, image_tensor):
    # 将图片数据发送到设备
    image_tensor = image_tensor.to(device)
    # 执行前向传播
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)  # 应用softmax函数获得概率分布
        max_prob, predicted_class = torch.max(probabilities, 1)  # 获取得分最高的预测类别及其概率
    return predicted_class.item(), max_prob.item()  # 返回类别和概率


def data_sampling():
    # 定义图片和模型的路径
    image_folder = '/home/qtian/Downloads/resnet_data/Rice2/gs_img_zengqiang/'  # 替换为你的图片文件夹路径
    model_path = './checkpoint/Best_model_ckpt.t7'  # 替换为你的模型检查点路径

    # 加载预训练模型
    model = load_pretrained_model(model_path)

    # 新增的采样字典
    sampling_dictionary = {}

    # 遍历文件夹中的所有图片
    for image_name in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_name)
        if image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            image_tensor = process_image(image_path)
            predicted_class, probability = predict_image(model, image_tensor)

            # 输出图片名称、预测的标签和概率
            print(f"Image: {image_name}, Predicted class: {predicted_class}, Probability: {probability:.4f}")

            # 更新采样字典
            if str(predicted_class) in sampling_dictionary:
                sampling_dictionary[str(predicted_class)].append([image_path, probability])
            else:
                sampling_dictionary[str(predicted_class)] = [[image_path, probability]]

    # 保存采样字典到JSON文件
    print("Saving sampling_dict.json")
    j = json.dumps(sampling_dictionary, indent=4)
    with open("sampling_dict.json", "w") as f:
        f.write(j)


def select_top_k(k=3000):
    sampled_image_dict = {}
    sampled_image_dict["all"] = []
    with codecs.open("./sampling_dict.json", "r", encoding="utf-8", errors="ignore") as f:
        load_data = json.load(f)

        for key in load_data.keys():
            print("label: ", key)
            all_items = load_data[key]
            all_items.sort(key=lambda x: x[1], reverse=True)
            all_items = np.array(all_items)
            print("each label item count: ", len(all_items))
            items_to_take = min(k, len(all_items))
            for index in range(items_to_take):
                sampled_image_dict["all"].append([all_items[index][0], int(key)])

    print("Saving.. selected image json")
    j = json.dumps(sampled_image_dict)
    with open("selected_image.json", "w") as f:
        f.write(j)


def main(args):
    if args.load_pretrained:
        data_sampling()
        select_top_k(args.k)
    else:
        assert args.load_pretrained == True, "You must have the weights of the pretrained model."


if __name__ == "__main__":
    args = get_args()
    main(args)
