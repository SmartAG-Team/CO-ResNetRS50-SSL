import sys
import os
import torch
import torch.nn as nn
import argparse
from tqdm import tqdm
# 获取当前文件夹的路径并添加上级文件夹
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.ResNetRS50.ResNetRS import ResNet
from preprocess import load_data
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device)     

def get_args():
    parser = argparse.ArgumentParser('parameters')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size, (default: 32)')
    parser.add_argument('--input-size', type=tuple, default=(224, 224), help='input data size, (default: (224, 224))')
    args = parser.parse_args()
    return args

def main(args):

    train_loader, val_loader, test_loader, unlabeled_loader = load_data(args)
    model = ResNet.build_model("resnetrs50").to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    # 测试模型
    val_checkpoint = torch.load('./model/ResNetRS50_SSL/ssl_90/Best_model_ckpt.t7')
    model.load_state_dict(val_checkpoint['model'])
    val_epoch = val_checkpoint['epoch']
    val_acc = val_checkpoint['acc']
    print("Load Validation Model Accuracy: ", val_acc, "epoch: ", val_epoch)
    model.eval()
    test_loss = 0
    test_correct = 0
    processed_data = 0
    all_preds = []
    all_targets = []
    progress_bar = tqdm(test_loader, desc="Test", mininterval=1)
    with torch.no_grad():
        for data, target in progress_bar:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            test_correct += pred.eq(target.view_as(pred)).sum().item()
            processed_data += data.size(0)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            average_test_loss = test_loss / processed_data
            average_test_accuracy = test_correct / processed_data
            progress_bar.set_description(f"Loss: {average_test_loss:.4f}, Acc: {average_test_accuracy:.4f}")

    conf_matrix = confusion_matrix(all_targets, all_preds)
    print("Confusion Matrix:")
    print(conf_matrix)

    # 计算Precision、Recall、F1 Score
    average_test_precision = precision_score(all_targets, all_preds, average='macro')
    average_test_recall = recall_score(all_targets, all_preds, average='macro')
    average_test_f1 = f1_score(all_targets, all_preds, average='macro')
    print(f"Test Precision: {average_test_precision:.4f}, Test Recall: {average_test_recall:.4f}, Test F1 Score: {average_test_f1:.4f}")
    # 计算每个类别的准确率
    accuracy_per_class = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
    for i in range(len(accuracy_per_class)):
        print(f"Class {i}: Accuracy: {accuracy_per_class[i]:.4f}")

if __name__ == '__main__':
    args = get_args()
    main(args)