import os
import torch
import torch.nn as nn
import argparse
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
# from torchvision.models.shufflenetv2 import shufflenet_v2_x1_0
# from model.ResNetRS50.ResNetRS import ResNet
# from model.VGG import vgg
# from model.AlexNet import AlexNet
# from model.EfficientNet import efficientnet_b0
from model.convnext import convnext_base

from preprocess import load_data
from torch.optim.lr_scheduler import _LRScheduler
from sklearn.metrics import confusion_matrix, classification_report
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device)

def get_args():
    parser = argparse.ArgumentParser('parameters')
    parser.add_argument('--epochs', type=int, default=80, help='number of epochs, (default: 100)')
    parser.add_argument('--num_classes', type=int, default=7, help='number of num_classes, (default: 1000)')
    parser.add_argument('--learning-rate', type=float, default=0.01, help='learning rate, (default: 1e-1)')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size, (default: 100)')
    parser.add_argument('--input-size', type=tuple, default=(32, 32), help='input data size, (default: (32, 32))')
    parser.add_argument('--student-network', type=bool, default=False, help='if proceeding with step 1 (False), if proceeding with step 4.(True), (default: False)')
    args = parser.parse_args()

    return args



class WarmUpLR(_LRScheduler):
    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)
    def get_lr(self):
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.learning_rate * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def main(args):
    train_loader, val_loader, test_loader = load_data(args)

    # model = resnet50(num_classes=args.num_classes).to(device)
    # model = ResnetRS.create_model(block=Bottleneck, layers=[3, 4, 6, 3], num_classes=7).to(device)
    # model=ResNet.build_model("resnetrs50").to(device)
    # model = shufflenet_v2_x1_0(num_classes=7).to(device)
    # model = vgg(model_name="vgg16").to(device)
    # model = AlexNet(num_classes=7).to(device)
    # model = efficientnet_b0(num_classes=7).to(device)
    model = convnext_base(num_classes=7).to(device)

    if args.student_network:
        filename = "student_network_Best_model_"
        checkpoint = torch.load('../data/ConvNext/checkpoint/' + filename + 'ckpt.t7')
        model.load_state_dict(checkpoint['model'])
        epoch = checkpoint['epoch']
        acc = checkpoint['acc']
        print("Load Model Accuracy: ", acc, "Load Model end epoch: ", epoch)

    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=5e-4, momentum=0.9)
    # optimizer = optim.NAdam(model.parameters(), lr=0.001, weight_decay=5e-4, momentum_decay=0.004)
    # optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    # optimizer = optim.RAdam(model.parameters(), lr=0.0001, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss().to(device)

    print(f"Total parameters in the model: {count_parameters(model):,}")


    best_val_accuracy = 0.0  # 保存最佳验证准确率
    epoch_list = []
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    val_precisions = []
    val_recalls = []
    val_f1_scores = []
    # 新增：创建保存损失和准确率的txt文件
    with open("../data/ConvNext/result.txt", "w") as f:
        for epoch in range(1, args.epochs + 1):
            epoch_list.append(epoch)
            model.train()
            train_loss = 0
            train_correct = 0
            processed_data = 0
            progress_bar = tqdm(train_loader, desc="Training", mininterval=1)
            for batch_idx, (data, target) in enumerate(progress_bar):
                adjust_learning_rate(optimizer, epoch, args)
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * data.size(0)
                y_pred = output.argmax(dim=1, keepdim=True)
                train_correct += y_pred.eq(target.view_as(y_pred)).sum().item()
                processed_data += data.size(0)
                average_train_loss = train_loss / processed_data
                average_train_accuracy = train_correct / processed_data
                progress_bar.set_description(f"Epoch {epoch:02d}, Loss: {average_train_loss:.4f}, Acc: {average_train_accuracy:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f},")
            train_losses.append(average_train_loss)
            train_accuracies.append(average_train_accuracy)


            # 验证模型
            model.eval()
            val_loss = 0
            val_correct = 0
            processed_data = 0
            all_preds = []
            all_targets = []
            progress_bar = tqdm(test_loader, desc="Validation", mininterval=1)
            with torch.no_grad():
                for data, target in progress_bar:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    loss = criterion(output, target)
                    val_loss += loss.item() * data.size(0)
                    pred = output.argmax(dim=1, keepdim=True)
                    val_correct += pred.eq(target.view_as(pred)).sum().item()
                    processed_data += data.size(0)
                    all_preds.extend(pred.cpu().numpy())
                    all_targets.extend(target.cpu().numpy())
                    average_val_loss = val_loss / processed_data
                    average_val_accuracy = val_correct / processed_data
                    progress_bar.set_description(f"Val Loss: {average_val_loss:.4f}, Val Accuracy: {average_val_accuracy:.4f}")

            conf_matrix = confusion_matrix(all_targets, all_preds)
            print("Confusion Matrix:")
            print(conf_matrix)
            class_report = classification_report(all_targets, all_preds)
            print("Classification Report:")
            print(class_report)
            class_accuracies = [conf_matrix[i, i] / conf_matrix[i, :].sum() for i in range(conf_matrix.shape[0])]
            for i, acc in enumerate(class_accuracies):
                print(f"Class {i} Accuracy {acc:.4f}")

            val_losses.append(average_val_loss)
            val_accuracies.append(average_val_accuracy)
            # 新增：将训练集、验证集和测试集的损失和准确率写入文件
            f.write(f"Epoch {epoch:02d}, Train Loss: {average_train_loss:.4f}, Train Acc: {average_train_accuracy:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}\n")
            f.write(f"Epoch {epoch:02d}, Val Loss: {average_val_loss:.4f}, Val Acc: {average_val_accuracy:.4f}\n")
            f.write(f"Epoch {epoch:02d}, Confusion Matrix:\n")
            f.write(f"{conf_matrix}\n")
            f.write(f"Epoch {epoch:02d}, Classification Report:\n")
            f.write(f"{class_report}\n")
            for i, acc in enumerate(class_accuracies):
                f.write(f"Class {i} Accuracy {acc:.4f}")
            f.write("\n")

            if best_val_accuracy < average_val_accuracy:
                print('Saving..')
                state = {
                    'model': model.state_dict(),
                    'loss': val_loss,
                    'acc': average_val_accuracy,
                    'epoch': epoch,
                }
                if not os.path.isdir('checkpoint'):
                    os.mkdir('checkpoint')
                filename = "Best_model_"
                torch.save(state, '../data/ConvNext/checkpoint/' + filename + 'ckpt.t7')
                best_val_accuracy = average_val_accuracy

        # 测试模型
        val_checkpoint = torch.load('../data/ConvNext/checkpoint/Best_model_ckpt.t7')
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
        class_report = classification_report(all_targets, all_preds)
        print("Classification Report:")
        print(class_report)
        class_accuracies = [conf_matrix[i, i] / conf_matrix[i, :].sum() for i in range(conf_matrix.shape[0])]
        for i, acc in enumerate(class_accuracies):
            print(f"Class {i} Accuracy {acc:.4f}")


        f.write(f"Test Loss: {average_test_loss:.4f}, Test Acc: {average_test_accuracy:.4f}\n")
        f.write("Confusion Matrix:\n")
        f.write(f"{conf_matrix}\n")
        f.write("Classification Report:\n")
        f.write(f"{class_report}\n")
        for i, acc in enumerate(class_accuracies):
            f.write(f"Class {i} Accuracy {acc:.4f}")
        f.write("\n")
        f.flush()

if __name__ == '__main__':
    args = get_args()
    main(args)
