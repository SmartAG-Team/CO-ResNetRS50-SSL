import sys
import os
import torch
import torch.nn as nn
import argparse
import torch.optim as optim
import numpy as np
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.ResNetRS50.ResNetRS import ResNet
from preprocess import load_data
from torch.optim.lr_scheduler import _LRScheduler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device)

def get_args():
    parser = argparse.ArgumentParser('parameters')
    parser.add_argument('--epochs', type=int, default=80, help='number of epochs, (default: 80)')
    parser.add_argument('--num_classes', type=int, default=7, help='number of num_classes, (default: 7)')
    parser.add_argument('--learning-rate', type=float, default=0.01, help='learning rate, (default: 1e-1)')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size, (default: 32)')
    parser.add_argument('--input-size', type=tuple, default=(224, 224), help='input data size, (default: (224, 224))')
    args = parser.parse_args()
    return args


class WarmUpLR(_LRScheduler):
    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]

def adjust_learning_rate(optimizer, epoch, args):
    lr = args.learning_rate * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def main(args):
    train_loader, val_loader, test_loader, unlabeled_loader = load_data(args)

    model = ResNet.build_model("resnetrs50").to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=5e-4, momentum=0.9)
    criterion = nn.CrossEntropyLoss().to(device)

    print(f"Total parameters in the model: {count_parameters(model):,}")
 
    best_val_accuracy = 0.0
    # 新增：创建CSV文件并写入列标题
    with open("../data/ResNetRS50/result.csv", "w") as f:
        f.write("Epoch,Train_Loss,Train_Acc,Val_Loss,Val_Acc,Test_Loss,Test_Acc,Test_Pre,Test_Recall,Test_F1,Test_1_Acc,Test_2_Acc,Test_3_Acc,Test_4_Acc,Test_5_Acc,Test_7_Acc,Test_8_Acc,Test_Confusion_Matrix\n")
    
        for epoch in range(1, args.epochs + 1):
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


            # 验证模型
            model.eval()
            val_loss = 0
            val_correct = 0
            processed_data = 0
            all_preds = []
            all_targets = []
            progress_bar = tqdm(val_loader, desc="Validation", mininterval=1)
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


            # 将结果写入CSV文件
            if epoch < args.epochs:
                f.write(f"{epoch},{average_train_loss:.4f},{average_train_accuracy:.4f},{average_val_loss:.4f},{average_val_accuracy:.4f},{None},{None},{None},{None},{None},{None},{None},{None},{None},{None},{None},{None},{None}\n")

            if best_val_accuracy < average_val_accuracy:
                print('Saving..')
                state = {
                    'model': model.state_dict(),
                    'loss': val_loss,
                    'acc': average_val_accuracy,
                    'epoch': epoch,
                }
                filename = "Best_model_"
                torch.save(state, '../model/ResNetRS50/' + filename + 'ckpt.t7')
                best_val_accuracy = average_val_accuracy

        # 测试模型
        val_checkpoint = torch.load('../model/ResNetRS50/Best_model_ckpt.t7')
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

        # 将结果写入CSV文件
        conf_matrix_str = np.array_str(conf_matrix).replace('\n', ' ')
        f.write(f"{epoch},{average_train_loss:.4f},{average_train_accuracy:.4f},{average_val_loss:.4f},{average_val_accuracy:.4f},{average_test_loss:.4f},{average_test_accuracy:.4f},{average_test_precision:.4f},{average_test_recall:.4f},{average_test_f1:.4f},{accuracy_per_class[0]:.4f},{accuracy_per_class[1]:.4f},{accuracy_per_class[2]:.4f},{accuracy_per_class[3]:.4f},{accuracy_per_class[4]:.4f},{accuracy_per_class[5]:.4f},{accuracy_per_class[6]:.4f},{conf_matrix_str}\n")



if __name__ == '__main__':
    args = get_args()
    main(args)