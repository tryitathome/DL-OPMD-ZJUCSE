import sys
import os
# 将当前脚本所在目录添加到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time
from tqdm import tqdm
from terminaltables import AsciiTable
import torch

from eval_metrics import calculate_confusion_matrix, calculate_precision_recall_f1
from train_vis import print_confusion_matrix

from apex import amp



'''基本单epoch训练'''
def train_epoch(model, loader, device, schedular, criterion, optimizer, epoch, total_epochs, mixup_fn, history, average_mode, thrs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    pred_list, target_list = [], []
    start_time = time.time()
    for idx, (images, labels) in enumerate(tqdm(loader, desc=f'Epoch {epoch+1}/{total_epochs} - Training')):
        images = images.to(device=device, non_blocking=True)
        labels = labels.to(device=device, non_blocking=True)

        outputs = model(images)

        loss = criterion(outputs, labels)

        pred_list.append(outputs.data)
        target_list.append(labels)
        
        optimizer.zero_grad()

        # 更新学习率
        num_steps = len(loader)
        schedular.step_update(epoch * num_steps + idx)

        loss.backward()

        # 梯度裁剪
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5, norm_type=2)

        optimizer.step()
        total_loss += loss.item()

    _, predicted = torch.max(outputs.data, 1)
    print(outputs.data.shape)
    print(predicted.shape)
    print(labels.shape)
    correct += (predicted == labels).sum().item()
    total += labels.size(0)

    average_loss = total_loss / len(loader)
    accuracy = 100 * correct / total

    # 计算其他指标
    precision, recall, f1_score = calculate_precision_recall_f1(torch.cat(pred_list), 
                                                                torch.cat(target_list), 
                                                                average_mode=average_mode,
                                                                thrs=thrs)

    # 保存历史数据
    history['train_loss'].append(average_loss)
    history['train_acc'].append(accuracy)
    history['train_precision'].append(precision)
    history['train_recall'].append(recall)
    history['train_f1_score'].append(f1_score)

    TITLE = 'Train Results'
    TABLE_DATA = (
        ('Top-1 Acc', 'Mean Precision', 'Mean Recall', 'Mean F1 Score'),
        ('{:.2f}'.format(accuracy), 
                         '{:.2f}'.format(precision), 
                         '{:.2f}'.format(recall),
                         '{:.2f}'.format(f1_score)
        )
    )
    table_instance = AsciiTable(TABLE_DATA, TITLE)
    print(table_instance.table)

    # 计算混淆矩阵
    confusion_matrix_data = calculate_confusion_matrix(torch.cat(pred_list), torch.cat(target_list))
    print_confusion_matrix(confusion_matrix_data)
    print(f'Train Loss: {average_loss:.4f}, Train Accuracy: {accuracy:.2f}%')

    # 同步
    torch.cuda.synchronize()

    # 计算训练时间
    end_time = time.time() 
    train_time = end_time - start_time
    print(f'Epoch {epoch+1} Train Time: {train_time:.2f}')
    print()
    return 


'''带有mixup的单epoch训练, 无法即时算出accuracy'''
def train_epoch_mixup(model, loader, device, schedular, criterion, optimizer, epoch, total_epochs, mixup_fn, accumulation_steps, clip_grad, amp_level, history):
    model.train()

    total_loss = 0
    num_steps = len(loader)

    start_time = time.time()
    for idx, (images, labels) in enumerate(tqdm(loader, desc=f'Epoch {epoch+1}/{total_epochs} - Training')):

        images = images.to(device=device, non_blocking=True)
        labels = labels.to(device=device, non_blocking=True)

        # 进行mixup
        if mixup_fn != None:
            images, labels = mixup_fn(images, labels)
        
        outputs = model(images)

        # 如果需要进行梯度累积
        if accumulation_steps > 1:
            loss = criterion(outputs, labels)
            loss = loss / accumulation_steps
            if amp_level != "O0":
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                if clip_grad:
                    grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), clip_grad)
                else:
                    grad_norm = get_grad_norm(amp.master_params(optimizer))
            else:
                loss.backward()
                total_loss += loss.item()
                if clip_grad:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                else:
                    grad_norm = get_grad_norm(model.parameters())
            if (idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                schedular.step_update(epoch * num_steps + idx)
        else:
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            total_loss += loss.item()
            if amp_level != "O0":
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                if clip_grad:
                    grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), clip_grad)
                else:
                    grad_norm = get_grad_norm(amp.master_params(optimizer))
            else:
                loss.backward()
                if clip_grad:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                else:
                    grad_norm = get_grad_norm(model.parameters())
            optimizer.step()
            schedular.step_update(epoch * num_steps + idx)
            
    average_loss = total_loss / len(loader)

    # 显存占用
    memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
    print(f'Epoch {epoch+1} Memory Used: {memory_used:.0f}MB')

    # 保存历史数据
    history['train_loss'].append(average_loss)
    print(f'Epoch {epoch+1} Train Loss: {average_loss:.4f}')

    # 同步
    torch.cuda.synchronize()

    # 计算训练时间
    end_time = time.time() 
    train_time = end_time - start_time
    print(f'Epoch {epoch+1} Train Time: {train_time:.2f}')
    print()
    return 


'''基本验证'''
def evaluate(model, loader, device, criterion, epoch, history, average_mode, thrs, description):
    
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    total_loss = 0
    correct = 0
    total = 0
    pred_list, target_list = [], []
    start_time = time.time()
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=description):
            images = images.to(device) 
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)

            pred_list.append(outputs.data)
            target_list.append(labels)

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    average_loss = total_loss / len(loader)
    accuracy = 100 * correct / total

    # 计算其他指标
    precision, recall, f1_score = calculate_precision_recall_f1(torch.cat(pred_list), 
                                                                torch.cat(target_list), 
                                                                average_mode=average_mode,
                                                                thrs=thrs)

    # 保存历史数据
    history['val_loss'].append(average_loss)
    history['val_acc'].append(accuracy)
    history['val_precision'].append(precision)
    history['val_recall'].append(recall)
    history['val_f1_score'].append(f1_score)

    TITLE = 'Validation Results'
    TABLE_DATA = (
        ('Top-1 Acc', 'Mean Precision', 'Mean Recall', 'Mean F1 Score'),
        ('{:.2f}'.format(accuracy), 
                         '{:.2f}'.format(precision), 
                         '{:.2f}'.format(recall),
                         '{:.2f}'.format(f1_score)
        )
    )
    table_instance = AsciiTable(TABLE_DATA, TITLE)
    print(table_instance.table)

    # 计算混淆矩阵
    confusion_matrix_data = calculate_confusion_matrix(torch.cat(pred_list), torch.cat(target_list))
    print_confusion_matrix(confusion_matrix_data)
    # print(f'{description} - Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%')
    print(f'Validation Loss: {loss:.4f}, Validation Accuracy: {accuracy:.2f}%')

    # 同步
    torch.cuda.synchronize()

    # 打印验证时间
    end_time = time.time()
    val_time = end_time - start_time
    if epoch >= 0:
        print(f'Epoch {epoch+1} Validation Time: {val_time:.2f}')
    else:
        # print(f'Test Validation Time: {val_time:.2f}')
        pass
    return accuracy


'''测试'''
def test(model, loader, device, criterion, epoch, history, average_mode, thrs, description):
    
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    total_loss = 0
    correct = 0
    total = 0
    pred_list, target_list = [], []
    start_time = time.time()
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=description):
            images = images.to(device) 
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)

            pred_list.append(outputs.data)
            target_list.append(labels)

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    average_loss = total_loss / len(loader)
    accuracy = 100 * correct / total

    # 计算其他指标
    precision, recall, f1_score = calculate_precision_recall_f1(torch.cat(pred_list), 
                                                                torch.cat(target_list), 
                                                                average_mode=average_mode,
                                                                thrs=thrs)

    # 保存历史数据
    history['val_loss'].append(average_loss)
    history['val_acc'].append(accuracy)
    history['val_precision'].append(precision)
    history['val_recall'].append(recall)
    history['val_f1_score'].append(f1_score)

    TITLE = 'Validation Results'
    TABLE_DATA = (
        ('Top-1 Acc', 'Mean Precision', 'Mean Recall', 'Mean F1 Score'),
        ('{:.2f}'.format(accuracy), 
                         '{:.2f}'.format(precision), 
                         '{:.2f}'.format(recall),
                         '{:.2f}'.format(f1_score)
        )
    )
    table_instance = AsciiTable(TABLE_DATA, TITLE)
    print(table_instance.table)

    # 计算混淆矩阵
    confusion_matrix_data = calculate_confusion_matrix(torch.cat(pred_list), torch.cat(target_list))
    print_confusion_matrix(confusion_matrix_data)
    # print(f'{description} - Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%')
    print(f'Validation Loss: {loss:.4f}, Validation Accuracy: {accuracy:.2f}%')

    # 同步
    torch.cuda.synchronize()

    # 打印验证时间
    end_time = time.time()
    val_time = end_time - start_time
    if epoch >= 0:
        print(f'Epoch {epoch+1} Validation Time: {val_time:.2f}')
    else:
        # print(f'Test Validation Time: {val_time:.2f}')
        pass
    return accuracy, precision, recall, f1_score


'''基本验证'''
def evaluate_imagenet_pretrain(model, loader, device, criterion, epoch, history, average_mode, thrs, description):
    
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    total_loss = 0
    correct = 0
    total = 0
    pred_list, target_list = [], []
    start_time = time.time()
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=description):
            images = images.to(device) 
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)

            pred_list.append(outputs.data)
            target_list.append(labels)

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    average_loss = total_loss / len(loader)
    accuracy = 100 * correct / total

    # 计算其他指标
    precision, recall, f1_score = calculate_precision_recall_f1(torch.cat(pred_list), 
                                                                torch.cat(target_list), 
                                                                average_mode=average_mode,
                                                                thrs=thrs)

    # 保存历史数据
    history['val_loss'].append(average_loss)
    history['val_acc'].append(accuracy)
    history['val_precision'].append(precision)
    history['val_recall'].append(recall)
    history['val_f1_score'].append(f1_score)

    TITLE = 'Validation Results'
    TABLE_DATA = (
        ('Top-1 Acc', 'Mean Precision', 'Mean Recall', 'Mean F1 Score'),
        ('{:.2f}'.format(accuracy), 
                         '{:.2f}'.format(precision), 
                         '{:.2f}'.format(recall),
                         '{:.2f}'.format(f1_score)
        )
    )
    table_instance = AsciiTable(TABLE_DATA, TITLE)
    print(table_instance.table)

    # 计算混淆矩阵
    confusion_matrix_data = calculate_confusion_matrix(torch.cat(pred_list), torch.cat(target_list))
    # print_confusion_matrix(confusion_matrix_data)
    # print(f'{description} - Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%')
    print(f'Validation Loss: {loss:.4f}, Validation Accuracy: {accuracy:.2f}%')

    # 同步
    torch.cuda.synchronize()

    # 打印验证时间
    end_time = time.time()
    val_time = end_time - start_time
    print(f'Epoch {epoch+1} Validation Time: {val_time:.2f}')
    return accuracy


'''获取梯度'''
def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm