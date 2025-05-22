from matplotlib import pyplot as plt


def print_confusion_matrix(confusion_matrix_data):
    """
    打印混淆矩阵的终端输出

    Parameters:
        confusion_matrix_data (numpy.ndarray): 混淆矩阵数据
    """
    # 获取混淆矩阵的行和列数
    num_classes = len(confusion_matrix_data)
    classes = range(1, num_classes + 1)

    # 打印混淆矩阵表头
    # print(f"+{'Confusion Matrix':<15}+----------+----------+----------+")
    print(f"+{'Confusion Matrix':<15}", end='')
    print_string = '+----------'
    print(str(print_string)*num_classes, end='')
    print('+')
    print('|                |', end='')
    for cls in classes:
        print(f'{cls:^10}|', end='')
    print()
    print(f"+{'---------------':<15}-", end='')
    print(str(print_string)*num_classes, end='')
    print('+')

    # 打印混淆矩阵内容
    for i in range(num_classes):
        print(f"|       {classes[i]:<9}|", end="")
        for j in range(num_classes):
            print(f"{confusion_matrix_data[i, j]:^10}|", end="")
        print()
        print(f"+{'---------------':<15}-", end='')
        print(str(print_string)*num_classes, end='')
        print('+')


def plot_history(history, epoch, directory, train_name):
    """
    将训练过程中的loss、accuracy实时可视化记录

    Parameters:
        history:    训练数据记录字典
        epoch:      当前epoch
        directory:  保存可视化图片的路径
        train_name: 当前训练名称
    """
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.grid()
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.grid()
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.savefig(f'{directory}/{train_name}_results.png')
    plt.close()


def plot_lr(history, epoch, directory, train_name):
    """
    绘制训练过程中学习率-epoch的可视化曲线

    Parameters:
        history: 训练数据记录字典
        epoch: 当前epoch
        directory: 保存可视化图片的路径
        train_name: 当前训练名称
    """
    plt.figure(figsize=(12, 5))
    
    plt.grid()
    plt.plot(history['lr'], label='Learning rate')
    plt.title('Learning Rate Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.savefig(f'{directory}/{train_name}_lr.png')
    plt.close()


def plot_mae_loss(loss, directory, train_name):
    """
    绘制训练过程中学习率-epoch的可视化曲线

    Parameters:
        history: 训练数据记录字典
        epoch: 当前epoch
        directory: 保存可视化图片的路径
        train_name: 当前训练名称
    """
    plt.figure(figsize=(12, 5))
    
    plt.grid()
    plt.plot(loss, label='MAE loss')
    plt.title('MAE loss over iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()

    plt.savefig(f'{directory}/{train_name}_mae.png')
    plt.close()
