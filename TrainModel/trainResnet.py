import copy
from tqdm.rich import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, transforms
from MyResNet import MyResNet

# 数据路径
data_dir = './datasets/resnet'
train_dir = data_dir + '/train'
val_dir = data_dir + '/val'

# 图像预处理和数据增强
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize([224,224]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# 加载数据集
image_datasets = {'train': datasets.ImageFolder(train_dir, data_transforms['train']),
                  'val': datasets.ImageFolder(val_dir, data_transforms['val'])}
dataloaders = {'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=16, shuffle=True, num_workers=0),
               'val': torch.utils.data.DataLoader(image_datasets['val'], batch_size=4, shuffle=False, num_workers=0)}
dataset_sizes = {'train': len(image_datasets['train']), 'val': len(image_datasets['val'])}
print(dataset_sizes)
# 查看模型的分类，后续用于后端
classes = image_datasets['train'].classes
train_losses = []
val_losses = []
train_accs = []
val_accs = []

if __name__ == '__main__':
    model = MyResNet(4)
    # print(model)
    model.cpu()
    # -----定义优化器和Loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # -----训练--验证-----
    num_epochs = 20
    best_acc = 0
    best_loss = 9999
    for epoch in range(num_epochs):
        message = 'Epoch {}/{} '.format(epoch+1, num_epochs)
        # 一个epoch有train和val两个阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 设置模型为训练模式
            else:
                model.eval()  # 设置模型为评估模式

            running_loss = 0.0
            running_corrects = 0
            batch = 0
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 梯度清零
                optimizer.zero_grad()
                # 当训练时候才使能梯度
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 计算损失值和准确率
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                batch += 1
                # if batch%20==0:
                #     print(f'Epoch: {epoch},Batch: {batch},Loss:{loss.item()}')
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            message += ' {} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc)
            # print(message)
            if phase == 'train':
                train_losses.append(epoch_loss)
                train_accs.append(epoch_acc)
            else:
                val_losses.append(epoch_loss)
                val_accs.append(epoch_acc)
            # 深拷贝模型
            if phase == 'val' and (epoch_acc >= best_acc and epoch_loss < best_loss):
                best_acc = epoch_acc
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts,'models/bestResnet.pt')
        print(message)

# 保存loss曲线
import matplotlib.pyplot as plt
plt.plot(range(1, len(train_losses)+1),train_losses, label='train_loss')
plt.plot(range(1, len(val_losses)+1),val_losses, label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss.png')

# 保存acc曲线
plt.clf()
plt.plot(range(1, len(train_accs)+1),train_accs, label='train_acc')
plt.plot(range(1, len(val_accs)+1),val_accs, label='val_acc')
plt.xlabel('Epoch')
plt.ylabel('Acc')
plt.legend()
plt.savefig('acc.png')