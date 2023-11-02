# 训练代码
import torch
import os
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn as nn
from model import AlexNet
from datetime import datetime   
from torch.utils.tensorboard import SummaryWriter

def evaluate(model, dataloader, device, data_name, epoch):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1) # 第一个返回值为最大值，第二个为最大值索引
            total +=labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100*correct/total
    print(f"Accuracy on the {data_name}: {accuracy :.2f}%")
    writer.add_scalar(f'Accuracy/{data_name}', accuracy, epoch)

# dataset 路径
data_dir = './lab2/data/Caltech101'
# log_dir = './lab2/log'
log_dir = "B:/log"

# 格式化当前时间
current_time = datetime.now()
folder_name = current_time.strftime("%m-%d-%H-%M")

tensorboard_dir = os.path.join(log_dir, folder_name)
os.makedirs(tensorboard_dir, exist_ok=True)
writer = SummaryWriter(tensorboard_dir)

# 网络参数
batchsize = 16
lr = 0.0001
momentum = 0.9
epochs = 500
validation_frequency = 10

# resize图片，变为224*224大小；对训练集进行数据增强；将图像转为tensor
# transforms提供了各种图像数据预处理方法
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),  # 调整图像大小
        # transforms.RandomHorizontalFlip(), # 随机翻转
        transforms.ToTensor(),
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),  
        transforms.ToTensor(),
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
    ])
}


# 创建dataset和dataloader
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val', 'test']} # 以for循环的方式创建了一个字典image_dataset，字典中有三个键“train”，“val”，“test”，键值为dataset.ImageFolder
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batchsize, shuffle=True) for x in ['train', 'val', 'test']} # 为三个集创建dataloader

# 创建模型
model = AlexNet()

# 设置训练设备 
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("successfully use gpu")
else:
    print("pay attention, use cpu now")

model = model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum) # 随机梯度下降

# 训练
print("start train")
for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0
    for inputs, labels in dataloaders['train']: # inputs:(16,3,224,224) labels:(16)
        # print(inputs.shape, labels.shape)
        inputs, labels = inputs.to(device), labels.to(device) # 移入GPU
        optimizer.zero_grad() # 梯度归零
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward() 
        optimizer.step() # 更新参数
        epoch_loss += loss.item() * inputs.size(0)

    avg_loss = epoch_loss/len(image_datasets['train'])
    writer.add_scalar('Loss/train', avg_loss, epoch)
    print(f"Epoch: {epoch+1}   loss: {avg_loss}")

    if (epoch+1)%validation_frequency == 0:
        evaluate(model, dataloaders['train'], device, 'train', epoch)
        evaluate(model, dataloaders['val'], device, 'validation', epoch)

evaluate(model, dataloaders['test'], device, 'test', epoch)
