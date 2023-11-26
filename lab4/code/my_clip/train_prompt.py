import numpy as np
import sys
sys.path.append('..')
import torch
import coop
from PIL import Image
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

# 设置随机种子
random_seed = 10
random.seed(random_seed)

def train(root_folder, epochs, model_name="RN50"):
    
    # 选择设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device is {device}")

    shot = 4
    M = 16
    
    # 获取所有子文件夹名称（即图片标签）
    subfolders = [f.path for f in os.scandir(root_folder) if f.is_dir()]

    # 加载预训练的CLIP模型
    model, preprocess = coop.load(model_name, device=device)

    # 类别集合
    text_class = []
    for folder_path in subfolders:
        class_name = os.path.basename(folder_path)
        # prompt_class_name = "a photo of a " + class_name
        prompt_class_name = class_name
        text_class.append(prompt_class_name)
    # 类别标签
    text = coop.tokenize(text_class).to(device)

    text_id = 0 # 初始化文件夹id
    total_num = 0 # 初始化图片数量
    target = torch.zeros(101*shot, dtype=torch.long) # 图片类别标签
    images = [] # 图片

    # 构建训练数据集
    for folder_path in subfolders:
        # 取出文件夹图片
        image_files = [f.path for f in os.scandir(folder_path) if f.is_file() and f.name.endswith(('.jpg', '.jpeg', '.png'))]
        if not image_files:
            print(f"No image files found in {folder_path}")
            continue
        
        # 获取shot个不同的随机索引
        random_indices = random.sample(range(len(image_files)), shot)
        for i in random_indices:
            image_path = image_files[i]
            image = preprocess(Image.open(image_path))
            images.append(image)
            target[total_num] = text_id
            total_num += 1
        text_id +=1

    targets = target.to(device)
    images = torch.stack(images).to(device)

    print(target)
    print(target.shape)
    print(images.shape)
    
    for param in model.parameters(): 
        param.requires_grad = False # 不计算梯度
    model.prompt.requires_grad = True # 只计算与prompt相关的梯度

    optimizer = optim.SGD(model.parameters(), lr=0.002) # 初始学习率
    # optimizer = optim.Adam(model.parameters(), lr=0.02)
    scheduler = CosineAnnealingLR(optimizer, epochs) # 余弦退火
    warmup_epochs = 1
    warmup_lr = 1e-5 # warmup技巧 
    warmup_optimizer = optim.SGD(model.parameters(), lr=warmup_lr)

    for epoch in range(epochs):
        if epoch < warmup_epochs:
            print("warmup")
            logits_per_image, _ = model(images, text)
            probs = logits_per_image.softmax(dim=-1)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(probs, targets) 
            warmup_optimizer.zero_grad()
            
            loss.backward()

            warmup_optimizer.step()
        else:
            logits_per_image, _ = model(images, text)
            # probs = logits_per_image.softmax(dim=-1)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(logits_per_image, targets) 
            optimizer.zero_grad()
            
            loss.backward()

            optimizer.step()

            scheduler.step() # 下一个学习率
            
            if epoch%10 == 0:
                print(f"epoch: {epoch}, loss: {loss}")   
    return M, shot, model, preprocess

def test(root_folder, epochs, M, shot, model, preprocess):
    print('start test')

    # 获取所有子文件夹名称（即图片标签）
    subfolders = [f.path for f in os.scandir(root_folder) if f.is_dir()]

    # 类别集合
    text_class = []
    for folder_path in subfolders:
        class_name = os.path.basename(folder_path)
        # prompt_class_name = "a photo of a " + class_name
        prompt_class_name = class_name
        text_class.append(prompt_class_name)

    text_id = 0 # 初始化文件夹id
    success_num = 0
    total_num = 0

    for folder_path in subfolders:
        # # 获取子文件夹名称作为类别名称
        # class_name = os.path.basename(folder_path)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # 获取子文件夹中的图像文件列表
        image_files = [f.path for f in os.scandir(folder_path) if f.is_file() and f.name.endswith(('.jpg', '.jpeg', '.png'))]

        if not image_files:
            print(f"No image files found in {folder_path}")
            continue

        # 对每张图像进行零样本分类
        for image_path in image_files:
            total_num += 1

            # 预处理图像
            image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

            # 处理文本标签
            text = coop.tokenize(text_class).to(device)

            # 模型推理
            with torch.no_grad():
                logits_per_image, logits_per_text = model(image, text)
                probs = logits_per_image.softmax(dim=-1).cpu().numpy()
            
            tids = np.argmax(probs, axis=1) # 获取每张图片最大概率文本索引
            tid = tids[0] # 第一张图片最大概率文本索引

            if tid == text_id:
                success = True
                success_num += 1
            else:
                success = False

            # # 输出结果
            # print(f"Predicted Class: {text_class[tid]}, Probability: {probs[0, tid]:.4f}, Success: {success}")
        
        text_id = text_id + 1 # 表示当前图片的ground truth
        torch.save(model.state_dict(), 'model.pth')
    print(f"success rate of {M}-embeddings {shot}-shot with {epochs}epochs: {success_num/total_num}")

if __name__ == '__main__':
    train_folder = "../data/caltech-101/101_ObjectCategories_divided/train"
    test_folder = "../data/caltech-101/101_ObjectCategories_divided/test"
    epochs = 200

    M, shot, model, preprocess = train(train_folder, epochs)
    test(test_folder, epochs, M, shot, model, preprocess)
