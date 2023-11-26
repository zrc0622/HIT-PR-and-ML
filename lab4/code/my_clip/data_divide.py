# 对数据集进行划分
import os
import shutil
import math
import random

# 数据集路径
data_path = '../data/caltech-101/101_ObjectCategories'
output_path = '../data/caltech-101/101_ObjectCategories_divided'

# 创建训练集、验证集和测试集的文件夹
train_path = os.path.join(output_path, 'train')
val_path = os.path.join(output_path, 'val')
test_path = os.path.join(output_path, 'test')

os.makedirs(train_path, exist_ok=True)
os.makedirs(val_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)

# 定义划分比例
train_ratio = 0.3
val_ratio = 0.0
test_ratio = 0.7

# 划分函数
def split_data(class_dir):
    images = os.listdir(class_dir)
    images.sort()  # 按数字升序排列图片

    n = len(images)
    train_count = math.floor(train_ratio * n)
    val_count = math.floor((train_ratio + val_ratio) * n)

    train_images = images[:train_count]
    val_images = images[train_count:val_count]
    test_images = images[val_count:]

    return train_images, val_images, test_images

# 遍历数据集中的每个类别文件夹
for category in os.listdir(data_path):
    category_path = os.path.join(data_path, category)

    train_category_path = os.path.join(train_path, category)
    val_category_path = os.path.join(val_path, category)
    test_category_path = os.path.join(test_path, category)

    os.makedirs(train_category_path, exist_ok=True)
    os.makedirs(val_category_path, exist_ok=True)
    os.makedirs(test_category_path, exist_ok=True)

    train_images, val_images, test_images = split_data(category_path)

    # 将图片分配到训练集、验证集和测试集的相应文件夹中
    for img in train_images:
        shutil.copy(os.path.join(category_path, img), os.path.join(train_category_path, img))

    for img in val_images:
        shutil.copy(os.path.join(category_path, img), os.path.join(val_category_path, img))

    for img in test_images:
        shutil.copy(os.path.join(category_path, img), os.path.join(test_category_path, img))
