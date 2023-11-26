import numpy as np
import sys
sys.path.append('..')
import torch
import clip
from PIL import Image
import os

def zero_shot_classification_folder(root_folder, model_name="RN50"):
    # 获取所有子文件夹名称（即图片标签）
    subfolders = [f.path for f in os.scandir(root_folder) if f.is_dir()]

    # 选择设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device is {device}")

    # 加载预训练的CLIP模型
    model, preprocess = clip.load(model_name, device=device)

    # 类别集合
    text_class = []
    for folder_path in subfolders:
        class_name = os.path.basename(folder_path)
        prompt_class_name = "a photo of a " + class_name
        # prompt_class_name = class_name
        text_class.append(prompt_class_name)

    text_id = 0 # 初始化文件夹id
    success_num = 0
    total_num = 0

    for folder_path in subfolders:
        # # 获取子文件夹名称作为类别名称
        # class_name = os.path.basename(folder_path)

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
            text = clip.tokenize(text_class).to(device)

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

    print(f"success rate: {success_num/total_num}")

if __name__ == '__main__':
    root_folder = "../data/caltech-101/101_ObjectCategories_divided/test"
    zero_shot_classification_folder(root_folder)
