# AlexNet 模型
import torch
import torch.nn as nn

'''
卷积层输出大小 = (输入图像大小 - 卷积核大小 + 2 * 填充数) ÷ 步幅大小 + 1
output.shape = (input.shape - kenal_size + 2 * padding) / stride + 1
'''

class AlexNet(nn.Module):
    def __init__(self, num_classes=102): # 输入(16,3,224,224)
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2), # ->(16,96,55,55)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2), # ->(16,96,27,27)
            nn.Conv2d(96, 256, kernel_size=5, padding=2), # ->(16,256,27,27)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2), # ->(16,256,13,13)
            nn.Conv2d(256, 384, kernel_size=3, padding=1), # ->(16,384,13,13)
            nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, padding=1), # ->(16,384,13,13)
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1), # ->(16,256,13,13)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2), # ->(16,256,6,6)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(), # 随机让一部分神经元失活（设置为零）
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 1000),
            nn.ReLU(),
            nn.Linear(1000, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1) # 将第一个维度到最后一个维度展平 ->(16,256*6*6)
        x = self.classifier(x)
        return x