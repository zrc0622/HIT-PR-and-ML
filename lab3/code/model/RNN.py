import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.tanh = nn.Tanh()
        
        # self.classfy1 = nn.Linear(64, 128)
        # self.classfy2 = nn.Linear(128, output_size)
        # self.relu = nn.ReLU()

        self.softmax = nn.Softmax(dim=1)

        # 初始化参数
        self.init_weights()

    def init_weights(self):
        # 使用Xavier初始化对模型参数进行初始化
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)

    def forward(self, x, hidden):
        seq_len, batch_size, embedding_dim = x.size() # seq_len为时间步(句子单词个数)
        outputs = []

        for i in range(seq_len):
            combined = torch.cat((x[i], hidden), dim=1) # 隐藏层和s_{t-1}输入层x_{t}二者先叠加，经过一个线性层和激活函数得到s_{t}
            hidden = self.tanh(self.i2h(combined)) 
            output = self.h2o(hidden) # s_{t}经过一个线性层得到输出
            outputs.append(output) # 分类问题只需要最后一个输出
        
        # class_output = self.classfy2(self.relu(self.classfy1(outputs[-1]))) # 取最后一个位置的输出，并连接全连接层进行分类
        
        outputs = torch.stack(outputs)
        class_output = self.softmax(outputs[-1])

        return class_output, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size) # 初始化隐藏状态
