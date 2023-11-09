import torch
import torch.nn as nn

# 用于分类
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.Wi = nn.Parameter(torch.Tensor(input_size, hidden_size * 4)) # 输入权重
        self.Wh = nn.Parameter(torch.Tensor(hidden_size, hidden_size * 4)) # 隐藏状态权重
        self.Wo = nn.Parameter(torch.Tensor(hidden_size, output_size)) # 输出层权重
        self.bi = nn.Parameter(torch.Tensor(hidden_size * 4)) # 输入偏置
        self.bh = nn.Parameter(torch.Tensor(hidden_size * 4)) # 隐藏状态偏置
        self.bo = nn.Parameter(torch.Tensor(output_size)) # 输出层偏置
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)   
        
        self.init_weights()

    def init_weights(self):
        # 初始化模型权重
        stdv = 1.0 / (self.hidden_size ** 0.5)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
    
    def forward(self, x, h_t):
        seq_len, batch_size, embedding_dim = x.size()
        c_t = h_t

        for i in range(seq_len):
            x_t = x[i, :, :]
            gates = x_t @ self.Wi + h_t @ self.Wh + self.bi # 实际上h_t是h_{t-1}
            in_gate, forget_gate, cell_gate, out_gate = torch.chunk(gates, 4, dim=1) # 按照第一维切成4个部分，分别作为输入门(f_t)、遗忘门(i_t)、细胞更新门(g_t)、输出门(o_t)
            in_gate = self.sigmoid(in_gate) # 嵌套相对应的激活函数
            forget_gate = self.sigmoid(forget_gate)
            cell_gate = self.tanh(cell_gate)
            out_gate = self.sigmoid(out_gate)
            
            c_t = c_t * forget_gate + in_gate * cell_gate # 实际上右边的c_t是c_{t-1}
            h_t = out_gate * self.tanh(c_t)
        
        output = h_t @ self.Wo + self.bo
        class_output = self.softmax(output)
        return class_output, h_t
    
    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size) # 初始化隐藏状态

# 用于预测    
class LSTM2(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM2, self).__init__()
        self.hidden_size = hidden_size
        self.Wi = nn.Parameter(torch.Tensor(input_size, hidden_size * 4)) # 输入权重
        self.Wh = nn.Parameter(torch.Tensor(hidden_size, hidden_size * 4)) # 隐藏状态权重
        self.Wo = nn.Parameter(torch.Tensor(hidden_size, output_size)) # 输出层权重
        self.bi = nn.Parameter(torch.Tensor(hidden_size * 4)) # 输入偏置
        self.bh = nn.Parameter(torch.Tensor(hidden_size * 4)) # 隐藏状态偏置
        self.bo = nn.Parameter(torch.Tensor(output_size)) # 输出层偏置
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)   
        
        self.init_weights()

    def init_weights(self):
        # 初始化模型权重
        stdv = 1.0 / (self.hidden_size ** 0.5)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
    
    def forward(self, x, h_t, test):
        if not test:
            seq_len, batch_size, embedding_dim = x.size()
            c_t = h_t
            outputs = []
            for i in range(seq_len):
                x_t = x[i, :, :]
                gates = x_t @ self.Wi + h_t @ self.Wh + self.bi # 实际上h_t是h_{t-1}
                in_gate, forget_gate, cell_gate, out_gate = torch.chunk(gates, 4, dim=1) # 按照第一维切成4个部分，分别作为输入门(f_t)、遗忘门(i_t)、细胞更新门(g_t)、输出门(o_t)
                in_gate = self.sigmoid(in_gate) # 嵌套相对应的激活函数
                forget_gate = self.sigmoid(forget_gate)
                cell_gate = self.tanh(cell_gate)
                out_gate = self.sigmoid(out_gate)
                
                c_t = c_t * forget_gate + in_gate * cell_gate # 实际上右边的c_t是c_{t-1}
                h_t = out_gate * self.tanh(c_t)
                output = h_t @ self.Wo + self.bo # (batch_size, output_size)
                outputs.append(output)
            
            outputs = torch.stack(outputs) # (seq_len, batch_size, output_size)

            return outputs, h_t
        else:
            seq_len, batch_size, embedding_dim = x.size()
            c_t = h_t
            outputs = []
            for i in range(144*7):
                if i < seq_len: 
                    x_t = x[i, :, :]
                else:
                    x_t = outputs[-1]
                gates = x_t @ self.Wi + h_t @ self.Wh + self.bi # 实际上h_t是h_{t-1}
                in_gate, forget_gate, cell_gate, out_gate = torch.chunk(gates, 4, dim=1) # 按照第一维切成4个部分，分别作为输入门(f_t)、遗忘门(i_t)、细胞更新门(g_t)、输出门(o_t)
                in_gate = self.sigmoid(in_gate) # 嵌套相对应的激活函数
                forget_gate = self.sigmoid(forget_gate)
                cell_gate = self.tanh(cell_gate)
                out_gate = self.sigmoid(out_gate)
                
                c_t = c_t * forget_gate + in_gate * cell_gate # 实际上右边的c_t是c_{t-1}
                h_t = out_gate * self.tanh(c_t)
                output = h_t @ self.Wo + self.bo # (batch_size, output_size)
                outputs.append(output)
            
            outputs = torch.stack(outputs) # (seq_len, batch_size, output_size)

            return outputs, h_t

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size) # 初始化隐藏状态
    
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.forward_lstm = LSTM(input_size, hidden_size, output_size)
        self.backward_lstm = LSTM(input_size, hidden_size, output_size)
        self.Wo = nn.Parameter(torch.Tensor(hidden_size * 2, output_size))  # 因为拼接了两个方向的隐藏状态，所以输出层权重大小为 hidden_size * 2
        self.bo = nn.Parameter(torch.Tensor(output_size))  # 输出层偏置
        self.softmax = nn.Softmax(dim=1)
        
        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / (self.hidden_size ** 0.5)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x, h_t):
        _, forward_output = self.forward_lstm(x, h_t)  # 正向LSTM
        _, backward_output= self.backward_lstm(torch.flip(x, [0]), h_t)  # 逆向LSTM，需要将输入逆序（reverse）再输入

        # 拼接两个方向的隐藏状态
        combined_output = torch.cat((forward_output, torch.flip(backward_output, [0])), dim=1)

        output = combined_output @ self.Wo + self.bo  # 输出层
        class_output = self.softmax(output)
        return class_output, combined_output

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size)