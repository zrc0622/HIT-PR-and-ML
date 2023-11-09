import torch
import torch.nn as nn

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.gru_update = nn.Linear(input_size + hidden_size, hidden_size) # 更新门
        self.gru_reset = nn.Linear(input_size + hidden_size, hidden_size) # 重置门
        self.gru_candidate = nn.Linear(input_size + hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        
    def initialize_weights(model):
        for name, param in model.named_parameters():
            if 'weight' in name:
                if 'gru' in name:
                    # 初始化GRU层权重
                    if 'update' in name or 'reset' in name or 'candidate' in name:
                        nn.init.xavier_uniform_(param.data)
                elif 'fc' in name:
                    # 初始化全连接层权重
                    nn.init.xavier_normal_(param.data)
            elif 'bias' in name:
                # 初始化偏置项
                nn.init.constant_(param.data, 0.0)
            
    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size) # 初始化隐藏状态

    def forward(self, x, hidden):
        seq_len, batch_size, embedding_dim = x.size()
        
        for i in range(seq_len):
            x_t = x[i, :, :]
            combined = torch.cat((x_t, hidden), dim=1)
            update_gate = self.sigmoid(self.gru_update(combined)) # z_t
            reset_gate = self.sigmoid(self.gru_reset(combined)) # r_t
            candidate = self.tanh(self.gru_candidate(torch.cat((x_t, reset_gate * hidden), dim=1)))
            hidden = update_gate * hidden + (1 - update_gate) * candidate
        
        output = self.fc(hidden)
        class_output = self.softmax(output)
        return class_output, hidden