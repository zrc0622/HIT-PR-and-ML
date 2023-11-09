import pandas as pd
from model.LSTM import LSTM2 as LSTM
import torch.nn as nn
import torch.optim as optim
import torch
import random
from datetime import datetime
import os
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

data_dir = '../data/jena_climate_2009_2016/jena_climate_2009_2016.csv'

data = pd.read_csv(data_dir)

data['Date Time'] = pd.to_datetime(data['Date Time'] ) # 转为datetime64[ns]
data.set_index('Date Time', inplace=True) # 按日期排序

train_data = data['2009':'2014']['Tdew (degC)'] # 取出训练集
test_data = data['2015':'2016']['Tdew (degC)']

input_size = 1
hidden_size = 128
output_size = 1
lr = 0.001 
windows_length = 144*180 # 一天有144条
batch_size = 1
batch_length = windows_length*batch_size

current_time = datetime.now()
log_dir = current_time.strftime("./log/forecast-%d-%H-%M-%S" + f"-{lr}") 
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
    print(f"Folder '{log_dir}' created successfully.")
else:
    print(f"Folder '{log_dir}' already exists.")
writer = SummaryWriter(log_dir=log_dir)

model = LSTM(input_size, hidden_size, output_size)
# model = torch.load('/home/lsy/projects/lstm/code/log/forecast-09-11-33-50-0.001/model.pth').to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 60
batch = 20

# 重新塑造数据以符合scaler的输入要求
train_data = train_data.values.reshape(-1, 1)
test_data = test_data.values.reshape(-1, 1)

scaler = MinMaxScaler()  # 创建一个MinMaxScaler对象
scaler.fit(train_data)  # 使用训练数据来拟合scaler

# 对训练数据和测试数据进行缩放
train_data_normalized = scaler.transform(train_data)
test_data_normalized = scaler.transform(test_data)

train_data_tensor = torch.tensor(train_data_normalized, dtype=torch.float32).to(device)
test_data_tensor = torch.tensor(test_data_normalized, dtype=torch.float32).to(device)

model.train()
for epoch in range(epochs):
    for _ in range(batch):
        i = random.randint(0, len(train_data) - batch_length - 1)

        inputs = train_data_tensor[i:i + batch_length].reshape(batch_size, windows_length, 1)
        targets = train_data_tensor[i + 1:i + batch_length + 1].reshape(batch_size, windows_length, 1)

        inputs = inputs.transpose(0, 1)
        targets = targets.transpose(0, 1)

        optimizer.zero_grad()
        
        # Initialize the hidden state
        hidden = model.init_hidden(batch_size).to(device)

        outputs, _ = model(inputs, hidden, False)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    writer.add_scalar('Loss/train', loss, epoch)

def test(start):
    test_sequence = train_data_tensor[start:start + 144*7].reshape(1, 144*7, 1)
    test_sequence = test_sequence.transpose(0, 1)
    true_values = test_sequence[-144*2:, :, :]
    model.eval()
    with torch.no_grad():
        hidden = model.init_hidden(1).to(device)
        predictions, hidden = model(test_sequence[:144*5, :, :], hidden, True) # (144*5,1,1)
        # next_prediction = []
        # next_prediction.append(predictions[-1])
        # for _ in range(144*2 - 1):
        #     predictions, hidden = model(next_prediction[-1].view(1,1,1), hidden)
        #     next_prediction.append(predictions[-1])

    # prediction_result = torch.cat(next_prediction).view(144*2,1,1)
    prediction_result = predictions[-144*2:, :, :]

    mse_loss = nn.MSELoss()
    mae_loss = nn.L1Loss()

    mse = mse_loss(torch.tensor(prediction_result), true_values).item()
    mae = mae_loss(torch.tensor(prediction_result), true_values).item()
        
    print(f"Mean Squared Error: {mse}")
    print(f"Mean Absolute Error: {mae}")
    print(start)

    result = np.array(torch.cat([test_sequence[0, :, :].view(-1).cpu(), predictions[:-1, :, :].view(-1).cpu()])) # 将前 144*5 与 prediction_result 拼接
    true = np.array(test_sequence.view(-1).cpu())
    plt.figure(figsize=(10, 6))
    plt.plot(true, label='true')
    plt.plot(result, label='prediction')
    plt.axvline(720, 0, 1, linestyle='dashed', color='red')
    plt.legend()
    plt.savefig(log_dir + f'/{start}.png')

    return prediction_result
    
for _ in range(20):
    start = random.randint(0, len(test_data) - 144*7 - 1)
    test(start)

torch.save(model, log_dir + '/model.pth')

writer.close()
