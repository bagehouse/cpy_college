import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.cuda

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

flight_data = pd.read_csv('LSTM/no5t.CSV')

all_data = flight_data['value'].values.astype(float)
test_data_size = 12

train_data = all_data[:-test_data_size]
test_data = all_data[-test_data_size:]

scaler = MinMaxScaler(feature_range=(-1, 1))
train_data_normalized = scaler.fit_transform(train_data .reshape(-1, 1))

#train_data_normalized = torch.FloatTensor(train_data_normalized).view(-1)
train_data_normalized = torch.FloatTensor(train_data_normalized).to(torch.float32).to(device)


train_window = 100
def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+tw:i+tw+1]
        inout_seq.append((train_seq ,train_label))
    return inout_seq

train_inout_seq = create_inout_sequences(train_data_normalized, train_window)

train_inout_seq = [(seq.to(device), labels.to(device)) for seq, labels in train_inout_seq]

class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)

    def forward(self, input_seq):
        self.hidden_cell = (self.hidden_cell[0].to(input_seq.device), self.hidden_cell[1].to(input_seq.device))
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

    
model = LSTM()
model = model.to(device)
model.add_module('linear',nn.Linear(100,1))
model = model.to(device)
model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size).to(device),
                     torch.zeros(1, 1, model.hidden_layer_size).to(device))
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)   
#print(model)

#训练
epochs = 10

# for i in range(epochs):
#     for seq, labels in train_inout_seq:
#         optimizer.zero_grad()
#         model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
#                         torch.zeros(1, 1, model.hidden_layer_size))
#         y_pred = model(seq)
#         single_loss = loss_function(y_pred, labels)
#         single_loss = single_loss.to(device)
#         single_loss.backward()
#         optimizer.step()
#     if i%25 == 1:
#         print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')
# print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')

# 梯度累积优化
grad_accumulation_steps = 10  # 设置梯度累积的步数
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for i in range(epochs):
    loss_sum = 0
    for j, (seq, labels) in enumerate(train_inout_seq):
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size).to(device),
                             torch.zeros(1, 1, model.hidden_layer_size).to(device))
        y_pred = model(seq)
        
        single_loss = loss_function(y_pred, labels)
        single_loss /= grad_accumulation_steps  # 梯度累积，除以步数
        single_loss.backward()
        
        if (j + 1) % grad_accumulation_steps == 0:
            optimizer.step()
            
        loss_sum += single_loss.item()
        
    if i % 25 == 1:
        print(f'epoch: {i:3} loss: {loss_sum / len(train_inout_seq):10.8f}')
#预测
fut_pred = 100

# #test_inputs = train_data_normalized[-train_window:].tolist().to(device)
# test_inputs = torch.FloatTensor(train_data_normalized[-train_window:]).to(device)

# #test_inputs = torch.FloatTensor(test_inputs).to(device)
# model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size).to(device),
#                      torch.zeros(1, 1, model.hidden_layer_size).to(device))
# model.eval()

# for i in range(fut_pred):
#     seq = torch.FloatTensor(test_inputs[-train_window:])
#     with torch.no_grad():
#         model.hidden_cell = (model.hidden_cell[0].to(seq.device), model.hidden_cell[1].to(seq.device))  # 添加这一行
#         #model.hidden = (torch.zeros(1, 1, model.hidden_layer_size),
#         #                torch.zeros(1, 1, model.hidden_layer_size))
#         test_inputs.append(model(seq).item())

# test_inputs[fut_pred:]
# actual_predictions = scaler.inverse_transform(np.array(test_inputs[train_window:] ).reshape(-1, 1))
# print(actual_predictions)
#test_inputs = train_data_normalized[-train_window:].clone()  # 直接使用已经创建的train_data_normalized
test_inputs = torch.FloatTensor(train_data_normalized[-train_window:]).to(torch.float32).to(device)
model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size).to(device),
                     torch.zeros(1, 1, model.hidden_layer_size).to(device))
model.eval()

for i in range(fut_pred):
    seq = test_inputs[-train_window:].clone()  # 使用clone复制张量
    with torch.no_grad():
        model.hidden_cell = (model.hidden_cell[0].to(seq.device), model.hidden_cell[1].to(seq.device))
        test_inputs = torch.cat((test_inputs, model(seq)), 0)

test_inputs = test_inputs[fut_pred:]
actual_predictions = scaler.inverse_transform(np.array(test_inputs[train_window:]).reshape(-1, 1))
print(actual_predictions)

x = np.arange(9780, 9880, 1)
print(x)


plt.title('predictions')
plt.ylabel('value')
plt.grid(True)
plt.autoscale(axis='x', tight=True)
plt.plot(flight_data['value'])
plt.plot(x,actual_predictions)
plt.show()

plt.title('predictions')
plt.ylabel('value')
plt.grid(True)
plt.autoscale(axis='x', tight=True)
plt.plot(flight_data['value'][-2000:])
plt.plot(x,actual_predictions)
plt.show()

plt.title('predictions')
plt.ylabel('value')
plt.grid(True)
plt.autoscale(axis='x', tight=True)
plt.plot(flight_data['value'][-500:])
plt.plot(x,actual_predictions)
plt.show()

plt.title('predictions')
plt.ylabel('value')
plt.grid(True)
plt.autoscale(axis='x', tight=True)
plt.plot(flight_data['value'][-200:])
plt.plot(x,actual_predictions)
plt.show()