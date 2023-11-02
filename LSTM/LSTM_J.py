import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Load the dataset
dataframe = pd.read_csv('data/no/no6 _all.CSV', usecols=[1], engine='python', skipfooter=3)
dataset = dataframe.values.astype('float32')

# Split the dataset into training and testing sets
train_size = int(len(dataset) * 0.67)
############
# scalerdata= scaler.fit_transform(dataset)
train_data = dataset[:train_size, :]
test_data = dataset[train_size:, :]
# Normalize the dataset
scaler = MinMaxScaler(feature_range=(-1, 1))


train_data = scaler.fit_transform(train_data)
#test_data = scaler.fit_transform(test_data)

train_data = torch.FloatTensor(train_data).view(-1)
train_window = 72

def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+tw:i+tw+1]
        inout_seq.append((train_seq ,train_label))
    return inout_seq

train_inout_seq = create_inout_sequences(train_data, train_window)
# Define the LSTM model
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.lstm = nn.LSTM(input_size, hidden_layer_size)

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

input_size=1
hidden_layer_size=64
output_size=1

model = LSTM(input_size, hidden_layer_size, output_size)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
print(model)

# Train the model
epochs = 300
train_output=[]
for i in range(epochs):
    for seq, labels in train_inout_seq:
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))

        y_pred = model(seq)
        if i == (epochs-1):
            train_output.append(y_pred.item())

        single_loss = criterion(y_pred, labels)
        single_loss.backward()
        optimizer.step()

    if i%25 == 1:
        print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')

fut_pred = 675
#input_date = 
test_inputs = train_data[-train_window:].tolist()
test_seq = []
model.eval()

for i in range(fut_pred):
    seq = torch.FloatTensor(test_inputs[-train_window:])
    with torch.no_grad():
        model.hidden = (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))
        test_output = model(seq)
        test_inputs.append(test_output.item())
       # test_seq.append(test_output.item())

print(test_inputs)
        
##############
actual_t_predictions = scaler.inverse_transform(np.array(train_output).reshape(-1, 1))
# actual_t_predictions=train_output
actual_predictions = scaler.inverse_transform(np.array(test_inputs).reshape(-1, 1))
# actual_predictions = test_seq
# actual_predictions=test_inputs[train_window:]
#actual_predictions = np.array(test_inputs[train_window:] ).reshape(-1, 1)



# Plot the results
x_train = np.arange(0, len(actual_t_predictions), 1)
x_test = np.arange(len(actual_t_predictions)+1, len(dataset), 1)

plt.plot(dataset, label='Original Data')
plt.grid(True)
plt.autoscale(axis='x', tight=True)
plt.plot(x_train,actual_t_predictions)
plt.plot(x_test,actual_predictions)
plt.show()
plt.savefig('data/no/no6_all_lstm.png')


