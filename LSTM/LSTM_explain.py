# -*- coding: utf-8 -*-
# @Time    : 2023/03/10 10:23
# @Author  : HelloWorld！
# @FileName: seq.py
# @Software: PyCharm
# @Operating System: Windows 10
# @Python.version: 3.8

import torch
import torch.nn as nn
import argparse
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


# 数据读取与基本处理
class LoadData:
    def __init__(self,data_path ):
        self.ori_data = pd.read_csv(data_path)
    def data_observe(self):
        self.ori_data.head()
        self.draw_data(self.ori_data)
    def draw_data(self, data):
        print(data.head())
        fig_size = plt.rcParams["figure.figsize"]
        fig_size[0] = 15
        fig_size[1] = 5
        plt.rcParams["figure.figsize"] = fig_size
        plt.title('Month vs Passenger')
        plt.ylabel('Total Passengers')
        plt.xlabel('Months')
        plt.grid(True)
        plt.autoscale(axis='x', tight=True)
        plt.plot(data['passengers'])
        plt.show()
    #数据预处理，归一化
    def data_process(self):
        flight_data = self.ori_data.drop(['year'], axis=1)  # 删除不需要的列
        flight_data = flight_data.drop(['month'], axis=1)  # 删删除不需要的列

        flight_data = flight_data.dropna()  # 滤除缺失数据
        dataset = flight_data.values  # 获得csv的值
        dataset = dataset.astype('float32')
        dataset=self.data_normalization(dataset)
        return dataset

    def data_normalization(self,x):
        '''
        数据归一化（0,1）
        :param x:
        :return:
        '''
        max_value = np.max(x)
        min_value = np.min(x)
        scale = max_value - min_value
        y = (x - min_value) / scale
        return y

#构建数据集，训练集、测试集
class CreateDataSet:
    def __init__(self, dataset,look_back=2):
        dataset = np.asarray(dataset)
        data_inputs, data_target = [], []
        for i in range(len(dataset) - look_back):
            a = dataset[i:(i + look_back)]
            data_inputs.append(a)
            data_target.append(dataset[i + look_back])
        self.data_inputs = np.array(data_inputs).reshape((-1, look_back))
        self.data_target = np.array(data_target).reshape((-1, 1))

    def split_train_test_data(self, rate=0.7):
        # 划分训练集和测试集，70% 作为训练集
        train_size = math.ceil(len(self.data_inputs) * rate)  #math.ceil()向上取整
        train_inputs = self.data_inputs[:train_size]
        train_target = self.data_target[:train_size]
        test_inputs = self.data_inputs[train_size:]
        test_target = self.data_target[train_size:]
        return train_inputs, train_target, test_inputs, test_target
# 构建模型
class LSTMModel(nn.Module):
    ''' 定义LSTM模型，由于pytorch已经集成LSTM，直接用即可'''
    def __init__(self, input_size, hidden_size=4, num_layers=2, output_dim=1):
        '''

        :param input_size:  输入数据的特征维数，通常就是embedding_dim(词向量的维度)
        :param hidden_size: LSTM中隐层的维度
        :param num_layers: 循环神经网络的层数
        :param output_dim:
        '''
        super(LSTMModel,self).__init__()
        self.lstm_layer=nn.LSTM(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers)
        self.linear_layer=nn.Linear(hidden_size,output_dim)
    def forward(self,x):
        x,_=self.lstm_layer(x)
        s, b, h = x.shape
        x = x.view(s * b, h)  # 转换成线性层的输入格式
        x=self.linear_layer(x)
        x= x.view(s, b, -1)
        return x
#模型训练
class Trainer:
    def __init__(self,args):
        self.num_epoch =args.num_epoch
        self. look_back=args.look_back
        self.batch_size=args.batch_size
        self.save_modelpath=args.save_modelpath #保存模型的位置
        load_data = LoadData(args.filepath)  # 加载数据
        self.dataset = load_data.data_process()  # 数据预处理
        dataset = CreateDataSet(self.dataset , look_back=args.look_back)  # 数据集开始构建

        self.train_inputs,  self.train_target,  self.test_inputs,  self.test_target = dataset.split_train_test_data()  # 拆分数据集为训练集、测试集
        self.data_inputs = dataset.data_inputs
        #改变下输入形状
        self.train_inputs = self.train_inputs.reshape(-1, self.batch_size, self.look_back)
        self.train_target = self.train_target.reshape(-1, self.batch_size, 1)
        self.test_inputs = self.test_inputs.reshape(-1, self.batch_size, self.look_back)
        self.data_inputs = self.data_inputs.reshape(-1, self.batch_size, self.look_back)

        self.model=self.build_model()
        self.loss =nn.MSELoss()
        self.optimizer=torch.optim.Adam(self.model.parameters(), lr=1e-2)
    def build_model(self):
        model=LSTMModel(input_size=self.look_back)
        return  model

#训练过程
    def train(self):
        #把数据转成torch形式的
        inputs= torch.from_numpy(self.train_inputs)
        target=torch.from_numpy(self.train_target)
        self.model.train() #训练模式
        #开始训练
        for epoch in range(self.num_epoch):
            #前向传播
            out=self.model(inputs)
            #计算损失
            loss=self.loss(out,target)
            #反向传播
            self.optimizer.zero_grad()  #梯度清零
            loss.backward()  #反向传播
            self.optimizer.step() #更新权重参数
            if epoch % 100 == 0:  # 每 100 次输出结果
                print('Epoch: {}, Loss: {:.5f}'.format(epoch, loss.item()))
                torch.save(self.model,self.save_modelpath+'/model'+str(epoch)+'.pth')
        torch.save(self.model, self.save_modelpath + '/model_last' +  '.pth')
        self.test()
    def test(self,load_model=False):

        if not load_model:
            self.model.eval()  # 转换成测试模式
            inputs = torch.from_numpy(self.data_inputs)
            # inputs = torch.from_numpy(self.test_inputs)
            output = self.model(inputs)  # 测试集的预测结果
        else:
            model=torch.load(self.save_modelpath+ '/model_last' +  '.pth')
            inputs = torch.from_numpy(self.data_inputs)
            # inputs = torch.from_numpy(self.test_inputs)
            output =model(inputs)  # 测试集的预测结果
        # 改变输出的格式
        output = output.view(-1).data.numpy() #把tensor摊平
        # 画出实际结果和预测的结果
        plt.plot(output, 'r', label='prediction')
        plt.plot(self.dataset, 'g', label='real')
        # plt.plot(self.dataset[1:], 'b', label='real')
        plt.legend(loc='best')
        plt.show()

if __name__ == '__main__':
    filepath ='LSTM/flights.csv'
    save_modelpath='model-path'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--num_epoch',type=int, default=1000, help='训练的轮数' )
    parser.add_argument('--filepath',type=str, default=filepath, help='数据文件')
    parser.add_argument('--look_back', type=int, default=2, help='根据前几个数据预测')
    parser.add_argument('--batch_size', type=int, default=2, help='batch size')
    parser.add_argument('--save_modelpath',type=str, default=save_modelpath, help='训练中模型要保存的位置')

    args=parser.parse_args()

    train=Trainer(args)
    train.train()
    train.test(load_model=True)



