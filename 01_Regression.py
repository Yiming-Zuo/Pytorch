from time import time

import numpy as np
import torch
import torch.nn as nn  # 神经网络模块
import torch.nn.functional as F  # 激励函数
import torch.optim as optim  # 优化器模块
import torch.utils.data as Data  # 读取数据模块
from torch.nn import init  # 初始化参数模块

# torch.manual_seed(1)    # 设计随机seed，用于复现

# ---------- 1.生成并读取数据集(也可以直接导入已有数据集) ----------

# 生成数据集
num_inputs = 2  # 特征数
num_examples = 1000  # 样本数
true_w = [2, -3.4]  # 正确参数
true_b = 4.2
features = torch.randn(num_examples, num_inputs,
                       dtype=torch.float32)  # 生成特征张量 shape=(1000, 2)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b  # 生成正确标签
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()),
                       dtype=torch.float32)  # 生成标签（含噪声）shape=1000

# 读取数据
batch_size = 10  # 设置batchsize
# 组合特征与标签成训练集
dataset = Data.TensorDataset(features, labels)
# 生成小批量随机样本
data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)

# ---------- 1.end -----------

# ---------- 2.定义并创建网络 ----------

# 定义网络
class Net(nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()  # 继承 __init__ 功能
        # 定义每层用什么样的形式
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # 隐藏层线性输出
        self.predict = torch.nn.Linear(n_hidden, n_output)   # 输出层线性输出

    def forward(self, x):
        # 前向传播输入值，神经网络分析出输出值
        x = F.relu(self.hidden(x))      # 激活隐藏层输出值
        x = self.predict(x)             # 输出层输出值
        return x


# 创建网络
net = Net(2, 10, 1)     # 创建网络
print(net)  # 查看网络结构

# 初始化模型参数
init.normal_(net.hidden.weight, mean=0, std=0.01)
init.constant_(net.predict.bias, val=0)
init.normal_(net.hidden.weight, mean=0, std=0.01)
init.constant_(net.predict.bias, val=0)  # 也可以直接修改bias的data: net[0].bias.data.fill_(0)

# 优化网络
# 损失函数
loss_func = nn.MSELoss()  # 均方差损失函数
# 优化器
optimizer = optim.SGD(net.parameters(), lr=0.02)  # 优化器 传入网络的所有参数和学习率
# # 为不同自网络设置不同的学习率
# optimizer = optim.SGD([
#                 # 如果对某个参数不指定学习率，就使用最外层的默认学习率
#                 {'params': net.hidden.parameters()}, # lr=0.03
#                 {'params': net.predict.parameters(), 'lr': 0.01}
#             ], lr=0.03)

# ---------- 2.end ----------

# ---------- 3.训练模型 ----------

start = time()  # 训练起始时间
epochs = 10


for epoch in range(epochs):
    train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
    for X, y in data_iter:  # 遍历样本
        output = net(X)  # 预测值
        l = loss_func(output, y.view(-1, 1)).sum()  # 损失
        optimizer.zero_grad()  # 梯度清零
        l.backward()  # 反向传播
        optimizer.step()  # 更新参数
        train_l_sum += l.item()
        train_acc_sum += (output.argmax(dim=1) == y.view(-1, 1)).sum().item()
        n += y.shape[0]
    print('epoch:%d | loss:%.4f | time:%.2f'
          % (epoch + 1, train_l_sum / n, time()-start))

# ---------- 3.end ----------
