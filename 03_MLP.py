import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F  # 激励函数
import torchvision
from torchvision import transforms
from time import time
import d2lzh_pytorch as d2l
from time import time

# 读取数据集
mnist_train = torchvision.datasets.FashionMNIST(root='./Datasets/FashionMNIST',
                                                train=True, download=False, transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root='./Datasets/FashionMNIST',
                                               train=False, download=False, transform=transforms.ToTensor())
# 读取小批量样本
batch_size = 200
train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle = True, num_workers = 4)
test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle = True, num_workers = 4)

# 定义模型
n_inputs = 28 * 28
n_hiddens = 256
n_outputs = 10


class Net(nn.Module):
    def __init__(self, n_inputs, n_hiddens ,n_outputs):
        super(Net, self).__init__()
        self.hidden = nn.Linear(n_inputs, n_hiddens)
        self.output = nn.Linear(n_hiddens, n_outputs)
    def forward(self, x):  # x-shape:(batch_size, 1, 28, 28)
        y = self.hidden(x.view(x.shape[0], n_inputs))  # x-shape:(batch_size, n_features)
        y = F.relu(y)
        y = self.output(y)
        return y

net = Net(n_inputs, n_hiddens, n_outputs)
print(net)

# 交叉熵损失函数
loss = nn.CrossEntropyLoss()

# 优化器
optimizer = torch.optim.SGD(net.parameters(), lr=0.08, weight_decay=0.005)

# 训练
n_epochs = 100
start = time()
n_batch = len(mnist_train) / batch_size

for epoch in range(n_epochs):
    train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
    for X, y in train_iter:
        y_hat = net(X.view(-1,n_inputs))
        l = loss(y_hat, y)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        train_l_sum += l.item()
        train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
        n += y.shape[0]

    test_acc = d2l.evaluate_accuracy(test_iter, net)
    print('epoch:%d | loss:%.4f | train acc:%.3f | test acc:%.3f | time:%.2f'
          % (epoch + 1, train_l_sum / n_batch, train_acc_sum / n, test_acc, time()-start))
