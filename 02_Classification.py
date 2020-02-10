import torch
from torch import nn
from torch.nn import init
import torchvision
from torchvision import transforms
from time import time
import d2lzh_pytorch as d2l

# 读取数据集
mnist_train = torchvision.datasets.FashionMNIST(root='./Datasets/FashionMNIST',
                                                train=True, download=False, transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root='./Datasets/FashionMNIST',
                                               train=False, download=False, transform=transforms.ToTensor())
# 读取小批量样本
batch_size = 256
train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle = True, num_workers = 4)
test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle = True, num_workers = 4)

# 定义模型
n_features = 28 * 28
n_outputs = 10


class Net(nn.Module):
    def __init__(self, n_features, n_outputs):
        super(Net, self).__init__()
        self.output = nn.Linear(n_features, n_outputs)
    def forward(self, x):  # x-shape:(batch_size, 1, 28, 28)
        x = self.output(x.view(x.shape[0], n_features))  # x-shape:(batch_size, n_features)
        return x


net = Net(n_features, n_outputs)

# 交叉熵损失函数
loss = nn.CrossEntropyLoss()

# 优化器
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

# 训练
n_epochs = 100
start = time()

for epoch in range(n_epochs):
    train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y).sum()
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        train_l_sum += l.item()
        train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
        n += y.shape[0]
    test_acc = d2l.evaluate_accuracy(test_iter, net)
    print('epoch:%d | loss:%.4f | train acc:%.3f | test acc:%.3f | time:%.2f'
          % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc, time()-start))