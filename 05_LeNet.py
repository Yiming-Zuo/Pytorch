# coding=utf-8
import torch
from torch import nn
import torchvision
from torchvision import transforms
from time import time
import d2lzh_pytorch as d2l

device = orch.device('cuda')
PATH = './LeNet.pt'

# 读取数据集
mnist_train = torchvision.datasets.FashionMNIST(root='./Datasets/FashionMNIST',
                                                train=True, download=False, transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root='./Datasets/FashionMNIST',
                                               train=False, download=False, transform=transforms.ToTensor())
# 读取小批量样本
batch_size = 200
train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle = True)
test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle = True)


# 定义模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # input(200, 1, 28, 28)
        self.conv = nn.Sequential(
            # in_channels, out_channels, kernel_size, stride, padding
            nn.Conv2d(1, 6, 5),  # (6, 24, 24)
            nn.ReLU(True),
            # kernel_size, stride
            nn.MaxPool2d(2, 2),  # (6, 12, 12)
            nn.Conv2d(6, 16, 5),  # (16, 8, 8)
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2)  # (16, 4, 4)
        )
        self.fc = nn.Sequential(
            nn.Dropout(),
            nn.Linear(16*4*4, 120),
            nn.ReLU(True),
            nn.Linear(120, 84),
            nn.ReLU(True),
            nn.Linear(84, 10),
        )

    def forward(self, x):  # x-shape:(batch_size, 1, 28, 28)
        y = self.conv(x)
        y = self.fc(y.view(y.size(0), -1))
        return y


net = Net()
net = net.cuda()
print(net)
# 交叉熵损失函数
loss = nn.CrossEntropyLoss()

# 优化器
optimizer = torch.optim.SGD(net.parameters(), lr=0.1, weight_decay=0)

# 训练
n_epochs = 20
start = time()
n_batch = len(mnist_train) / batch_size

# net.load_state_dict(torch.load(PATH))

print('********** start training **********')
for epoch in range(n_epochs):
    train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
    for X, y in train_iter:
        X, y = X.to(device), y.to(device)
        y_hat = net(X)  # 二维输入
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
print('********** training completed **********')

print('********** save parameters **********')
# 保存模型参数
torch.save(net.state_dict(), PATH)
print('********** save successful **********')