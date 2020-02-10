import torch
from torch import nn
import torchvision
from torchvision import transforms
from time import time
import d2lzh_pytorch as d2l

device = torch.device('cuda:0')

# 读取数据集
data_transform = transforms.Compose([
        transforms.Resize(size=(224,224)),
        transforms.ToTensor(),
    ])
train = torchvision.datasets.ImageFolder(root='./Datasets/PlantDisease/train', transform=data_transform)
test = torchvision.datasets.ImageFolder(root='./Datasets/PlantDisease/test', transform=data_transform)
# 读取小批量样本
batch_size = 256
train_iter = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle = True)
test_iter = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle = True)


# Inception块
class Inception(nn.Module):
    def __init__(self, in_c, c1, c2, c3, c4):
        super(Inception, self).__init__()
        # p1
        self.p1 = nn.Sequential(
            nn.Conv2d(in_c, c1, 1),
            nn.ReLU(True)
        )
        # p2
        self.p2 = nn.Sequential(
            nn.Conv2d(in_c, c2[0], 1),
            nn.Conv2d(c2[0], c2[1], 3, 1, 1),
            nn.ReLU(True)
        )
        # p3
        self.p3 = nn.Sequential(
            nn.Conv2d(in_c, c3[0], 1),
            nn.Conv2d(c3[0], c3[1], 5, 1, 2),
            nn.ReLU(True)
        )
        # p4
        self.p4 = nn.Sequential(
            nn.MaxPool2d(3, 1, 1),
            nn.Conv2d(in_c, c4, 1),
            nn.ReLU(True)
        )

    def forward(self, x):
        p1 = self.p1(x)
        p2 = self.p2(x)
        p3 = self.p3(x)
        p4 = self.p4(x)
        y = torch.cat((p1, p2, p3, p4), dim=1)
        return y


# 定义模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # input(64, 3, 224, 224)
        self.b1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),  # (64, 112, 112)
            nn.ReLU(True),
            nn.MaxPool2d(3, 2, 1),  # (64, 56, 56)
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(64, 64, 1),
            nn.Conv2d(64, 192, 3, 1, 1),  # (192, 56, 56)
            nn.MaxPool2d(3, 2, 1),  # (192, 28, 28)
        )
        self.b3 = nn.Sequential(
            Inception(192, 64, (96, 128), (16, 32), 32),  # (256, 28, 28)
            Inception(256, 128, (128, 192), (32, 96), 64),  # (480, 28, 28)
            nn.MaxPool2d(3, 2, 1)  # (480, 14, 14)
        )
        self.b4 = nn.Sequential(
            Inception(480, 192, (96, 208), (16, 48), 64),
            Inception(512, 160, (112, 224), (24, 64), 64),
            Inception(512, 128, (128, 256), (24, 64), 64),
            Inception(512, 112, (144, 288), (32, 64), 64),
            Inception(528, 256, (160, 320), (32, 128), 128),  # (832, 14, 14)
            nn.MaxPool2d(3, 2, 1)  # (832, 7, 7)
        )
        self.b5 = nn.Sequential(
            Inception(832, 256, (160, 320), (32, 128), 128),
            Inception(832, 384, (192, 384), (48, 128), 128),
            nn.AvgPool2d(7)
        )
        self.fc = nn.Linear(1024, 59)

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


net = Net()
net = net.cuda()
print(net)
# 交叉熵损失函数
loss = nn.CrossEntropyLoss()

# 优化器
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-4)

# 训练
n_epochs = 100
start = time()

net.load_state_dict(torch.load('./params/GoogleNet/1.pt'))

print('********** start training **********')
for epoch in range(n_epochs):
    test_acc = d2l.evaluate_accuracy(test_iter, net)
    # test_acc = 0
    # 保存模型参数
    PATH = './params/GoogleNet/' + str(epoch) + '.pt'
    torch.save(net.state_dict(), PATH)

    train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
    train_acc_sum_1, n_1 = 0.0, 0
    for step, (X, y) in enumerate(train_iter):
        X, y = X.to(device), y.to(device)
        y_hat = net(X)
        l = loss(y_hat, y)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        train_l_sum += float(l)
        train_acc_sum = (y_hat.argmax(dim=1) == y).sum().item()
        train_acc_sum_1 += (y_hat.argmax(dim=1) == y).sum().item()
        n = y.shape[0]
        n_1 += y.shape[0]
        print('epoch:%d | step:%d | loss:%.4f | train acc:%.3f | train acc_aver:%.3f | last test acc:%.3f | time:%.2f'
              %(epoch, step, l.item(), (train_acc_sum / n), (train_acc_sum_1 / n_1), test_acc, time()-start))

print('********** training completed **********')

