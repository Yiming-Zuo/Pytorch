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
batch_size = 64
train_iter = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle = True)
test_iter = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle = True)


# 定义模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # input(64, 3, 224, 224)
        self.conv = nn.Sequential(
            nn.Conv2d(3, 96, 11, 4, 1),  # (96, 54, 54)
            nn.ReLU(True),
            nn.Conv2d(96, 96, 1),  # (96, 54, 54)
            nn.ReLU(True),
            nn.Conv2d(96, 96, 1),  # (96, 54, 54)
            nn.ReLU(True),
            nn.MaxPool2d(3, 2),  # (96, 26, 26)

            nn.Conv2d(96, 256, 5, 1, 2),  # (256, 26, 26)
            nn.ReLU(True),
            nn.Conv2d(256, 256, 1),  # (256, 26, 26)
            nn.ReLU(True),
            nn.Conv2d(256, 256, 1),  # (256, 26, 26)
            nn.ReLU(True),
            nn.MaxPool2d(3, 2),  # (256, 12, 12)

            nn.Conv2d(256, 384, 3, 1, 1),  # (384, 12, 12)
            nn.ReLU(True),
            nn.Conv2d(384, 384, 1),  # (384, 12, 12)
            nn.ReLU(True),
            nn.Conv2d(384, 384, 1),  # (384, 12, 12)
            nn.ReLU(True),
            nn.MaxPool2d(3, 2),  # (384, 5, 5)
            nn.Dropout(p=0.5),

            nn.Conv2d(384, 1024, 3, 1, 1),  # (1024, 5, 5)
            nn.ReLU(True),
            nn.Conv2d(1024, 1024, 1),  # (1024, 5, 5)
            nn.ReLU(True),
            nn.Conv2d(1024, 59, 1),  # (1000, 5, 5)
            nn.ReLU(True),
            nn.AvgPool2d(5),  # (59, 1, 1)
        )

    def forward(self, x):
        y = self.conv(x)
        y = y.view(y.size(0), -1)
        return y


net = Net()
net = net.cuda()
print(net)
# 交叉熵损失函数
loss = nn.CrossEntropyLoss()

# 优化器
optimizer = torch.optim.Adam(net.parameters(), lr=1e-5, weight_decay=1e-4)

# 训练
n_epochs = 100
start = time()

net.load_state_dict(torch.load('./params/NiN/_2.pt'))

print('********** start training **********')
for epoch in range(n_epochs):
    test_acc = d2l.evaluate_accuracy(test_iter, net)
    # test_acc = 0
    # 保存模型参数
    PATH = './params/NiN/' + str(epoch) + '.pt'
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

