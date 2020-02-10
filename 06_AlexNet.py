import torch
from torch import nn
import torchvision
from torchvision import transforms
from time import time
import d2lzh_pytorch as d2l

device = torch.device('cuda')

# 读取数据集
data_transform = transforms.Compose([
        transforms.Resize(size=(227,227)),
        transforms.ToTensor(),
    ])
train = torchvision.datasets.ImageFolder(root='./Datasets/PlantDisease/train', transform=data_transform)
test = torchvision.datasets.ImageFolder(root='./Datasets/PlantDisease/test', transform=data_transform)
# 读取小批量样本
batch_size = 512
train_iter = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle = True)
test_iter = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle = True)


# 定义模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # input(300, 3, 227, 227)
        self.conv = nn.Sequential(
            # in_channels, out_channels, kernel_size, stride, padding
            nn.Conv2d(3, 96, 11, 4),  # (96, 55, 55)
            nn.ReLU(True),
            # kernel_size, stride
            nn.MaxPool2d(3, 2),  # (96, 27, 27)
            nn.Conv2d(96, 256, 5, 1, 2),  # (256, 27, 27)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),  # (256, 13, 13)
            nn.Conv2d(256, 384, 3, 1, 1),  # (384, 13, 13)
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, 3, 1, 1),  # (384, 13, 13)
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, 3, 1, 1),  # (256, 13, 13)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),  # (256, 6, 6)
        )
        self.fc = nn.Sequential(
            nn.Linear(256*6*6, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.Dropout(),
            nn.ReLU(True),
            nn.Linear(4096, 59),
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
optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=0)

# 训练
n_epochs = 100
start = time()

# net.load_state_dict(torch.load(PATH))

print('********** start training **********')
for epoch in range(n_epochs):
    # test_acc = d2l.evaluate_accuracy(test_iter, net)
    # print('test_acc:%.3f' %test_acc)
    print('********** save parameters **********')
    # 保存模型参数
    PATH = './params/' + str(epoch) + '.pt'
    torch.save(net.state_dict(), PATH)
    print('********** save successful **********')
    train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
    train_acc_sum_1, n_1 = 0.0, 0
    for step, (X, y) in enumerate(train_iter):
        X, y = X.to(device), y.to(device)
        y_hat = net(X)  # 二维输入
        l = loss(y_hat, y)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        train_l_sum += l.item()
        train_acc_sum = (y_hat.argmax(dim=1) == y).sum().item()
        train_acc_sum_1 += (y_hat.argmax(dim=1) == y).sum().item()
        n = y.shape[0]
        n_1 += y.shape[0]
        print('epoch:%d | step:%d | loss:%.4f | train acc:%.3f | train acc_sum:%.3f | time:%.2f'
              %(epoch, step, l.item(), (train_acc_sum / n), (train_acc_sum_1 / n_1), time()-start))
    test_acc = d2l.evaluate_accuracy(test_iter, net)
    print('test_acc:%.3f' %test_acc)
print('********** training completed **********')

