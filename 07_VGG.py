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
        # input(256, 3, 224, 224)
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),  # (64, 224, 224)
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1),  # (64, 224, 224)
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # (64, 112, 112)

            nn.Conv2d(64, 128, 3, 1, 1),  # (128, 112, 112)
            nn.ReLU(True),
            nn.Conv2d(128, 128, 3, 1, 1),  # (128, 112, 112)
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # (128, 56, 56)

            nn.Conv2d(128, 256, 3, 1, 1),  # (256, 56, 56)
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1),  # (256, 56, 56)
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1),  # (256, 56, 56)
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # (256, 28, 28)

            nn.Conv2d(256, 512, 3, 1, 1),  # (256, 28, 28)
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1),  # (512, 28, 28)
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1),  # (512, 28, 28)
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # (512, 14, 14)

            nn.Conv2d(512, 512, 3, 1, 1),  # (512, 14, 14)
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1),  # (512, 14, 14)
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1),  # (512, 14, 14)
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # (512, 7, 7)
        )
        self.fc = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.Dropout(),
            nn.ReLU(True),
            nn.Linear(4096, 59),
        )

    def forward(self, x):
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

net.load_state_dict(torch.load('./params/VGG/1_.pt'))

print('********** start training **********')
for epoch in range(n_epochs):
    test_acc = d2l.evaluate_accuracy(test_iter, net)
    # test_acc = 0
    # 保存模型参数
    PATH = './params/VGG/' + str(epoch) + '.pt'
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

