#__author_= 'SherlockLiao'
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets

# 定义超参数
batch_size = 128        # 批的大小
learning_rate = 1e-2    # 学习率
num_epoches = 20        # 遍历训练集的次数

# 数据类型转换，转换成numpy类型
# def to_np(x):
#    return x.cpu().data.numpy()


# 下载训练集 MNIST 手写数字训练集
train_dataset = datasets.MNIST(
    root='../data', train=True, transform=transforms.ToTensor(), download=True)

test_dataset = datasets.MNIST(
    root='../data', train=False, transform=transforms.ToTensor())

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# 定义 Convolution Network 模型
class Cnn(nn.Module):
    def __init__(self, in_dim, n_class):
        super(Cnn, self).__init__()    # super用法:Cnn继承父类nn.Model的属性，并用父类的方法初始化这些属性
        self.conv = nn.Sequential(     # padding=2保证输入输出尺寸相同(参数依次是:输入深度，输出深度，ksize，步长，填充)
            nn.Conv2d(in_dim, 6, 5, stride=1, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5, stride=1, padding=0),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2))

        self.fc = nn.Sequential(
            nn.Linear(400, 120),
            nn.Linear(120, 84),
            nn.Linear(84, n_class))

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


model = Cnn(1, 10)  # 图片大小是28x28,输入深度是1，最终输出的10类
use_gpu = torch.cuda.is_available()  # 判断是否有GPU加速
if use_gpu:
    model = model.cuda()

# 定义loss和optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# logger = Logger('./logs')
# 开始训练
for epoch in range(num_epoches):
    print('epoch {}'.format(epoch + 1))      # .format为输出格式，format括号里的即为左边花括号的输出
    print('*' * 10)
    running_loss = 0.0
    running_acc = 0.0
    for i, data in enumerate(train_loader, start=1):  # enumerate枚举，遍历的起始keys=1
        img, label = data
        # cuda
        if use_gpu:
            img = img.cuda()
            label = label.cuda()
        # img = Variable(img)
        # label = Variable(label)
        # 向前传播
        out = model(img)
        loss = criterion(out, label)
        running_loss += loss.item() * label.size(0)
        _, pred = torch.max(out, 1)
        num_correct = (pred == label).sum()
        accuracy = (pred == label).float().mean()
        running_acc += num_correct.item()
        # 向后传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        """
        # ========================= Log ======================
        step = epoch * len(train_loader) + i
        # (1) Log the scalar values
        info = {'loss': loss.data[0], 'accuracy': accuracy.data[0]}

        for tag, value in info.items():
            logger.scalar_summary(tag, value, step)

        # (2) Log values and gradients of the parameters (histogram)
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            logger.histo_summary(tag, to_np(value), step)
            logger.histo_summary(tag + '/grad', to_np(value.grad), step)

        # (3) Log the images
        info = {'images': to_np(img.view(-1, 28, 28)[:10])}

        for tag, images in info.items():
            logger.image_summary(tag, images, step)
        if i % 300 == 0:
            print('[{}/{}] Loss: {:.6f}, Acc: {:.6f}'.format(
                epoch + 1, num_epoches, running_loss / (batch_size * i),
                running_acc / (batch_size * i)))
        """
    print('Finish {} epoch, Loss: {:.6f}, Acc: {:.6f}'.format(
        epoch + 1, running_loss / (len(train_dataset)), running_acc / (len(train_dataset))))
    model.eval()
    eval_loss = 0
    eval_acc = 0
    for data in test_loader:
        img, label = data
        # if use_gpu:
        #     img = Variable(img, volatile=True).cuda()
        #     label = Variable(label, volatile=True).cuda()
        # else:
        # img = Variable(img)
        # label = Variable(label)

        out = model(img)
        loss = criterion(out, label)
        eval_loss += loss.item() * label.size(0)
        _, pred = torch.max(out, 1)
        num_correct = (pred == label).sum()
        eval_acc += num_correct.item()
    print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
        test_dataset)), eval_acc / (len(test_dataset))))
    print()

# 保存模型
torch.save(model.state_dict(), './cnn.pth')
