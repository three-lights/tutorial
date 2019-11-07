# -*- coding:utf-8 -*-
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pickle
import cv2
import math
import torch.optim as optim
# cifar10图片大小 3通道 32*32


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # kernel 3*3
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1)  # 64*32*32
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.maxpool1 = nn.MaxPool2d(2, stride=2)  # 64*16*16
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)  # 128*16*16
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.maxpool2 = nn.MaxPool2d(2, stride=2)  # 128*8*8
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)  # 256*8*8
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.maxpool3 = nn.MaxPool2d(2, stride=2)  # 256*4*4
        self.bn3 = nn.BatchNorm2d(256)

        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)  # 512*4*4
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.maxpool4 = nn.MaxPool2d(2, stride=2)  # 512*2*2
        self.bn4 = nn.BatchNorm2d(512)

        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)  # 512*2*2
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.maxpool5 = nn.MaxPool2d(2, stride=2)  # 512*1*1
        self.bn5 = nn.BatchNorm2d(512)

        self.drop_out = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(512, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 10)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()


    def forward(self, x):

        x = self.conv1_1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv1_2(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool1(x)

        x = self.conv2_1(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv2_2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.maxpool2(x)

        x = self.conv3_1(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.conv3_2(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.conv3_3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.maxpool3(x)

        x = self.conv4_1(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.conv4_2(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.conv4_3(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.maxpool4(x)

        x = self.conv5_1(x)
        x = self.bn5(x)
        x = F.relu(x)
        x = self.conv5_2(x)
        x = self.bn5(x)
        x = F.relu(x)
        x = self.conv5_3(x)
        x = self.bn5(x)
        x = F.relu(x)
        x = self.maxpool5(x)

        x = x.view(x.size(0), -1)  # 展平

        x = self.drop_out(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.drop_out(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)

        x = F.log_softmax(x, dim=1)

        # x = x.argmax(dim=1)  # argmax():找出最大值的索引

        return x


# 读取cifar10的图片
def unpickle(file):
    with open(file, 'rb') as fo:
        data_dict = pickle.load(fo, encoding='bytes')
    return data_dict


# 可视化图片
def img_show(file):
    data_dict = unpickle(file)
    # data_dict.keys():([b'batch_label', b'labels', b'data', b'filenames'])

    img = np.reshape(data_dict[b'data'][2], (3, 32, 32))
    img = img.transpose(2, 1, 0)
    img = img.transpose(1, 0, 2)
    # print(img.shape)
    # print(dataDict[b'labels'][2])
    # emptyImage = np.zeros(img.shape, np.uint8)

    cv2.imshow("Image", img)
    cv2.imwrite("./Image.jpg", img)  # 在当前路径下将img内容输出到Image.jpg

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # 实例化网络模型
    model = Net()
    print(model)

    # for epoch in range(1, 6):
    # print("----------------train epoch:{}----------------".format(epoch))

    # 读取训练图片和标签
    file = '/Users/apple/Documents/cvData/cifar-10-batches-py/data_batch_1'  # +epoch.__str__()
    data_dict = unpickle(file)

    # 交叉熵损失函数，优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    for i in range(0, 100):
        # running_loss = 0.0

        data = torch.from_numpy(np.reshape(data_dict[b'data'], (10000, 3, 32, 32)))[100 * i:100 * (i + 1)].float()
        labels = torch.from_numpy(np.reshape(data_dict[b'labels'], (10000, 1)))[100 * i:100 * (i + 1)].float()
        label = labels.long().squeeze()  # 压缩维度，将[10000,1]压缩为[10000]
        # 网络输出值
        output = model(data)
        print(output.size())
        # 计算损失
        loss = criterion(output, label)

        # 将所有参数的梯度都置零
        optimizer.zero_grad()
        # 误差反向传播计算参数梯度
        loss.backward()
        # 通过梯度做一步参数更新
        optimizer.step()

        # running_loss += loss.item()
        _, pred = torch.max(output, 1)  # 预测最大值所在的位置标签(_最大值，pred位置)
        # num_correct = (pred == label).sum().item()  # 每次训练，预测正确的数量
        accuracy = (pred == label).float().mean().item()  # 每次训练，预测正确的比例，=(num_correct/1000)

        print("--------------------------------------------------------")
        print('{} ... {} Image, Loss: {:.6f}, Acc: {:.6f}'.format(100 * i, 100 * i + 100, loss, accuracy))

    print("----------------eval----------------")
    # 读取训练图片和标签
    file = '/Users/apple/Documents/cvData/cifar-10-batches-py/test_batch'
    data_dict = unpickle(file)
    data = torch.from_numpy(np.reshape(data_dict[b'data'], (10000, 3, 32, 32)))[:100].float()
    labels = torch.from_numpy(np.reshape(data_dict[b'labels'], (10000, 1)))[:100].float()
    label = labels.long().squeeze()  # 压缩维度，将[10000,1]压缩为[10000]

    eval_loss = 0
    # eval_acc = 0

    # 网络输出值
    output = model(data)

    # 计算损失
    loss = criterion(output, label)
    eval_loss += loss.item()
    _, pred = torch.max(output, 1)  # 预测最大值所在的位置标签(_最大值，pred位置)
    # num_correct = (pred == label).sum().item()  # 预测正确的数量
    accuracy = (pred == label).float().mean().item()  # 预测正确的比例
    print("--------------------------------------------------------")
    print('Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss, accuracy))
