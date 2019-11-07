# -*- coding:utf-8 -*-
import pickle
import torch
import numpy as np
import torch.nn as nn
from models import VGG
import torch.optim as optim


CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# 读取cifar10的图片
def unpickle(file):
    with open(file, 'rb') as fo:
        data_dict = pickle.load(fo, encoding='bytes')
    return data_dict


if __name__ == '__main__':
    # 实例化网络模型
    model = VGG.vgg16_bn()
    print(model)
    # 交叉熵损失函数，优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    for epoch in range(1, 6):
        print("----------------train epoch:{}----------------".format(epoch))

        # 读取训练图片和标签
        file = '/Users/apple/Documents/cvData/cifar-10-batches-py/data_batch_'+epoch.__str__()
        data_dict = unpickle(file)

        for i in range(0,100):
            running_loss = 0.0
            loss = 0.0
            data = torch.from_numpy(np.reshape(data_dict[b'data'], (10000, 3, 32, 32)))[100 * i:100 * (i + 1)].float()
            labels = torch.from_numpy(np.reshape(data_dict[b'labels'], (10000, 1)))[100 * i:100 * (i + 1)].float()
            label = labels.long().squeeze()  # 压缩维度，将[10000,1]压缩为[10000]
            # 网络输出值
            output = model(data)

            # 计算损失
            loss = criterion(output, label)
            running_loss += loss.item()
            _, pred = torch.max(output, 1)  # 预测最大值所在的位置标签(_最大值，pred位置)
            num_correct = (pred == label).sum().item()  # 每次训练，预测正确的数量
            accuracy = (pred == label).float().mean().item()  # 每次训练，预测正确的比例
            # 将所有参数的梯度都置零
            optimizer.zero_grad()
            # 误差反向传播计算参数梯度
            loss.backward()
            # 通过梯度做一步参数更新
            optimizer.step()

            print("--------------------------------------------------------")
            print('{} ... {} Image, Loss: {:.6f}, Acc: {:.6f}'.format(100 * i, 100 * i + 100, running_loss, accuracy))

    print("----------------eval----------------")
    # 读取训练图片和标签
    file = '/Users/apple/Documents/cvData/cifar-10-batches-py/test_batch'
    data_dict = unpickle(file)
    data = torch.from_numpy(np.reshape(data_dict[b'data'], (10000, 3, 32, 32)))[:100].float()
    labels = torch.from_numpy(np.reshape(data_dict[b'labels'], (10000, 1)))[:100].float()
    label = labels.long().squeeze()  # 压缩维度，将[10000,1]压缩为[10000]

    eval_loss = 0
    eval_acc = 0

    # 网络输出值
    output = model(data)

    # 计算损失
    loss = criterion(output, label)
    eval_loss += loss.item()
    _, pred = torch.max(output, 1)  # 预测最大值所在的位置标签(_最大值，pred位置)
    num_correct = (pred == label).sum().item()  # 预测正确的数量
    accuracy = (pred == label).float().mean().item()  # 预测正确的比例
    print("--------------------------------------------------------")
    print('Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss, accuracy))

