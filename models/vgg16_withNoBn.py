import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # kernel 3*3
        # 3 input image channel, 64 output channels
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1)  # 64*224*224
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.maxpool1 = nn.MaxPool2d(2, stride=2)  # 64*112*112

        # 64 input image channel, 128 output channels
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)  # 128*112*112
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.maxpool2 = nn.MaxPool2d(2, stride=2)  # 128*56*56

        # 128 input image channel, 256 output channels
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)  # 256*56*56
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, 3)
        self.maxpool3 = nn.MaxPool2d(2, stride=2)  # 256*28*28

        # 256 input image channel, 512 output channels
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)  # 512*28*28
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, 3)
        self.maxpool4 = nn.MaxPool2d(2, stride=2)  # 512*14*14

        # 512 input image channel, 512 output channels
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)  # 512*14*14
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, 3)
        self.maxpool5 = nn.MaxPool2d(2, stride=2)  # 512*7*7

        self.fc1 = nn.Linear(512*7*7, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 1000)

    def forward(self, x):

        in_size = x.size(0)  # batch_size

        out = self.conv1_1(x)
        out = F.relu(out)
        out = self.conv1_2(out)
        out = F.relu(out)
        out = self.maxpool1(out)

        out = self.conv2_1(out)
        out = F.relu(out)
        out = self.conv2_2(out)
        out = F.relu(out)
        out = self.maxpool2(out)

        out = self.conv3_1(out)
        out = F.relu(out)
        out = self.conv3_2(out)
        out = F.relu(out)
        out = self.conv3_3(out)
        out = F.relu(out)
        out = self.maxpool3(out)

        out = self.conv4_1(out)
        out = F.relu(out)
        out = self.conv4_2(out)
        out = F.relu(out)
        out = self.conv4_3(out)
        out = F.relu(out)
        out = self.maxpool4(out)

        out = self.conv5_1(out)
        out = F.relu(out)
        out = self.conv5_2(out)
        out = F.relu(out)
        out = self.conv5_3(out)
        out = F.relu(out)
        out = self.maxpool5(out)

        out = out.view(in_size, -1)  # 展平

        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)
        out = F.relu(out)

        out = F.log_softmax(out, dim=1)

        return out


if __name__ == "__main__":
    # 计算图片大小
    net = Net()
    data = torch.randn(1, 3, 224, 224)
    print(data.size())
    out = net.conv1_1(data)
    out = net.conv1_2(out)
    out = net.maxpool1(out)
    out = net.conv2_1(out)
    out = net.conv2_2(out)
    out = net.maxpool2(out)
    out = net.conv3_1(out)
    out = net.conv3_2(out)
    out = net.conv3_3(out)
    out = net.maxpool3(out)
    out = net.conv4_1(out)
    out = net.conv4_2(out)
    out = net.conv4_3(out)
    out = net.maxpool4(out)
    out = net.conv5_1(out)
    out = net.conv5_2(out)
    out = net.conv5_3(out)
    out = net.maxpool5(out)
    print(out.size())
