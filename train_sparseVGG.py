import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import torch
from SparseVGGNet import SparseVGGNet
import torch.nn as nn
import torch.optim as optim
import pandas as pd


# 读取数据
path = r"data\indicator_diagram.mat"
data = sio.loadmat(path, squeeze_me=True)
# print(data.keys())  # 'Y', 'X_csv', 'X_png'

X_png = data["X_png"]
Y = data["Y"]
# X_png = X_png[0:7000]
# # print(X_png.shape)
# Y = Y[0:7000]
print(X_png.shape, Y.shape)  # (17087, 190, 400) (17087, 12)

# 训练数据和对应的标签
train_data = X_png
train_label = Y

batch_size = 32
train_data_loader = DataLoader(dataset=train_data, shuffle=False, batch_size=batch_size)
train_label_loader = DataLoader(dataset=train_label, shuffle=False, batch_size=batch_size)

train_loader_size = len(train_data_loader)
train_dataset_label_size = len(train_label_loader)
print(len(train_data_loader))
print(len(train_label_loader))

# 训练方式
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# 网络模型
VGG_model = SparseVGGNet()
# print(VGG_model)
VGG_model = VGG_model.to(device=device)  # GPU训练

# 损失函数
criterion = nn.CrossEntropyLoss()
# 优化器
learning_rate = 1e-4
optimizer = optim.SGD(VGG_model.parameters(), lr=learning_rate)


# 训练过程
def train(model):
    epoch = 50
    total_train_step = 0
    loss_record = []
    loss_iter = []
    net = model.train()

    df = pd.DataFrame(columns=['epoch', 'train loss'])  # 列名
    df.to_csv("./SparseVGG_train_loss_all.csv", index=False)
    for i in range(epoch):
        print("------第{}轮训练开始-----".format(i + 1))

        total_train_loss = 0
        # 训练步骤开始
        for (j, img), label in zip(enumerate(train_data_loader), train_label_loader):
            # print(img.shape)  # torch.Size([8, 190, 400])
            img_data = torch.unsqueeze(img, dim=1).to(device)
            # print(img_data.shape)  # torch.Size([8, 1, 190, 400])
            img_data = img_data.type(torch.FloatTensor).to(device)
            # print("type:", type(label))
            label_data = label.to(device)
            # print(label_data.shape)  # torch.Size([8, 12])
            label_data = label_data.type(torch.FloatTensor).to(device)
            out = net(img_data)  # 训练的结果
            # print(out[0].shape)  # torch.Size([8, 12])
            loss = criterion(out[0], label_data)
            total_train_loss = total_train_loss + loss.item()

            optimizer.zero_grad()  # 梯度清零
            loss.backward()  # 反向传播
            optimizer.step()  # 更新权重

            total_train_step = total_train_step + 1

            print("|batch[{}/{}]||batch_loss {: .8f}|".format(j + 1, len(train_data_loader), loss.item()))
        train_loss_all = total_train_loss / train_loader_size
        loss_record.append(train_loss_all)
        loss_iter.append(i)
        print("整体训练集上的Loss: {}".format(total_train_loss / train_loader_size))

        list = [i, train_loss_all]
        data_loss = pd.DataFrame([list])
        data_loss.to_csv("./SparseVGG_train_loss_all.csv", mode='a', header=False, index=False)

        if i % 10 == 9:
            torch.save(net.state_dict(), "./result/SparseVGG_exp_{}.pth".format(i))
            print("模型已保存")

    plt.plot(loss_iter, loss_record)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    # plt.savefig("loss_curve.png")
    # plt.savefig("loss_curve.eps", format="eps")


if __name__ == '__main__':
    train(VGG_model)
