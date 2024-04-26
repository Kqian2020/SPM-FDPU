import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchsummary import summary


# 定义稀疏卷积块
class SparseConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, sparse=True):
        super(SparseConvBlock, self).__init__()
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        if sparse:
            conv = nn.utils.spectral_norm(conv)  # 使用 spectral normalization 来稀疏化卷积层
        self.conv = conv
        self.batchnorm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x)
        x = self.batchnorm(x)
        return x


# 定义稀疏 VGGNet
class SparseVGGNet(nn.Module):
    def __init__(self, num_classes=12, sparse=True):
        super(SparseVGGNet, self).__init__()
        self.sparse = sparse
        self.conv1 = SparseConvBlock(1, 3, self.sparse)
        self.conv2 = SparseConvBlock(3, 16, self.sparse)
        self.conv3 = SparseConvBlock(16, 32, self.sparse)
        self.conv4 = SparseConvBlock(32, 64, self.sparse)
        self.fc1 = nn.Linear(64*11*25, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv3(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv4(x)
        x = F.max_pool2d(x, 2, 2)
        # print("1", x.shape)
        # x = x.view(-1, 512)
        flatten_x = torch.flatten(x, 1)
        x = torch.flatten(x, 1)
        # print("2", x.shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        # print(x.shape)
        x = self.fc4(x)
        # print(x.shape)
        return x, flatten_x


# # 创建稀疏 VGGNet 实例
# sparse_vgg = SparseVGGNet(sparse=True)
#
# # 打印稀疏 VGGNet 结构
# print(sparse_vgg)

# if __name__ == '__main__':
#
#     # 创建VGGNet实例
#     # vgg = VGGNet()
#     # print(vgg)
#
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     b, c, h, w = 16, 1, 190, 400
#     inputs = torch.randn([b, c, h, w]).cuda()
#     print("inputs_shape: ", inputs.shape)
#     model = SparseVGGNet().to('cuda')
#     print(next(model.parameters()).is_cuda)
#     outputs = model(inputs)
#     # print(outputs.shape)
#     print("outputs: ", len(outputs))
#     print("outputs[0]: ", outputs[0])
#     print("outputs[1]: ", outputs[1])
#     print("outputs[0]_shape: ", outputs[0].shape)
#     print("outputs[1]_shape: ", outputs[1].shape)
#     # print(model)
#     # summary(model, input_size=(c, h, w), batch_size=b, device="cuda")