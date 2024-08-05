import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.attention(x)


# 扩张卷积模块定义
class ConvDilationBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        # 根据扩张率调整填充保持输出尺寸不变
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1,
                              padding=(kernel_size - 1) * dilation // 2, dilation=dilation)
        # 批量归一化层
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)  # 应用卷积
        x = self.bn(x)  # 应用批量归一化
        x = F.relu(x)  # 应用ReLU激活函数
        return x


class NDCNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 初始卷积层，扩展输入图像通道
        self.conv1 = nn.Conv2d(1, 3, 1)
        self.bn1 = nn.BatchNorm2d(3)

        # 第二层普通卷积
        self.conv2 = nn.Conv2d(3, 3, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(3)
        # 扩张卷积层增强感受野
        self.dilated_conv2 = ConvDilationBlock(3, 3, 3, dilation=2)

        # 更多的卷积和扩张卷积层
        self.conv3 = nn.Conv2d(6, 3, 5, padding=2)
        self.bn3 = nn.BatchNorm2d(3)
        self.dilated_conv3 = ConvDilationBlock(3, 3, 5, dilation=2)

        self.conv4 = nn.Conv2d(6, 3, 7, padding=3)
        self.bn4 = nn.BatchNorm2d(3)
        self.dilated_conv4 = ConvDilationBlock(3, 3, 7, dilation=2)

        self.conv5 = nn.Conv2d(6, 3, 9, padding=4)
        self.bn5 = nn.BatchNorm2d(3)
        self.dilated_conv5 = ConvDilationBlock(3, 3, 9, dilation=2)

        # 最后一层卷积整合所有特征，并还原成单通道输出
        self.conv6 = nn.Conv2d(15, 1, 3, padding=1)

        # 残差注意力模块
        self.res_att1 = ChannelAttention(6)
        self.res_att2 = ChannelAttention(6)
        self.res_att3 = ChannelAttention(6)
        self.res_att4 = ChannelAttention(15)

    def forward(self, x):
        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = F.relu(self.bn2(self.conv2(x1)))
        x2 = self.dilated_conv2(x2)
        concat1 = torch.cat((x1, x2), 1)
        concat1 = self.res_att1(concat1)

        x3 = F.relu(self.bn3(self.conv3(concat1)))
        x3 = self.dilated_conv3(x3)
        concat2 = torch.cat((x2, x3), 1)
        concat2 = self.res_att2(concat2)

        x4 = F.relu(self.bn4(self.conv4(concat2)))
        x4 = self.dilated_conv4(x4)
        concat3 = torch.cat((x3, x4), 1)
        concat3 = self.res_att3(concat3)

        x5 = F.relu(self.bn5(self.conv5(concat3)))
        x5 = self.dilated_conv5(x5)
        concat4 = torch.cat((x1, x2, x3, x4, x5), 1)
        concat4 = self.res_att4(concat4)

        x6 = F.relu(self.conv6(concat4))
        DC_clean_image = x6 * x - x6 + 1
        return DC_clean_image
