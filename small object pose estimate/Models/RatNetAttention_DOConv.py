import torch
import torch.nn as nn
from torchvision import models
from Models.do_conv_pytorch import DOConv2d


# 中间层特征提取
class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers

    def forward(self, x):
        outputs = []
        for name, module in self.submodule._modules.items():
            if name is "fc": x = x.view(x.size(0), -1)
            x = module(x)
            if name in self.extracted_layers:
                outputs.append(x)
        return outputs


class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            DOConv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            DOConv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = DOConv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out


class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            DOConv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class Net_ResnetAttention_DOConv(nn.Module):
    def __init__(self, model_path, extract_list, device, train, n_channels, nof_joints):
        super(Net_ResnetAttention_DOConv, self).__init__()
        self.n_classes = nof_joints
        self.n_channels = n_channels
        self.model_path = model_path
        self.extract_list = extract_list
        self.device = device

        self.cbam1 = CBAM(channel=256)
        self.cbam2 = CBAM(channel=256)
        self.cbam3 = CBAM(channel=256)
        self.Up1 = up_conv(ch_in=2048, ch_out=256)
        self.Up2 = up_conv(ch_in=256, ch_out=256)
        self.Up3 = up_conv(ch_in=256, ch_out=256)


        self.outConv = nn.Conv2d(256, self.n_classes, kernel_size=(1, 1), stride=(1, 1))
        if train:
            nn.init.normal_(self.outConv.weight, std=0.001)
            nn.init.constant_(self.outConv.bias, 0)
        self.resnet = models.resnet50(pretrained=False)
        if train:
            self.resnet.load_state_dict(torch.load(self.model_path, map_location=self.device), strict=False)
        self.SubResnet = FeatureExtractor(self.resnet, self.extract_list)  # 提取resnet层

    def forward(self, img):
        f = self.SubResnet(img)
        f1 = self.Up1(f[0])
        f1 = self.cbam1(f1) + f1
        f2 = self.Up2(f1)
        f2 = self.cbam2(f2) + f2
        f3 = self.Up3(f2)
        f3 = self.cbam3(f3) + f3
        out = self.outConv(f3)
        return out
