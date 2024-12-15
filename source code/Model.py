import torch
import torch.nn as nn
from UNet import UNet
from util import NUM_CHANNELS

class Model(nn.Module):
    def __init__(self, in_channels=NUM_CHANNELS):
        """
        Intermediate block in UNet
        Args:
          in_channels (int):  Number of input channels.
        """
        super(Model, self).__init__()
        self.early1 = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                              out_channels=32,
                                              kernel_size=7, padding=3),
                                    nn.ELU())
        self.early2 = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                              out_channels=32,
                                              kernel_size=7, padding=3),
                                    nn.ELU())
        self.conv1 = nn.Conv2d(in_channels=32,
                               out_channels=32,
                               kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=32,
                               out_channels=2,
                               kernel_size=3, padding=1)
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=2,
                                             out_channels=32,
                                             kernel_size=1), nn.Sigmoid())
        self.convLast = nn.Conv2d(in_channels=32,
                                  out_channels=2,
                                  kernel_size=3, padding=1)

        # two objects for separate training?
        self.stage1 = UNet()
        self.stage2 = UNet(in_channels=64)

    def forward(self, x):
        early1Out = self.early1(x)
        early2Out = self.early2(x)
        stage1Out = self.stage1(early1Out)
        skip1 = self.conv1(stage1Out)
        conv2Out = self.conv2(stage1Out)
        out1 = x[:, :2] + conv2Out
        conv3Out = self.conv3(out1)
        multiplied = conv3Out * skip1
        added = multiplied + stage1Out
        concated = torch.cat((added, early2Out), dim=1)
        stage2Out = self.stage2(concated)
        out2 = self.convLast(stage2Out)
        return out1, out2
