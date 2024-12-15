from UNet import UNet
import torch.nn as nn

class smallModel(nn.Module):
    def __init__(self, in_channels=12):
        """
        Intermediate block in UNet
        Args:
          in_channels (int):  Number of input channels.
        """
        super(smallModel, self).__init__()
        self.early1 = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                              out_channels=32,
                                              kernel_size=7, padding=3),
                                    nn.ELU())
        self.conv2 = nn.Conv2d(in_channels=32,
                               out_channels=2,
                               kernel_size=3, padding=1)
        self.stage1 = UNet()

    def forward(self, x):
        early1Out = self.early1(x)
        stage1Out = self.stage1(early1Out)
        conv2Out = self.conv2(stage1Out)
        out = x[:, :2] + conv2Out
        return out
