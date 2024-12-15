import torch
import torch.nn as nn

class IBlock(nn.Module):
    def __init__(self, out_channels, in_channels=None):
        """
        Intermediate block in UNet
        Args:
          in_channels (int):  Number of input channels.
        """
        super(IBlock, self).__init__()
        # variable number of in_channels, if different than out_channels specify.
        self.in_channels = in_channels if in_channels else out_channels
        self.unit1 = nn.Sequential(nn.Conv2d(in_channels=self.in_channels,
                                             out_channels=out_channels,
                                             kernel_size=3, padding=1),
                                   nn.ELU())
        self.unit2 = nn.Sequential(nn.Conv2d(in_channels=self.in_channels + out_channels,
                                             out_channels=out_channels,
                                             kernel_size=3, padding=1),
                                   nn.ELU())
        self.unit3 = nn.Sequential(nn.Conv2d(in_channels=self.in_channels + 2 * out_channels,
                                             out_channels=out_channels,
                                             kernel_size=3, padding=1),
                                   nn.ELU())
        self.skip = nn.Conv2d(in_channels=self.in_channels, out_channels=out_channels,
                              kernel_size=3, padding=1)

    def forward(self, x):
        # YOUR CODE HERE
        out_layer1 = self.unit1(x)
        # skip layer2, concatenate along channels
        inp_layer2 = torch.cat((out_layer1, x), 1)
        out_layer2 = self.unit2(inp_layer2)
        # skip layer3
        inp_layer3 = torch.cat((out_layer2, out_layer1, x), 1)
        out_layer3 = self.unit3(inp_layer3)
        yskip = self.skip(x)
        out = yskip + out_layer3
        return out
