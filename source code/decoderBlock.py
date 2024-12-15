import torch
import torch.nn as nn
from IBlock import IBlock

class decoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, out_conv=None):
        """
        Intermediate block in UNet
        Args:
          in_channels (int):  Number of input channels.
        """
        super(decoderBlock, self).__init__()
        if out_conv:
            self.model = nn.Sequential(IBlock(out_channels, in_channels),
                                       nn.ConvTranspose2d(in_channels=out_channels, out_channels=out_conv,
                                                          kernel_size=4, padding=1, stride=2))
        else:
            self.model = IBlock(out_channels, in_channels)

    def forward(self, x, skip):
        inp = torch.cat((x, skip), 1)
        return self.model(inp)