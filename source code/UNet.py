import torch
import torch.nn as nn
from IBlock import IBlock
from decoderBlock import decoderBlock

class UNet(nn.Module):
    def __init__(self,in_channels=32,N=32):
        """
        Intermediate block in UNet
        Args:
          in_channels (int):  Number of input channels.
        """
        super(UNet, self).__init__()
        # encoder part (important- make sure input dims 2,3 are power of 2)
        self.block1 = IBlock(N,in_channels)
        self.block2 = nn.Sequential(nn.Conv2d(in_channels=N, out_channels=2*N,
                                            kernel_size=4,padding=1, stride=2),
                                    IBlock(2*N))
        self.block3 = nn.Sequential(nn.Conv2d(in_channels=2*N, out_channels=2*N,
                                            kernel_size=4,padding=1, stride=2),
                                    IBlock(2*N))
        self.block4 = nn.Sequential(nn.Conv2d(in_channels=2*N, out_channels=4*N,
                                            kernel_size=4,padding=1, stride=2),
                                    IBlock(4*N))
        self.block5 = nn.Sequential(nn.Conv2d(in_channels=4*N, out_channels=4*N,
                                            kernel_size=4,padding=1, stride=2),
                                    IBlock(4*N),
                                    nn.ConvTranspose2d(in_channels=4*N, out_channels=4*N,
                                            kernel_size=4,padding=1, stride=2))
        # decoder part
        self.block6 = decoderBlock(8*N,4*N,2*N)
        self.block7=decoderBlock(4*N,4*N,2*N)
        self.block8 = decoderBlock(4*N,2*N,N)
        # no more up-sampling
        self.block9 = decoderBlock(2*N, N)

    def forward(self, x):
        out_layer1 = self.block1(x)
        out_layer2 = self.block2(out_layer1)
        out_layer3 = self.block3(out_layer2)
        out_layer4 = self.block4(out_layer3)
        out_layer5 = self.block5(out_layer4)
        out_layer6 = self.block6(out_layer5,out_layer4)
        out_layer7 = self.block7(out_layer6,out_layer3)
        out_layer8 = self.block8(out_layer7,out_layer2)
        out = self.block9(out_layer8,out_layer1)
        return out