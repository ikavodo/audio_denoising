import torch
import torch.nn as nn


class IBlock(nn.Module):
    def __init__(self, out_channels, in_channels=None):
        """
        Intermediate block in decoderBlock
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


class UNet(nn.Module):
    def __init__(self,in_channels=32,N=32):
        """
        Main block in network
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