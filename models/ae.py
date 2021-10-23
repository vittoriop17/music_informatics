import torch
from torch import nn
from torch.nn import Module

# see https://arxiv.org/pdf/2107.03312.pdf for implementation


def check_args(args):
    pass
    # raise NotImplementedError()


class Encoder(Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        # C = args.C if hasattr(args, "C") else Exception("Fix a value for C")
        # K = args.K if hasattr(args, "K") else Exception("Fix a value for K")
        C = 2
        K = 16*C
        self.conv_1 = nn.Conv1d(in_channels=2, out_channels=C, kernel_size=7, device=args.device)
        self.enc_block_1 = EncoderBlock(2*C, 2, args.device)
        self.enc_block_2 = EncoderBlock(4*C, 4, args.device)
        self.enc_block_3 = EncoderBlock(8*C, 5, args.device)
        self.enc_block_4 = EncoderBlock(16*C, 8, args.device)
        self.conv_2 = nn.Conv1d(in_channels=16*C, out_channels=K, kernel_size=3, device=args.device)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.enc_block_1(x)
        x = self.enc_block_2(x)
        x = self.enc_block_3(x)
        x = self.enc_block_4(x)
        x = self.conv_2(x)
        return x


class Decoder(Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        # C = args.C if hasattr(args, "C") else Exception("Fix a value for C")
        # K = args.K if hasattr(args, "K") else Exception("Fix a value for K")
        C = 2
        K = 16*C
        self.conv_1 = nn.Conv1d(in_channels=K, out_channels=8*C, kernel_size=7, device=args.device)
        self.dec_block_1 = DecoderBlock(8*C, 8, args.device)
        self.dec_block_2 = DecoderBlock(4*C, 5, args.device)
        self.dec_block_3 = DecoderBlock(2*C, 4, args.device)
        self.dec_block_4 = DecoderBlock(C, 2, args.device)
        self.conv_2 = nn.Conv1d(in_channels=C//2, out_channels=1, kernel_size=3, padding=1326, device=args.device)
        self.net = nn.Sequential(self.conv_1,
                                 self.dec_block_1,
                                 self.dec_block_2,
                                 self.dec_block_3,
                                 self.dec_block_4,
                                 self.conv_2)

    def forward(self, x):
        x = self.net(x)
        return x


class ResidualUnit(Module):
    def __init__(self, in_c, out_c, dilation, device):
        super(ResidualUnit, self).__init__()
        self.conv1d_1 = nn.Conv1d(in_channels=in_c, out_channels=out_c, kernel_size=7, dilation=dilation, device=device)
        self.conv1d_2 = nn.Conv1d(in_channels=out_c, out_channels=out_c, kernel_size=1, device=device)

    def forward(self, x: torch.Tensor):
        x_1 = self.conv1d_1(x)
        x_2 = self.conv1d_2(x_1)
        shapes = x.shape[0], x.shape[1], x.shape[2] - x_2.shape[2]
        return torch.cat((torch.zeros(shapes), x_2), dim=2) + x


class EncoderBlock(Module):
    def __init__(self, n, s, device):
        super(EncoderBlock, self).__init__()
        self.residual_1 = ResidualUnit(n//2, n//2, 1, device)
        self.residual_2 = ResidualUnit(n//2, n//2, 3, device)
        self.residual_3 = ResidualUnit(n//2, n//2, 9, device)
        self.conv1d = nn.Conv1d(in_channels=n//2, out_channels=n, kernel_size=2*s, stride=s, device=device)

    def forward(self, x):
        x = self.residual_1(x)
        x = self.residual_2(x)
        x = self.residual_3(x)
        x = self.conv1d(x)
        return x


class DecoderBlock(Module):
    def __init__(self, n, s, device):
        super(DecoderBlock, self).__init__()
        self.conv1d = nn.ConvTranspose1d(in_channels=n, out_channels=n//2, kernel_size=2*s, stride=s, device=device)
        self.residual_1 = ResidualUnit(n//2, n//2, 1, device)
        self.residual_2 = ResidualUnit(n//2, n//2, 3, device)
        self.residual_3 = ResidualUnit(n//2, n//2, 9, device)

    def forward(self, x):
        x = self.conv1d(x)
        x = self.residual_1(x)
        x = self.residual_2(x)
        x = self.residual_3(x)
        return x


class Music_AE(Module):
    def __init__(self, args):
        super(Music_AE, self).__init__()
        check_args(args)
        self.encoder = Encoder(args)
        self.decoder = Decoder(args)

    def forward(self, x):
        x_enc = self.encoder(x)
        x_dec = self.decoder(x_enc)
        return x_dec
