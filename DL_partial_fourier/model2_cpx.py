import math
import torch
import torch.nn as nn

class ComplexConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=True):
        super(ComplexConv2d, self).__init__()
        padding = kernel_size // 2
        self.conv_r = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.conv_i = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

    def forward(self, x):
        input_r, input_i = torch.split(x, x.shape[1]//2, dim=1)
        y1 = self.conv_r(input_r) - self.conv_i(input_i)
        y2 = self.conv_r(input_i) + self.conv_i(input_r)
        return torch.cat((y1, y2), dim=1)

class MeanShift(nn.Conv2d):
    def __init__(
            self, rgb_range,
            rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std

class ResBlock(nn.Module):
    def __init__(
        self, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for _ in range(3):
            m.append(ComplexConv2d(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if _ == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res

class Net_cpx(nn.Module):
    def __init__(self):
        super(Net_cpx, self).__init__()

        n_resblocks = 12
        n_feats = 24
        kernel_size = 3
        self.scale = 1
        n_colors = 1

        self.head = ComplexConv2d(n_colors, n_feats, kernel_size)
        self.body = nn.Sequential(*[
            ResBlock(n_feats, kernel_size, res_scale=1) for _ in range(n_resblocks)
        ])
        self.tail = ComplexConv2d(n_feats, n_colors, kernel_size)

        self.register_parameter("t", nn.Parameter(-2*torch.ones(1)))

    def forward(self, x):
        or_im = x
        nsize = x.size()
        mid = int(nsize[3]/2)
        # pf = math.floor(nsize[3] * 0.25)  # this is = 64
        pf = math.floor(nsize[3] * 0.45)
        pf_com = nsize[3] - pf
        
        actual = int(mid - pf)
        or_k = torch.complex(or_im[:, 0, :, :], or_im[:, 1, :, :])
        or_k = torch.fft.ifftshift(or_k, dim=(1, 2))
        or_k = torch.fft.fft2(or_k, dim=(1, 2))
        or_k = 1/math.sqrt(nsize[2] * nsize[3]) * torch.fft.fftshift(or_k, dim=(1, 2))

        for _ in range(3):
            new_k = torch.complex(x[:, 0, :, :], x[:, 1, :, :])
            new_k = torch.fft.ifftshift(new_k, dim=(1, 2))
            new_k = torch.fft.fft2(new_k, dim=(1, 2))
            new_k = 1/math.sqrt(nsize[2] * nsize[3]) * torch.fft.fftshift(new_k, dim=(1, 2))
            # new_k[:, :, pf: mid] = or_k[:, :, pf: mid]
            new_k[:, :, actual: mid] = or_k[:, :, actual: mid]
            new_k = torch.fft.ifftshift(new_k, dim=(1, 2))
            new_k = torch.fft.ifft2(new_k, dim=(1, 2))
            new_k = math.sqrt(nsize[2] * nsize[3]) * torch.fft.fftshift(new_k, dim=(1, 2))
            new_k = torch.stack((torch.real(new_k), torch.imag(new_k)), dim=1)
            x = x + self.t * (new_k - or_im)

            res = self.head(x)
            res = self.body(res)
            x = self.tail(res) + x

        new_k = torch.complex(x[:, 0, :, :], x[:, 1, :, :])
        new_k = torch.fft.ifftshift(new_k, dim=(1, 2))
        new_k = torch.fft.fft2(new_k, dim=(1, 2))
        new_k = 1/math.sqrt(nsize[2] * nsize[3]) * torch.fft.fftshift(new_k, dim=(1, 2))
        # new_k[:, :, :pf] = or_k[:, :, :pf]
        # new_k[:, :, mid:] = or_k[:, :, mid:]
        new_k[:, :, :actual] = or_k[:, :, :actual]
        new_k[:, :, mid:] = or_k[:, :, mid:]
        new_k = torch.fft.ifftshift(new_k, dim=(1, 2))
        new_k = torch.fft.ifft2(new_k, dim=(1, 2))
        new_k = math.sqrt(nsize[2] * nsize[3]) * torch.fft.fftshift(new_k, dim=(1, 2))
        y = torch.stack((torch.real(new_k), torch.imag(new_k)), dim=1)
        return y

class Net_MS_cpx(nn.Module):
    def __init__(self):
        super(Net_MS_cpx, self).__init__()

        n_resblocks = 16
        n_feats = 32
        kernel_size = 3
        self.scale = 1
        n_colors = 1

        # define head module
        self.head = ComplexConv2d(3, n_feats, kernel_size)

        # define body module
        m_body = [ResBlock(n_feats, kernel_size, res_scale=1) for _ in range(n_resblocks)]
        m_body.append(ComplexConv2d(n_feats, n_feats, kernel_size))
        self.body = nn.Sequential(*m_body)

        # define tail module
        self.tail = ComplexConv2d(n_feats, n_colors, kernel_size)

        self.register_parameter("t", nn.Parameter(-2*torch.ones(1)))

    def forward(self, x):
        nsize = x.size()
        pf = math.floor(nsize[3]*0.40)
        pf_com = nsize[3] - pf

        or_im = torch.complex(x[:, 0, :, :], x[:, 3, :, :])  # central slice
        or_k = torch.fft.ifftshift(or_im, dim=(1, 2))
        or_k = torch.fft.fft2(or_k, dim=(1, 2))
        or_k = 1/math.sqrt(nsize[2]*nsize[3])*torch.fft.fftshift(or_k, dim=(1, 2))

        x_or = x
        y = x[:, 0:2, :, :]
        y[:, 1, :, :] = x[:, 3, :, :]  # the central slice
        or_im = y

        for i in range(2):
            x = x_or
            x[:, 0, :, :].data = y[:, 0, :, :]
            x[:, 3, :, :].data = y[:, 1, :, :].data

            new_k = torch.complex(x[:, 0, :, :], x[:, 3, :, :])
            new_k = torch.fft.ifftshift(new_k, dim=(1, 2))
            new_k = torch.fft.fft2(new_k, dim=(1, 2))
            new_k = 1/math.sqrt(nsize[2]*nsize[3])*torch.fft.fftshift(new_k, dim=(1, 2))

            new_k[:, :, :pf] = or_k[:, :, :pf]  # only keep the measured data
            new_k = torch.fft.ifftshift(new_k, dim=(1, 2))
            new_k = torch.fft.ifft2(new_k, dim=(1, 2))
            new_k = math.sqrt(nsize[2]*nsize[3])*torch.fft.fftshift(new_k, dim=(1, 2))
            new_k = torch.stack((torch.real(new_k), torch.imag(new_k)), dim=1)
            y1 = y + self.t*(new_k - or_im)  # t learnable parameter
            x[:, 0, :, :].data = y1[:, 0, :, :]
            x[:, 3, :, :].data = y1[:, 1, :, :]

            res = self.head(x)
            res = self.body(res)
            y = self.tail(res) + y

        new_k = torch.complex(y[:, 0, :, :], y[:, 1, :, :])
        new_k = torch.fft.ifftshift(new_k, dim=(1, 2))
        new_k = torch.fft.fft2(new_k, dim=(1, 2))
        new_k = 1/math.sqrt(nsize[2]*nsize[3])*torch.fft.fftshift(new_k, dim=(1, 2))
        new_k[:, :, pf_com:] = or_k[:, :, pf_com:]
        new_k = torch.fft.ifftshift(new_k, dim=(1, 2))
        new_k = torch.fft.ifft2(new_k, dim=(1, 2))
        new_k = math.sqrt(nsize[2]*nsize[3])*torch.fft.fftshift(new_k, dim=(1, 2))
        y = torch.stack((torch.real(new_k), torch.imag(new_k)), dim=1)
        return y

class Net_cpx_2D(nn.Module):
    def __init__(self):
        super(Net_cpx_2D, self).__init__()

        n_resblocks = 16
        n_feats = 32
        kernel_size = 3
        n_colors = 1

        self.scale = 1

        # Define modules
        self.head = ComplexConv2d(n_colors, n_feats, kernel_size)
        self.body = nn.Sequential(*[ResBlock(n_feats, kernel_size, res_scale=1) for _ in range(n_resblocks)])
        self.tail = ComplexConv2d(n_feats, n_colors, kernel_size)

        # Learnable parameter
        self.register_parameter("t", nn.Parameter(-2 * torch.ones(1)))

    def forward(self, x):
        or_im = x
        nsize = x.size()
        pf_1 = math.floor(nsize[3] * 0.40)
        pf_com_1 = nsize[3] - pf_1
        pf_0 = math.floor(nsize[2] * 0.40)
        pf_com_0 = nsize[2] - pf_0

        or_k = torch.complex(or_im[:, 0, :, :], or_im[:, 1, :, :])
        or_k = torch.fft.ifftshift(or_k, dim=(1, 2))
        or_k = torch.fft.fft2(or_k, dim=(1, 2))
        or_k = 1 / math.sqrt(nsize[2] * nsize[3]) * torch.fft.fftshift(or_k, dim=(1, 2))

        y = x
        for i in range(3):
            x = y
            new_k = torch.complex(x[:, 0, :, :], x[:, 1, :, :])
            new_k = torch.fft.ifftshift(new_k, dim=(1, 2))
            new_k = torch.fft.fft2(new_k, dim=(1, 2))
            new_k = 1 / math.sqrt(nsize[2] * nsize[3]) * torch.fft.fftshift(new_k, dim=(1, 2))

            new_k[:, :pf_0, :] = or_k[:, :pf_0, :]
            new_k[:, :, :pf_1] = or_k[:, :, :pf_1]

            new_k = torch.fft.ifftshift(new_k, dim=(1, 2))
            new_k = torch.fft.ifft2(new_k, dim=(1, 2))
            new_k = math.sqrt(nsize[2] * nsize[3]) * torch.fft.fftshift(new_k, dim=(1, 2))
            new_k = torch.stack((torch.real(new_k), torch.imag(new_k)), dim=1)
            x = x + self.t * (new_k - or_im)

            res = self.head(x)
            res = self.body(res)
            y = self.tail(res) + x

        new_k = torch.complex(y[:, 0, :, :], y[:, 1, :, :])
        new_k = torch.fft.ifftshift(new_k, dim=(1, 2))
        new_k = torch.fft.fft2(new_k, dim=(1, 2))
        new_k = 1 / math.sqrt(nsize[2] * nsize[3]) * torch.fft.fftshift(new_k, dim=(1, 2))
        new_k_ = or_k
        new_k_[:, :pf_0, :] = new_k[:, :pf_0, :]
        new_k_[:, :, :pf_1] = new_k[:, :, :pf_1]
        new_k = new_k_
        new_k = torch.fft.ifftshift(new_k, dim=(1, 2))
        new_k = torch.fft.ifft2(new_k, dim=(1, 2))
        new_k = math.sqrt(nsize[2] * nsize[3]) * torch.fft.fftshift(new_k, dim=(1, 2))
        y = torch.stack((torch.real(new_k), torch.imag(new_k)), dim=1)
        return y