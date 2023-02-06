import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import math
import torch.nn.functional as F
class Normalization(nn.Module):
    def __init__(self):
        super(Normalization, self).__init__()
        self.alpha = Parameter(torch.ones(1))
        self.beta  = Parameter(torch.zeros(1))

    def forward(self, x):
        x = torch.nn.functional.normalize(x, dim=1)
        return x * self.alpha + self.beta


class HighPass(nn.Module):
    def __init__(self, w_hpf, device):
        super(HighPass, self).__init__()
        self.register_buffer('filter',
                             torch.tensor([[-1, -1, -1],
                                           [-1, 8., -1],
                                           [-1, -1, -1]]) / w_hpf)

    def forward(self, x):
        filter = self.filter.unsqueeze(0).unsqueeze(1).repeat(x.size(1), 1, 1, 1)
        return F.conv2d(x, filter, padding=1, groups=x.size(1))





class AdaIN(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features*2)

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta

class ConvBlock(torch.nn.Module):
    """docstring for ConvBlock"""
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=1, padding=0):
        super(ConvBlock, self).__init__()

        self.main_module = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2))
        
    def forward(self, x):
        return self.main_module(x)

class TransConvBlock(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=4, stride=1, padding=0):
        super(TransConvBlock, self).__init__()

        self.main_module = nn.Sequential(

            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),

            nn.LeakyReLU(0.2))
        
    def forward(self, x):
        return self.main_module(x)


class ConnectionBlk(nn.Module):
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2),
                 normalize=False, downsample=True):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = downsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = nn.Conv2d(dim_in, dim_in, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        if self.normalize:
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)  # unit variance


class ConnectionUpsampleBlk(nn.Module):
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2),
                 normalize=False, upsample=True):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.upsample = upsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = nn.Conv2d(dim_in, dim_in, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        if self.normalize:
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)  # unit variance



class ResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2),
                 normalize=False, downsample=True):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = downsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = nn.Conv2d(dim_in, dim_in, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        if self.normalize:
            self.norm1 = nn.BatchNorm2d(dim_in, affine=True)
            self.norm2 = nn.BatchNorm2d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)  # unit variance



class AdainResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, style_dim=100, w_hpf=0,
                 actv=nn.LeakyReLU(0.2), upsample=True):
        super().__init__()
        self.w_hpf = w_hpf
        self.actv = actv
        self.upsample = upsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out, style_dim)

    def _build_weights(self, dim_in, dim_out, style_dim=64):
        self.conv1 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_out, dim_out, 3, 1, 1)
        self.norm1 = AdaIN(style_dim, dim_in)
        self.norm2 = AdaIN(style_dim, dim_out)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s):
        x = self.norm1(x, s)
        x = self.actv(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv1(x)
        x = self.norm2(x, s)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x, s):
        out = self._residual(x, s)
        if self.w_hpf == 0:
            out = (out + self._shortcut(x)) / math.sqrt(2)
        return out

class Encoder(torch.nn.Module):
    def __init__(self, channels, latent_size = 64, style_dim = 256, token = False):
        super().__init__()
        # Filters [256, 512, 1024]
        # Input_dim = channels (Cx64x64)
        # Output_dim = 1
        # C x 256 x 256
        n = 4
        # self.conv = nn.Sequential(
        #     nn.Conv2d(channels, 64, kernel_size=3, padding=2,stride=1, bias=False),
        #     nn.InstanceNorm2d(64, affine=True),
        #     nn.LeakyReLU(0.2))

        #128
        self.conv_1 = ResBlk(dim_in = channels, dim_out = 32)

        # # 128 x 64 x 32

        self.conv_2 = ResBlk(dim_in = 32, dim_out = 64)

        # # 256 x 32 x 16
        self.conv_3 = ResBlk(dim_in = 64, dim_out = 128)

        # # 1024 x 16 x 8
        self.conv_4 = ResBlk(dim_in = 128, dim_out = 256)

        # 4 x 8
        self.conv_5 = ResBlk(dim_in = 256, dim_out = 512)
        # 4
     #   self.conv_6 = ResBlk(dim_in = 512, dim_out = 512)
        # 
        #self.conv_7 = ResBlk(dim_in = 512, dim_out = 512)
        #self.conv_6 = ConvBlock(in_channels=1024, out_channels=latent_size, kernel_size=3, stride=2, padding=1)
        # # 1024 x 16 x 16
        # self.output = nn.Sequential(
        #     nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1),
        #     )
        self.avg_pool = nn.AdaptiveAvgPool2d((2, 2))
        self.mean = nn.Linear(512*2*2, latent_size)
        self.style = nn.Linear(512*2*2, style_dim)
        self.token = token
        # self.z = nn.Conv2d(in_channels=512, out_channels=latent_size, kernel_size=1, bias = False)
        # self.style = nn.Conv2d(in_channels=512, out_channels=style_dim, kernel_size=1, bias = False)

        if token:
            self.input_token = nn.Parameter(torch.randn(2, 128,128))

        # self.attribute = nn.Sequential(
        #     nn.Conv2d(in_channels=512, out_channels=3, kernel_size=4, stride=2, padding=1),
        #     )
        # 100 x 8 x 8
#        self.last_layer = nn.Linear(1024, latent_size)

    def forward(self, x):
        bs = x.shape[0]
        en_ft = []
      #  if self.token:
        # input_token = self.input_token[y.long()].unsqueeze(1)
        # x = torch.cat((x, input_token), dim = 1)

        x = self.conv_1(x)
        en_ft.append(x)

        x = self.conv_2(x)
        en_ft.append(x)


        x = self.conv_3(x)
        en_ft.append(x)

        x = self.conv_4(x)
        en_ft.append(x)

        x = self.conv_5(x)
        en_ft.append(x)


        # x = self.conv_6(x)
        # en_ft.append(x)

        # x = self.output(x)
       # x = self.conv_7(x)
        x = self.avg_pool(x)
        x = x.view(bs, -1)
        mu = self.mean(x)
        s = self.style(x)
        # z = self.z(x).view(bs, -1)
        # s = self.style(x).view(bs, -1)
        return mu, s, en_ft

class Generator(torch.nn.Module):
    def __init__(self, channels, n_attr = 2, latent_size = 64,  style_dim = 256):
        super().__init__()
        # Filters [1024, 512, 256]
        # Input_dim = 100
        # Output_dim = C (number of channels)
        # 3,256,256
        # 102 8x8
        #1x1

        #self.linear = nn.Linear(latent_size+n_attr, 512*4*4)
        # 4x4   
        #self.conv = AdainResBlk(dim_in=latent_size+n_attr, dim_out=512, style_dim = style_dim)
        self.conv =  TransConvBlock(in_channels=latent_size+n_attr, out_channels=512, kernel_size=4, stride=2, padding=1)

        self.conv0 = AdainResBlk(dim_in=1024, dim_out=512, style_dim = style_dim)
        self.conv1 = AdainResBlk(dim_in=1024, dim_out=256, style_dim = style_dim)




        # 8x8
        self.conv2 = AdainResBlk(dim_in=512, dim_out=128, style_dim = style_dim)
        # 256, 16 x 16   
        self.conv3  = AdainResBlk(dim_in=256, dim_out=64, style_dim = style_dim)
        # 32 x 32
        self.conv4  = AdainResBlk(dim_in=128, dim_out=32, style_dim = style_dim)

        # 64 x 64

        self.conv5  = AdainResBlk(dim_in=64, dim_out=16, style_dim = style_dim)

        self.last_conv = nn.Conv2d(in_channels=16, out_channels=channels, kernel_size=1, bias = False)
        # 128, 128 x 128

        self.output = nn.Tanh()
        self.token = nn.Parameter(torch.randn(2, n_attr))

    def forward(self, x, y, s, en_ft):
        bs = x.shape[0]
        latent_token = self.token[y.long()]

        x = torch.cat((x, latent_token), dim = -1).unsqueeze(-1).unsqueeze(-1)

        # x = self.linear(x)  ##1024, 16 x 16
        # x = x.view(bs, 512,4,4)
        x = self.conv(x)               # 2x2

        x = torch.cat((x, en_ft[-1]), dim = 1)


        x = self.conv0(x, s)   # 4x4
        x = torch.cat((x, en_ft[-2]), dim = 1)


        # x = self.conv0(x, s)   # 8x8
        # x = torch.cat((x, en_ft[-2]), dim = 1)

        x = self.conv1(x, s)

        x = torch.cat((x, en_ft[-3]), dim = 1)
        x = self.conv2(x, s)
        
        x = torch.cat((x, en_ft[-4]), dim = 1)

        x = self.conv3(x, s)
        
        x = torch.cat((x, en_ft[-5]), dim = 1)
        x = self.conv4(x, s)
        
        x = torch.cat((x, en_ft[-6]), dim = 1)

        x = self.conv5(x, s)
        

        x = self.last_conv(x)

        return self.output(x)

'''
class Discriminator(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Filters [256, 512, 1024]
        # Input_dim = channels (Cx64x64)
        # Output_dim = 1
   
        self.main_module = nn.Sequential(
            # Image (Cx32x32)
            nn.Conv2d(in_channels=channels, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # State (256x16x16)
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # State (512x8x8)
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True))
            # outptut of main module --> State (1024x4x4)
        self.avg_pool = nn.AdaptiveAvgPool2d((2, 2))
        self.output = nn.Sequential(
            nn.Linear(1024*2*2, 1),
            # Output 1
            nn.Sigmoid())

    def forward(self, x, return_feature = False):
        x = self.main_module(x)
        # x = x.mean(-1).mean(-1)
        x = self.avg_pool(x)
        x = x.view(-1, 1024*2*2)
        if return_feature:
            return x
        else:
            return self.output(x)

    def feature_extraction(self, x):
        # Use discriminator for feature extraction then flatten to vector of 16384 features
        x = self.main_module(x)
        
        return x.view(-1, 1024*4*4)
'''

class Discriminator(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Filters [256, 512, 1024]
        # Input_dim = channels (Cx64x64)
        # Output_dim = 1
   
        self.main_module = nn.Sequential(
            # Image (Cx32x32)
            ResBlk(dim_in = channels, dim_out = 64),

            ResBlk(dim_in = 64, dim_out = 128),

            ResBlk(dim_in = 128, dim_out = 256),

            # State (256x16x16)
            ResBlk(dim_in = 256, dim_out = 512),

            # State (512x8x8)
            ResBlk(dim_in = 512, dim_out = 1024))
            # outptut of main module --> State (1024x4x4)
        self.avg_pool = nn.AdaptiveAvgPool2d((2, 2))
        self.output = nn.Sequential(
            nn.Linear(1024*2*2, 1),
            # Output 1
            nn.Sigmoid())

    def forward(self, x, return_feature = False):
        x = self.main_module(x)
        # x = x.mean(-1).mean(-1)
        x = self.avg_pool(x)
        x = x.view(-1, 1024*2*2)
        if return_feature:
            return x
        else:
            return self.output(x)



class Encoder_VAE(torch.nn.Module):
    def __init__(self, channels, latent_size = 64 , token = True):
        super().__init__()
        # Filters [256, 512, 1024]
        # Input_dim = channels (Cx64x64)
        # Output_dim = 1
        # C x 256 x 256
        n = 4
        # self.conv = nn.Sequential(
        #     nn.Conv2d(channels, 64, kernel_size=3, padding=2,stride=1, bias=False),
        #     nn.InstanceNorm2d(64, affine=True),
        #     nn.LeakyReLU(0.2))


        self.conv_1 = ResBlk(dim_in = channels, dim_out = 64)

        # # 128 x 32 x 32

        self.conv_2 = ResBlk(dim_in = 64, dim_out = 128)

        # # 256 x 16 x 16
        self.conv_3 = ResBlk(dim_in = 128, dim_out = 256)

        # # 1024 x 8 x 8
        self.conv_4 = ResBlk(dim_in = 256, dim_out = 512)

        # 4 x 4
        self.conv_5 = ResBlk(dim_in = 512, dim_out = 1024)
        #self.conv_6 = ConvBlock(in_channels=1024, out_channels=latent_size, kernel_size=3, stride=2, padding=1)
        # # 1024 x 16 x 16
        # self.output = nn.Sequential(
        #     nn.Conv2d(in_channels=1024, out_channels=latent_size, kernel_size=3, stride=2, padding=1),
        #     )
        self.avg_pool = nn.AdaptiveAvgPool2d((2, 2))
        self.mean = nn.Linear(1024*2*2, latent_size)
        self.std = nn.Linear(1024*2*2, latent_size)


        if token:
            self.input_token = nn.Parameter(torch.randn(2, 128,128))
        # self.attribute = nn.Sequential(
        #     nn.Conv2d(in_channels=512, out_channels=3, kernel_size=4, stride=2, padding=1),
        #     )
        # 100 x 8 x 8
#        self.last_layer = nn.Linear(1024, latent_size)

    def forward(self, x, y):
        bs = x.shape[0]

        
        input_token = self.input_token[y.long()].unsqueeze(1)
        x = torch.cat((x, input_token), dim = 1)

        x = self.conv_1(x)
        #print(x.shape)
        x = self.conv_2(x)
        #print(x.shape)

        x = self.conv_3(x)
       # print(x.shape)

        x = self.conv_4(x)
        #print(x.shape)

        x = self.conv_5(x)
        
        x = self.avg_pool(x)
        x = x.view(bs, -1)
        mu = self.mean(x)
        log_var = self.std(x)
       # style = self.style(x)
       # attr = self.attribute(x)
        return mu, log_var, mu




class Decoder_VAE(torch.nn.Module):
    def __init__(self, channels, n_attr = 10, latent_size = 64, style_dim = 256):
        super().__init__()
        # Filters [1024, 512, 256]
        # Input_dim = 100
        # Output_dim = C (number of channels)
        # 3,256,256
        # 102 8x8
        #1x1
        # self.conv1 = nn.Sequential(
        #                          nn.Conv2d(in_channels=latent_size+n_attr, out_channels=1024, kernel_size=3, stride=2, padding=1))

        self.linear = nn.Linear(latent_size+n_attr, 1024*4*4)
        # 4x4   
        self.conv1 = AdainResBlk(dim_in=1024, dim_out=512, style_dim = style_dim)

        # 8x8
        self.conv2 = AdainResBlk(dim_in=512, dim_out=256, style_dim = style_dim)
        # 256, 16 x 16   
        self.conv3  = AdainResBlk(dim_in=256, dim_out=128, style_dim = style_dim)
        # 32 x 32
        self.conv4  = AdainResBlk(dim_in=128, dim_out=64, style_dim = style_dim)

        # 64 x 64

        self.conv5  = AdainResBlk(dim_in=64, dim_out=32, style_dim = style_dim)

        self.last_conv = nn.Conv2d(in_channels=32, out_channels=channels, kernel_size=1, bias = False)
        # 128, 128 x 128

        self.output = nn.Tanh()
        self.token = nn.Parameter(torch.randn(2, n_attr))


    def forward(self, x,  y, s):
        bs = x.shape[0]
        latent_token = self.token[y.long()]

        x = torch.cat((x, latent_token), dim = -1)

        x = self.linear(x)  ##1024, 16 x 16
        x = x.view(bs, 1024,4,4)

     #   x = torch.cat((x, en_ft[-1]), dim = 1)


        x = self.conv1(x, s)

    #    x = torch.cat((x, en_ft[-2]), dim = 1)
        x = self.conv2(x, s)
        
      #  x = torch.cat((x, en_ft[-3]), dim = 1)

        x = self.conv3(x, s)
        
       # x = torch.cat((x, en_ft[-4]), dim = 1)
        x = self.conv4(x, s)
        
        #x = torch.cat((x, en_ft[-5]), dim = 1)

        x = self.conv5(x, s)
        

        x = self.last_conv(x)

        return self.output(x)


