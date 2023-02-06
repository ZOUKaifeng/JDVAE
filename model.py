

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import itertools
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
import numpy as np
from layers_v2 import AdainResBlk, ResBlk

import random

import torch.distributions as dist



def sample_gumbel(shape, eps=1e-20):
    unif = torch.rand(*shape).cuda()
    g = -torch.log(-torch.log(unif + eps))
    return g

def sample_gumbel_softmax(logits, g, temperature):
    """
        Input:
        logits: Tensor of log probs, shape = BS x k
        temperature = scalar
        
        Output: Tensor of values sampled from Gumbel softmax.
                These will tend towards a one-hot representation in the limit of temp -> 0
                shape = BS x k
    """
    #g = sample_gumbel(logits.shape)
    h = (g + logits)/temperature
    y = torch.softmax(h, dim = 1)
    return y





class Encoder_VAE(torch.nn.Module):
    def __init__(self, channels, latent_size = 64 ):
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
        self.token_size = 32

        self.conv_1 = ResBlk(dim_in = 3, dim_out = 64)

        # self.token_1 = nn.Parameter(torch.randn(2, 1, 64, 64, requires_grad=True))

        # # 128 x 32 x 32

        self.conv_2 = ResBlk(dim_in = 64, dim_out = 128)

        # self.token_2 = nn.Parameter(torch.randn(2, 1, 32, 32, requires_grad=True))


        # # 256 x 16 x 16
        self.conv_3 = ResBlk(dim_in = 128, dim_out = 256)

        # self.token_3 = nn.Parameter(torch.randn(2, 1, 16, 16, requires_grad=True))


        # # 1024 x 8 x 8
        self.conv_4 = ResBlk(dim_in = 256, dim_out = 512)

        # self.token_4 = nn.Parameter(torch.randn(2, 1, 8, 8, requires_grad=True))


        # 4 x 4
        self.conv_5 = ResBlk(dim_in = 512, dim_out = 512)

        self.mean = nn.Linear(512*4*4+self.token_size, latent_size)
        self.std = nn.Linear(512*4*4+self.token_size, latent_size)

        self.c = nn.Linear(512*4*4, 32)
        
        self.classifier =  nn.Linear(512*4*4, 2)
        self.temperature = 0.05

    def forward(self, x, y, supervise, test = False):
        bs = x.shape[0]
        

        x = self.conv_1(x)
        

        x = self.conv_2(x)

        x = self.conv_3(x)


        x = self.conv_4(x)


        x = self.conv_5(x)
        
        x = x.view(bs, -1)

        logit = self.classifier(x)
        attr = self.c(x)
  
        pred_y = y

        if test:
            pred_y = torch.argmax(torch.softmax(logit, dim =1), dim = 1)

        #prob = torch.softmax(coeff, dim =1) #(bs, 32) (32, 16)

        #attr = prob.unsqueeze(2)*self.token
        #attr = torch.sum(attr, dim = 1)
        
        h = torch.cat((x, attr), dim = -1)

        mu = self.mean(h)
        log_var = self.std(h)

        
        return mu, log_var, attr, logit




class Decoder_VAE(torch.nn.Module):
    def __init__(self, channels, n_attr = 10, latent_size = 64, style_dim = 256):
        super().__init__()


        self.linear = nn.Linear(latent_size+n_attr+32, 512*4*4)

        self.token_1 = nn.Parameter(torch.randn(2, 1, 4, 4, requires_grad=True))
        # 4x4   
        self.conv1 = AdainResBlk(dim_in=513, dim_out=256, style_dim = style_dim)

        self.token_2 = nn.Parameter(torch.randn(2, 1, 8, 8, requires_grad=True))

        self.conv2 = AdainResBlk(dim_in=257, dim_out=128, style_dim = style_dim)
        
        self.token_3 = nn.Parameter(torch.randn(2, 1, 16, 16, requires_grad=True))  


        self.conv3  = AdainResBlk(dim_in=129, dim_out=64, style_dim = style_dim)

        self.token_4 = nn.Parameter(torch.randn(2, 1, 32, 32, requires_grad=True))  
        # 32 x 32
        self.conv4  = AdainResBlk(dim_in=65, dim_out=32, style_dim = style_dim)

        # 64 x 64

        self.conv5  = AdainResBlk(dim_in=32, dim_out=16, style_dim = style_dim)

        self.last_conv = nn.Conv2d(in_channels=16, out_channels=channels, kernel_size=1, bias = False)
        # 128, 128 x 128

        self.output = nn.Tanh()
        self.token = nn.Parameter(torch.randn(2, n_attr))

    def forward(self, x,y_attr,y, s):
        bs = x.shape[0]

        latent_token = self.token[y.long()]



        x = torch.cat((x, latent_token), dim = -1)

        y_attr = y_attr* y.unsqueeze(1)

        x = torch.cat((x, y_attr), dim = -1)

        x = self.linear(x)  ##1024, 16 x 16
        x = x.view(bs, 512,4,4)

        token_1 = self.token_1[y.long()]
        x = torch.cat((x, token_1), dim = 1)

        x = self.conv1(x, s)

        token_2 = self.token_2[y.long()]
        x = torch.cat((x, token_2), dim = 1)

        x = self.conv2(x, s)
        
        token_3 = self.token_3[y.long()]
        x = torch.cat((x, token_3), dim = 1)

        x = self.conv3(x, s)
        
        token_4 = self.token_4[y.long()]
        x = torch.cat((x, token_4), dim = 1)

        x = self.conv4(x, s)
        

        x = self.conv5(x, s)
        

        x = self.last_conv(x)

        return self.output(x)






def KLD(mu, log_sigma):
    return -0.5 * torch.sum(1 + log_sigma - mu.pow(2) - (log_sigma).exp(), -1)

def gaussian_nll(mu, log_sigma, x):
    return 0.5 * torch.pow((x - mu) / log_sigma.exp(), 2) + log_sigma + 0.5 * np.log(2 * np.pi)

class semiVAE(torch.nn.Module):
    def __init__(self, latent_size = 64, n_attr = 2, pool_size = 50):
        super().__init__()
        print("Building model DVAE...")

        self.Q = Encoder_VAE(channels = 4, latent_size = latent_size)
        self.G = Decoder_VAE(channels = 3, n_attr = n_attr, latent_size = latent_size, style_dim = latent_size)
         
        self.latent_size = latent_size
 
        self.n_attr = n_attr

        self.sample_attr = []
        self.optimizer_G = torch.optim.Adam(itertools.chain(self.G.parameters(), self.Q.parameters()), lr=0.0001)
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, x, y):
        bs, c, h, w = x.shape
        self.device = x.device  
        input_ = x

        mean, log_var, y_attr, logit = self.Q(x, y, True, True)
        correct = self.cal_correct(logit, y)
        pred_y = (torch.argmax(torch.softmax(logit, dim =1), dim = 1)).int()
        z = mean

        recon_x = self.G(z, y_attr, pred_y, mean)


        vae_loss = self.loss_funtion(recon_x, mean, log_var, x)
        loss = vae_loss

        for i, l in enumerate(y):
            if l.item() == 1:

                self.sample_attr.append(y_attr[i])

            if len(self.sample_attr) == 50:
                self.sample_attr.pop(0)

        oppo_y = 1-y
        oppo_attr = torch.zeros((bs, 32)).cuda()
        for i in range(bs):
            s = random.sample(self.sample_attr, 1)[0].detach()
            oppo_attr[i] =s 
        oppo_x = self.G(z,oppo_attr, oppo_y, mean)


        return recon_x, oppo_x, correct, vae_loss





    def reparameterize(self, mu, logvar):

        batch_size = mu.shape[0]
        dim = logvar.shape[-1]

        std = torch.exp(logvar * 0.5)
  
        z = torch.normal(mean = 0, std = 1, size=(batch_size, dim)).to(mu.device)
        z = z*std + mu


        return z

    def loss_funtion(self,  x, mu, log_var, real_x):
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        #log_sigma = torch.tensor([0]).to(self.device) 
       # gaussian = gaussian_nll(real_x, log_sigma, x)
        gaussian = F.mse_loss(real_x, x)
       # gaussian = gaussian.sum(-1).sum(-1).sum(-1).mean()
        loss =  0.0001*kld_loss + gaussian
        return loss

    def cal_correct(self, logit, y):
        pred = (torch.argmax(torch.softmax(logit, dim =1),  dim = 1)).int().squeeze()
        correct = (pred == y).sum()

        return correct


    def optimize_parameters(self,  x, y, supervise = True):

        self.results = {}
        self.optimizer_G.zero_grad() 

        
        mean, log_var, y_attr, logit = self.Q(x, y, supervise)
        y = y
        p = torch.softmax(logit, dim =1)


        cls_loss = - p[torch.arange(p.size(0)), y.long()].mean()#self.bce(logit, y.unsqueeze(1).float())


        z = self.reparameterize(mean, log_var)


        recon_x = self.G(z,y_attr, y, mean)

        vae_loss = self.loss_funtion(recon_x, mean, log_var, x)
        loss = vae_loss + cls_loss

        loss.backward()
        self.optimizer_G.step() 

        self.results["gaussian"] = vae_loss
        self.results["kld"] = cls_loss
        self.results["loss"] = loss

        
        return self.results



