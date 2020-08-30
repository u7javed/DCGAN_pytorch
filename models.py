import torch.nn as nn
import torch.nn.functional as F
import numpy as np

'''Discriminator'''
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        #input shape: [B, 3, 64, 64]
        self.pred = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=(5, 5), stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=(5, 5), stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=(5, 5), stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1024, kernel_size=(5, 5), stride=2, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(1024, 1, kernel_size=(4, 4), stride=2, padding=1, bias=False),
            nn.Sigmoid()

        )

    def forward(self, x):
        x = self.pred(x)
        return x


'''Generator'''
class Generator(nn.Module):
    def __init__(self, latent_size=100):
        super(Generator, self).__init__()

        self.conv = nn.Sequential(
            nn.ConvTranspose2d(100, 1042, kernel_size=(4, 4), stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1042),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(1042, 512, kernel_size=(4, 4), stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(512, 256, kernel_size=(4, 4), stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 3, kernel_size=(4, 4), stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1, 1, 1) # = [B, latent_size + embedding_dim, 1, 1]
        x = self.conv(x)
        return x