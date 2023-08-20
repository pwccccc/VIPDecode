import numpy as np
import torch 
from torch import nn

from models.generator import Generator
from models.discriminator import Discriminator
from dataset import Data
import scipy.io as scio
from torch.utils.data import DataLoader
from utils import add_bin
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import cfg

import matplotlib.pyplot as plt

def main(args):

    # datam = scio.loadmat('./sort_data/VIP_decode_data.mat')
    
    data = torch.load(args.train_path)
    batch_size = args.batch_size
    ge_bin_size = args.ge_bin_size
    latent_dim = args.latent_dim
    args.seq_len = args.seq_len // ge_bin_size
    generator = Generator(args)
    discriminator = Discriminator(args)

    MyDataset = Data(add_bin(data['data'], window_size=ge_bin_size), data['label'])
    data_loader = DataLoader(MyDataset, batch_size= batch_size, shuffle=True)

    g_optimizer = torch.optim.Adam(generator.parameters(), lr=args.g_lr, weight_decay=0.0001)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.d_lr, weight_decay=0.0001)

    loss_fn = nn.BCELoss()
    # loss_fn = nn.MSELoss()
    labels_one = torch.ones(batch_size, 1)
    labels_zeros = torch.zeros(batch_size, 1)

    d_losses = []
    g_losses = []
    for epoch in range(args.epoch):
        generator.train()
        discriminator.train()
        gelosses = 0.
        dilosses = 0.
        for i, batch in enumerate(data_loader):
            traindata, label = batch
            

            for _ in range(1):
                
                z = torch.randn(batch_size, latent_dim)
                pred = generator(z, (label-1).long())
                print(pred.shape)
                d_optimizer.zero_grad()
                real_loss = loss_fn(discriminator(traindata, (label-1).long()), labels_one)
                fake_loss = loss_fn(discriminator(pred.detach(), (label-1).long()), labels_zeros)
                d_loss = real_loss + fake_loss
                dilosses += d_loss.item()
                d_loss.backward()
                d_optimizer.step()
                del pred
                del d_loss
            for _ in range(5):
                z = torch.randn(batch_size, latent_dim)
                pred = generator(z, (label-1).long()).detach()
                g_optimizer.zero_grad()
                recons_loss = torch.abs(pred - traindata).mean()
                g_loss = recons_loss*0.05 + loss_fn(discriminator(pred, (label-1).long()), labels_one)
                gelosses += g_loss.item()
                g_loss.backward()
                g_optimizer.step()
                del g_loss
                del pred
            del traindata
            print("now the batch is {}".format(i))
        print('now the epoch is {}'.format(epoch))
        g_losses.append(gelosses/len(data_loader))
        d_losses.append(dilosses/len(data_loader))

    return generator

def generate(generator, args):
    data = torch.load(args.train_path)
    batch_size = data['data'].shape[0]
    latent_dim = args.latent_dim

    # generator = torch.load('./results/210 generator ge_bin_size_3.pth')

    # generator = Generator()
    # generator.load_state_dict(torch.load('generator.pth'))
    z = torch.randn(batch_size, latent_dim)
    label = torch.tensor(data['label'])
    result = []
    for _ in range(1):
        generator.eval()
        with torch.no_grad():
            pred = generator(z, (label-1).long())
            print(pred.shape)
            result.append(pred)
    
    torch.save(result, args.output_path)



if __name__ == "__main__":
    args = cfg.parse_args()
    generator = main(args)
    generate(generator, args)
           
