import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
import torch
import pc_diff
# from denoising_diffusion_pytorch import Unet, GaussianDiffusion
from pc_diff import Unet, GaussianDiffusion
import numpy as np
# import visual
from torch.utils.data import DataLoader
import random
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from itertools import cycle
import torch.utils.data as data


def gcn_preprocess(adj: np.ndarray) -> np.ndarray:
    n_node = len(adj)
    adj_tilde = adj + np.eye(n_node, dtype=np.int32)
    diag = adj_tilde.sum(axis=1)
    diag = 1 / (diag ** (0.5))
    D_tilde = np.diag(diag)
    pre = D_tilde @ adj_tilde @ D_tilde
    return torch.tensor(pre, dtype=torch.float32).to("cuda")    

def pre_train_step():
    batch_size = 64
    train_lr = 1e-3
    device = "cuda"
    raw_data = np.load("../dataset/simulation/raw_data.npy")[:2048]
    adj = np.array([[0, 0, 0, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 0, 0, 0, 0, 0, 1],
                    [1, 1, 1, 0, 0, 0, 0, 0, 1],
                    [1, 1, 1, 0, 0, 0, 0, 0, 1],
                    [1, 1, 1, 0, 0, 0, 0, 0, 1],
                    [1, 1, 1, 0, 0, 0, 0, 0, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 0]])
    adj_norm = gcn_preprocess(adj)
    train_x = raw_data[:,:-1]
    train_y = raw_data[:, -1]
    train_x = torch.tensor(train_x).to(device)
    train_x = train_x.view(train_x.shape[0], train_x.shape[1], 1)
    train_y = torch.tensor(train_y).to(device)
    train_dataset = data.TensorDataset(train_x, train_y)
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    gcn_encoder = pc_diff.G_Net(in_channels=9, hidden_channels=64, out_channels=32*4).to(device=device)
    gcn_encoder.train()
    optimizer = torch.optim.Adam(gcn_encoder.parameters(), lr=train_lr, betas=(0.9,0.99))
    loss = torch.nn.MSELoss().to(device=device)
    for epoch in range(450):
        losses = []
        for step, (batch_x, batch_y) in enumerate(train_loader):
            optimizer.zero_grad()
            batch_x = batch_x.to(device)
            adjs = [adj_norm for _ in range(batch_size)]
            adjs = torch.stack(adjs)
            graph_data = [batch_x, adj_norm]
            batch_y = batch_y.to(device)
            out = gcn_encoder(graph_data, pre_train=True).view(-1)
            # print(out, batch_y)
            los = loss(out.float(), batch_y.float())
            los = los.to(device)
            los.backward()
            optimizer.step()
            losses.append(los.item()/batch_x.size(0))
        print('Epoch:{}, TrainLoss:{:.5f}'.format(epoch+1, np.mean(losses)))
    torch.save(gcn_encoder.state_dict(), 'gcn_encoder_32.pth')


pre_train_step()