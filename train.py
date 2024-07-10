import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
import torch
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

# import visual

color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
         '#e377c2', '#7f7f7f',
         '#bcbd22', '#17becf']

def masked(masked_scheme, masked_ratio, data_set): 
    nums, _, seq_len, dim = data_set.shape
    feature_masked_num = int(dim * masked_ratio[1])
    time_masked_num = int(seq_len * masked_ratio[0])
    masks = torch.ones_like(data_set)
    assert masked_scheme in ['RM', 'RBM', 'BM']       
    for mask in masks:
        feature_masked = random.sample(range(dim), feature_masked_num)
        if masked_scheme == 'BM':
            start = random.randint(0, seq_len - time_masked_num)
            mask[start:start + time_masked_num, :] = 0
        else:
            for feature in feature_masked:
                if masked_scheme == 'RM':
                    t_masked = random.sample(range(seq_len), time_masked_num)
                    mask[t_masked, feature] = 0
                elif masked_scheme == 'RBM':
                    start = random.randint(0, seq_len - time_masked_num)
                    mask[start:start + time_masked_num, feature] = 0
    masked_data_set = masks * data_set
    return masks, masked_data_set  


def mix_dataset(data_list):
    idx = np.random.permutation(len(data_list))
    data = []
    for i in range(len(data_list)):
        data.append(data_list[idx[i]])
    return np.array(data)


PC = True
batch_size = 64
train_lr = 1e-5
seq_len = 64
miss_ratio = [0.4, 1]
ismasked = False
adam_betas = (0.9, 0.99)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
# data = data_loading.real_data_loading(data_name='furnace', seq_len=64)
# train_list = data[:15000]
# test_list = data[15000:]
# train_data = mix_dataset(train_list)
# test_data = mix_dataset(test_list)
# np.save("dataset/not_shuffled/test.npy", test_data)
# np.save("dataset/not_shuffled/train.npy", train_data)
train_set = np.load("../dataset/simulation/gen_model_train.npy")
test_data = np.load('../dataset/simulation/miss_test_{}_{}.npy'.format(miss_ratio[0], miss_ratio[1]))
gcn_encoder = torch.load('gcn_encoder.pth')
model = Unet(dim=64, dim_mults=(1, 2), channels=1, self_condition=False, PC=PC)
model.GCN_Encoder.load_state_dict(gcn_encoder)
model = model.to(device)
for param in model.GCN_Encoder.parameters():
    param.requires_grad = False

if ismasked:
    diffusion = GaussianDiffusion(model=model, device=device, timesteps=400, sampling_timesteps=200, beta_schedule='linear', objective='pred_noise', PC=PC )
else:
    diffusion = GaussianDiffusion(model=model, device=device, timesteps=400, sampling_timesteps=200, beta_schedule='linear', objective='pred_noise', PC=PC )
train_data = train_set[:3968]
dataset = torch.tensor(train_data)
dataset = dataset.view(train_data.shape[0], 1, train_data.shape[1], train_data.shape[2])
print(train_data.shape[0])
raw_data = np.load("../dataset/simulation/raw_data.npy")[:2048]
raw_data = raw_data[:, :-1]
raw_data = torch.tensor(raw_data).to(device)
raw_data = raw_data.view(raw_data.shape[0], raw_data.shape[1], 1)
raw_dataloader = DataLoader(dataset=raw_data, batch_size=batch_size, shuffle=True)
ori_dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
opt = torch.optim.Adam(filter(lambda p : p.requires_grad, diffusion.parameters()), lr = train_lr, betas = adam_betas)
losses = []
loss_epoch = []
val_rmse = 1000.0
for epoch in range(1, 4001):
    for batch, data in enumerate(zip(ori_dataloader, cycle(raw_dataloader))):
        ori_ = data[0]
        raw_ = data[1]
        ori_ = ori_.float().to(device)
        if PC:
            loss = diffusion(ori_, raw_, ismasked, epoch)
        else:
            loss = diffusion(ori_, ismasked, epoch)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    avg_loss = sum(losses[-len(ori_dataloader):])/len(ori_dataloader)
    print(f'Finished epoch {epoch}. Average loss for this epoch: {avg_loss:05f}')
    loss_epoch.append(avg_loss)
    if epoch % 200 == 0:
        # diffusion.eval()
        # save_path = "../result/seq_/train_mask_var/".format(seq_len)
        # rmse_test = impute_test(diffusion=diffusion, epoch=epoch, save_path=save_path)
        # if rmse_test < val_rmse:
        #     val_rmse = rmse_test
        #     torch.save(diffusion.state_dict(), '../result/seq_{}/train_mask_var/epoch_best_{}.pkl'.format(seq_len, epoch))
        # diffusion.train()
        torch.save(diffusion.state_dict(),  '../result/pre_mode/epoch_model_{}.pkl'.format(epoch))
loss_epoch = np.array(loss_epoch)
np.save('../result/pre_mode/loss.npy', loss_epoch)
        
    #     test_out = diffusion.impute(masked_data=masked_data_test, mask=mask_test)
    #     mask_test_ = np.int64(1) - mask_test
    #     impute_data = mask_test_*test_out + masked_data_test
    #     curve_visual(test_data, mask_test, impute_data, epoch=epoch)
            # loop.set_description(f'Epoch[{epoch}/{self.num_epoch}]')
            # loop.set_postfix(loss=avg_loss)
# torch.save(diffusion.state_dict(), 'masked_repaint_result/trained_model/ddpm_1000_noshuffle.pkl')
      

