import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
os.environ['CUDA_LAUNCH_BLOCKING'] = "3"
import torch
from pc_diff import Unet, GaussianDiffusion
import numpy as np
import visual
from torch.utils.data import DataLoader
import random
import matplotlib.pyplot as plt




    
    
def impute_test(diffusion):
    seq_len = 64
    miss_ratios =[[0.6, 0.5], [0.6, 0.5]]
    for miss_ratio in miss_ratios:
        test_set = np.load('../dataset/seq_/miss_test_{}_{}.npy'.format(miss_ratio[0], miss_ratio[1]))
        test_gt = test_set[:2000]
        test_mask = test_set[2000:4000]
        test_masked_data = test_set[4000:]

        test_out = diffusion.repaint(masked_data=test_masked_data, mask=test_mask)
        mask_test_ = np.int64(1) - test_mask
        rmse = np.sqrt(np.mean((test_out * mask_test_ - test_gt * mask_test_) ** 2))
        mae = np.mean(np.abs(test_out * mask_test_ - test_gt * mask_test_))
        visual.curve_visual(test_gt, test_mask, test_out, mask_scheme='RBM', save_path='../result/seq_{}/test_mask_var/tsdiff_{}_{}_e2000.png'.format(seq_len, miss_ratio[0], miss_ratio[1]), rmse=round(rmse, 4))
        np.save('../result/seq_{}/test_mask_var/tsdiff_{}_{}_e2000.npy'.format(seq_len, miss_ratio[0], miss_ratio[1]), test_out)
        print("seq_{}_tsdiff_{}_{},  rmse: {}, mae: {}".format(seq_len, miss_ratio[0], miss_ratio[1], round(rmse, 4), round(mae, 4)))

def impute_valid(diffusion, raw_data, epoch = 4000, scheme = 'test_mask_var'):
    # test_set_0_2 = np.load('../dataset/seq_/miss_test_{}_{}.npy'.format(0.2, 0.5))
    # valid_gt = test_set_0_2[:50]
    # valid_mask = test_set_0_2[2000:2050]
    # valid_masked_data = test_set_0_2[4000:4050]
    miss_ratios = [ [0.4, 1], [0.6,1], [0.8, 1]]
    for miss_ratio in miss_ratios:
        test_set = np.load('../dataset/simulation/miss_test_{}_{}.npy'.format(miss_ratio[0], miss_ratio[1]))
        valid_gt = test_set[:2000]
        valid_mask = test_set[2000:4000]
        valid_masked_data = test_set[4000:]
        valid_out = diffusion.repaint(masked_data=valid_masked_data, mask=valid_mask, raw_data=raw_data)
        np.save('../result/test/{}_{}_impute_{}_{}.npy'.format(scheme, epoch, miss_ratio[0], miss_ratio[1]), valid_out)
    # save_path = '../result/traffic/{}/valid_e{}_impute.png'.format(scheme, epoch)
    # valid_gt = np.load('../dataset/seq_/valid_gt.npy')
    # valid_mask = np.load('../dataset/seq_/valid_mask.npy')
    # valid_masked_data = np.load('../dataset/seq_/valid_masked_data.npy')
    # valid_out = diffusion.repaint(masked_data=valid_masked_data, mask=valid_mask)
    # # valid_out = np.load('../result/seq_/{}/valid_e{}_impute.npy'.format(scheme, epoch))
    # mask_valid_ = np.int64(1) - valid_mask
    # visual.curve_visual(valid_gt, valid_mask, valid_out, mask_scheme='RBM', save_path=save_path)
    # np.save('../result/seq_/{}/valid_e{}_impute.npy'.format(scheme, epoch), valid_out)
    # for i in range(4):
    #     rmse = np.sqrt(np.mean((valid_out[i*50: i*50 + 50] * mask_valid_[i*50: i*50 + 50] - valid_gt[i*50: i*50 + 50] * mask_valid_[i*50: i*50 + 50]) ** 2))
    #     mae = np.mean(np.abs(valid_out[i*50: i*50 + 50] * mask_valid_[i*50: i*50 + 50] - valid_gt[i*50: i*50 + 50] * mask_valid_[i*50: i*50 + 50]))
    #     print("valid_{}_{},  rmse: {}, mae: {}".format(i*0.2 + 0.2, 0.5, round(rmse, 4), round(mae, 4)))


def generate_valid(diffusion, raw_data, ori_data, epoch, scheme):
    seq_len = ori_data.shape[1]
    data_dim = ori_data.shape[2]
    gen_data = diffusion.sample(raw_data, seq_len, data_dim, 2000)
    # gen_data = np.load('../result/test/TDM_1000_gen.npy')
    np.save('../result/test/{}_{}_gen.npy'.format(scheme, epoch), gen_data)
    visual.gen_visual(ori_data, gen_data, 'pca', save_path='../result/test/{}_{}_gen_pca.png'.format(scheme, epoch))
    visual.gen_visual(ori_data, gen_data, 'tsne', save_path='../result/test/{}_{}_gen_tsne.png'.format(scheme, epoch))

def valid():
    train_scheme = 'TDM'
    test_scheme = 'TDM_PC_pre'
    PC = True
    epoch = 3000
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # train_data = data_loading.real_data_loading(data_name='CO_2', seq_len=64, train_=True)
    seq_len = 64
    model = Unet(dim=64, dim_mults=(1, 2), channels=1, self_condition=False, PC=PC)
    model = model.to(device)
    if PC:
        print("plus pc")
        raw_data = np.load("../dataset/simulation/raw_data.npy")[:2000]
        raw_data = raw_data[:, :-1]
        raw_data = torch.tensor(raw_data).float()
        raw_data = raw_data.view(raw_data.shape[0], raw_data.shape[1], 1)
        diffusion = GaussianDiffusion(model=model, device=device, timesteps=400, sampling_timesteps=200, beta_schedule='linear', objective='pred_noise', PC=PC)
        diffusion.load_state_dict(torch.load('../result/pre_mode/epoch_model_{}.pkl'.format(epoch)))
    else:
        raw_data = None
        diffusion = GaussianDiffusion(model=model, device=device, timesteps=400, sampling_timesteps=200, beta_schedule='linear', objective='pred_noise', PC=PC)
        diffusion.load_state_dict(torch.load('../result/simulation/epoch_model_{}.pkl'.format(epoch)))
    # generate_test(diffusion, np.load('dataset/shuffled/train.npy'))
    impute_valid(diffusion, raw_data, epoch, test_scheme)
    train_set = np.load("../dataset/simulation/gen_model_train.npy")[:2000]
    # generate_valid(diffusion, raw_data, train_set, epoch, test_scheme)






def generate_test(diffusion, ori_data):
    seq_len = ori_data.shape[1]
    data_dim = ori_data.shape[2]
    gen_data = diffusion.sample(seq_len, data_dim, 2000)
    np.save('../result/seq_{}/gen/mask_var_2000.npy'.format(seq_len), gen_data)
    visual.gen_visual(ori_data, gen_data, 'pca', save_path='../result/seq_{}/gen/mask_var_pca_2000.png'.format(seq_len))
    visual.gen_visual(ori_data, gen_data, 'tsne', save_path='../result/seq_{}/gen/mask_var_tsne_2000.png'.format(seq_len))


def show():
    seq_len = 64
    test_set = np.load('../dataset/seq_64/miss_test_0.4_0.5.npy')
    test_out = np.load('../result/seq_64/test_nomask/tsdiff_0.4_0.5_e1000.npy')
    test_gt = test_set[:2000]
    test_mask = test_set[2000:4000]
    test_masked_data = test_set[4000:]
    rmse = np.sqrt(np.mean((test_out * test_mask - test_gt * test_mask) ** 2))
    visual.curve_visual(test_gt, test_mask, test_out, mask_scheme='RBM', save_path='../result/seq_{}/test_nomask/tsdiff_{}_{}_e1000_3.png'.format(seq_len, 0.4, 0.5), rmse=round(rmse, 4))




valid()
# show()
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# # device = 'cpu'
# print(device)
# # train_data = data_loading.real_data_loading(data_name='CO_2', seq_len=64, train_=True)
# seq_len = 64
# model = Unet(dim=64, dim_mults=(1, 2, 4), channels=1, self_condition=False)
# model = model.to(device)
# diffusion = GaussianDiffusion(model=model, device=device, timesteps=400, sampling_timesteps=200, beta_schedule='linear', objective='pred_noise' )
# diffusion.load_state_dict(torch.load('../result/seq_{}/train_mask_var/epoch_model_2000.pkl'.format(seq_len)))
# # generate_test(diffusion, np.load('dataset/shuffled/train.npy'))
# impute_test(diffusion)
# train_set = np.load("../dataset/seq_{}/gen_model_train.npy".format(seq_len))[:2000]
# generate_test(diffusion, train_set)




