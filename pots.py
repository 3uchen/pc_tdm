import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
import faulthandler
# 在import之后直接添加以下启用代码即可
faulthandler.enable()
import numpy as np
import data_loading
import random
import visual
import torch
print(torch.__version__)
from sklearn.preprocessing import StandardScaler

from pypots.data import load_specific_dataset
from pypots.imputation import SAITS, GPVAE, Transformer, USGAN, CSDI, BRITS
from pypots.utils.metrics import cal_mae
from pygrinder import mcar, masked_fill
import matplotlib.pyplot as plt

color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
         '#e377c2', '#7f7f7f',
         '#bcbd22', '#17becf']


def curve_visual(data, masks, cnn_result, mask_scheme):
    fig, axes = plt.subplots(6, 5, figsize=(12, 8))
    print(data.shape)
    fig.suptitle('Result of Imputation', y=0.99)
    x = np.arange(data.shape[1])
    sample_no = random.sample(range(len(data)), 6)
    for i in range(6):
        ori = data[sample_no[i]]
        mask = masks[sample_no[i]]
        cnn_impute = cnn_result[sample_no[i]]
        has_zeros = np.any(mask == 0, axis=0)
        masked_feature_no = np.where(has_zeros)[0]
        for j in range(5):
            ax = axes[i, j]
            ax.plot(x, ori[:, masked_feature_no[j]], linestyle="-", label=f'real')   
            feature = masked_feature_no[j]
            mask_feature = mask[:, feature]
            zero_indices = [index for index, value in enumerate(mask_feature) if value == 0]
            cnn_ = cnn_impute[zero_indices, feature]
            if mask_scheme == 'RM':
                ax.scatter(zero_indices, cnn_, c=color[2], marker="o", label=f'impute')
            else:
                x_mask = np.arange(zero_indices[0], zero_indices[0]+len(zero_indices))
                ax.plot(x_mask, cnn_impute[zero_indices, feature], c=color[2], label=f'impute' )
            ax.set_xlim(0, data.shape[1])
            
            if i == 2:
                ax.set_xlabel('timestamp', labelpad=10)  # labelpad调整标签与轴的距离
            else:
                ax.set_xticks([])  # 不显示x刻度

            if j == 0:
                ax.set_ylabel('y', rotation=0, labelpad=15)  # rotation调整标签旋转角度
            else:
                ax.set_yticks([])  # 不显示y刻度
    plt.tight_layout()
    plt.legend()
    plt.show(block = True)


def masked(masked_scheme, masked_ratio, data_set):
    nums, seq_len, dim = data_set.shape
    feature_masked_num = int(dim*masked_ratio[1])
    time_masked_num = int(seq_len*masked_ratio[0])
    masks = np.ones_like(data_set)
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
    return masks, masks*data_set


# X = data_loading.real_data_loading(data_name='furnace', seq_len=64)
# train_set = X[:15000]
# np.save("dataset/train.npy", train_set)
# test_set = X[15000:15100]
# np.save("dataset/test.npy", test_set)
seq_len = 64
miss_ratios = [[0.2, 0.5], [0.4, 0.5], [0.6, 0.5], [0.8, 0.5]]
train_set = np.load("../dataset/seq_{}_xy/gen_model_train.npy".format(seq_len))
test_gt, test_mask, test_masked_data= [], [], []
for miss_ratio in miss_ratios:
    test_set = np.load('../dataset/seq_{}_xy/miss_test_{}_{}.npy'.format(seq_len, miss_ratio[0], miss_ratio[1]))
    test_gt.append(test_set[:2000])
    test_mask.append(test_set[2000:4000])
    test_masked_data.append(test_set[4000:])


# train_intact, train_set, train_missing_mask, train_indicating_mask = mcar(train_set, 0.1)
# train_set = masked_fill(train_set, 1 - train_missing_mask, np.nan)
# train_dataset = {"X":train_set}

# test_intact, test_set, test_missing_mask, test_indicating_mask = mcar(test_set, 0.4)
# test_set = masked_fill(test_set, 1 - test_missing_mask, np.nan)
# test_dataset = {"X":test_set}

train_mask, train_masked_data = masked('RBM', [0.5, 0.5], train_set)
train_masked_data[train_masked_data == 0] = np.nan
train_dataset = {"X": train_masked_data}






# X_intact, X, missing_mask, indicating_mask = mcar(X, 0.1) # hold out 10% observed values as ground truth
# X = masked_fill(X, 1 - missing_mask, np.nan)
# dataset = {"X": X}
# print(dataset["X"].shape, dataset["X"], type(dataset["X"]))  # (11988, 48, 37), 11988 samples, 48 time steps, 37 features
# np.savetxt('dataset.txt', dataset["X"][0])
# Model training. This is PyPOTS showtime.
# saits = SAITS(n_steps=train_set.shape[1], n_features=train_set.shape[2], n_layers=2, d_model=256, d_inner=128, n_heads=4, d_k=64, d_v=64, dropout=0.1, epochs=80)
#saits = USGAN(n_steps=train_set.shape[1], n_features=train_set.shape[2], n_layers=2, d_model=256, d_inner=128, n_heads=4, d_k=64, d_v=64, dropout=0.1, epochs=80)
# saits = CSDI(n_features=train_set.shape[2], n_layers=4, n_heads=4, n_channels=3, d_time_embedding=64, d_feature_embedding=64, d_diffusion_embedding=64, epochs=80)
# saits = GPVAE(n_steps=train_set.shape[1], n_features=train_set.shape[2], latent_size=64, window_size=3, epochs=50)
saits = BRITS(n_steps=train_set.shape[1], n_features=train_set.shape[2], rnn_hidden_size=64, epochs=50)
# Here I use the whole dataset as the training set because ground truth is not visible to the model, you can also split it into train/val/test sets
saits.fit(train_dataset)
for i in range(len(miss_ratios)):
    test_masked_data_i = test_masked_data[i]
    test_masked_data_i[test_masked_data_i == 0] = np.nan
    test_dataset = {"X": test_masked_data_i}
    imputation = saits.impute(test_dataset)  # impute the originally-missing values and artificially-missing values
    mask = np.int64(1) - test_mask[i]
    rmse = np.sqrt(np.mean((imputation * mask - test_gt[i] * mask) ** 2))
    mae = np.mean(np.abs(imputation * mask - test_gt[i] * mask))
    visual.curve_visual(test_gt[i], test_mask[i], imputation, mask_scheme='RBM', save_path='../result/seq_{}/baseline/brits_{}_{}.png'.format(seq_len, miss_ratios[i][0], miss_ratios[i][1]), rmse=round(rmse, 4))
    np.save('../result/seq_{}/baseline/brits_{}_{}.npy'.format(seq_len, miss_ratios[i][0], miss_ratios[i][1]), imputation)
    print("seq_{}_brits_{}_{},  rmse: {}, mae: {}".format(seq_len, miss_ratios[i][0], miss_ratios[i][1], round(rmse, 4), round(mae, 4)) )

     # calculate mean absolute error on the ground truth (artificially-missing values)

# print(mae)



