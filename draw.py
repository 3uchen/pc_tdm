from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import math

color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
         '#e377c2', '#7f7f7f',
         '#bcbd22', '#17becf']

label = ['repaint', 'fixed ratio', 'variation', 'x0_noise']
def rmse():
    valid_2 = [[0.0072, 0.0065, 0.0063, 0.0058], [0.0106, 0.0094, 0.0081, 0.0091], [0.0074, 0.0074, 0.0169, 0.0087]]
    valid_4 = [[0.0112, 0.0101, 0.0095, 0.0088], [0.0375, 0.0352, 0.0375, 0.0345], [0.0117, 0.0145, 0.0516, 0.0449]]
    valid_6 = [[0.0142, 0.0122, 0.0111, 0.0106], [0.061, 0.0815, 0.0755, 0.0707], [0.0142, 0.0278, 0.0932, 0.0857]]
    valid_8 = [[0.0178, 0.0153, 0.0153, 0.0144], [0.0833, 0.0978, 0.0836, 0.0837], [0.02, 0.0377, 0.1009, 0.1029]]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('RMSE')
    x = [1000, 2000, 3000, 4000]
    ax1 = axes[0, 0]
    ax1.set_title('(0.2, 0.5)')
    for i in range(3):
        ax1.plot(x, valid_2[i], marker='o', color=color[i], label=label[i])
    ax2 = axes[0, 1]
    ax2.set_title('(0.4, 0.5)')
    for i in range(3):
        ax2.plot(x, valid_4[i], marker='o', color=color[i], label=label[i])
    ax3 = axes[1, 0]
    ax3.set_title('(0.6, 0.5)')
    for i in range(3):
        ax3.plot(x, valid_6[i], marker='o', color=color[i], label=label[i])
    ax4 = axes[1, 1]
    ax4.set_title('(0.8, 0.5)')
    for i in range(3):
        ax4.plot(x, valid_8[i], marker='o', color=color[i], label=label[i])
    ax2.legend()
    ax3.set_xlabel('epoch', labelpad=4)
    ax4.set_xlabel('epoch', labelpad=4)
    ax1.set_xticks(x)
    ax2.set_xticks(x)
    ax3.set_xticks(x)
    ax4.set_xticks(x)
    plt.savefig('../result/seq_/compare/rmse.png')

def loss():
    loss_1 = np.load('/data/anonym4/zc/ddcls/result/seq_/train_nomask/loss.npy')[4:]
    loss_2 = np.load('/data/anonym4/zc/ddcls/result/seq_/train_mask_nor/loss.npy')[4:]
    loss_3 = np.load('/data/anonym4/zc/ddcls/result/seq_/train_mask_var/loss.npy')[4:]
    loss_4 = np.load('/data/anonym4/zc/ddcls/result/seq_/train_x0_noise/loss.npy')[4:]
    fig = plt.figure(figsize=(12,8))
    plt.title('Loss')
    x = np.arange(5,4001)
    plt.plot(x, loss_1, color=color[0], label=label[0])
    plt.plot(x, loss_2, color=color[1], label=label[1])
    plt.plot(x, loss_3, color=color[2], label=label[2])
    plt.plot(x, loss_4, color=color[3], label=label[3])
    plt.legend()
    plt.xlabel('epoch')
    plt.savefig('../result/seq_/compare/loss2.png')


def curve(epoch):
    no  = [60, 70, 90, 110, 120, 123]
    valid_gt = np.load('../dataset/simulation/miss_test_0.4_1.npy')[:2000]
    valid_mask = np.load('../dataset/simulation/miss_test_0.4_1.npy')[2000:4000]
    valid_masked_data = np.load('../dataset/simulation/miss_test_0.4_1.npy')
    fig, axes = plt.subplots(6, 5, figsize=(12, 8))
    # fig.suptitle('Result of Imputation', y=0.99)
    x = np.arange(64)
    valid = np.load('../result/test/TDM_PC_pre_3000_impute_0.4_1.npy')
    rmse = np.sqrt(np.mean((valid * valid_mask - valid_gt * valid_mask) ** 2))
    mae = np.mean(np.abs(valid * valid_mask - valid_gt * valid_mask))
    print(rmse,mae)
    for i in range(6):
        ori = valid_gt[no][i]
        mask = valid_mask[no][i]
        impute_1 = valid[no][i]
        has_zeros = np.any(mask == 0, axis=0)
        masked_feature_no = np.where(has_zeros)[0]
        for j in range(5):
            ax = axes[i, j]
            ax.plot(x, ori[:, masked_feature_no[j]], linestyle="-", color = 'green', label=f'real')   
            feature = masked_feature_no[j]
            mask_feature = mask[:, feature]
            zero_indices = [index for index, value in enumerate(mask_feature) if value == 0]
            x_mask = np.arange(zero_indices[0], zero_indices[0]+len(zero_indices))
            ax.plot(x_mask, impute_1[zero_indices, feature], linestyle=":", c='blue', label='predict' )
            ax.set_xlim(0, 64)
            
            if i == 5 and j == 2:
                ax.set_xlabel('timestamp', labelpad=10)  # labelpad调整标签与轴的距离
            if j == 0 and i == 3:
                ax.set_ylabel('y', rotation=0, labelpad=15)  # rotation调整标签旋转角度
            if i == 0 and j == 4:
                ax.legend() 
    plt.tight_layout()
    # plt.show()
    # plt.legend()
    plt.savefig('../result/test/TDM_PC_pre_3000_impute_0.4_1.png')

def scaler(ori_data):
    ori_data = StandardScaler().fit_transform(ori_data)
    ori_data = MinMaxScaler(feature_range=(-1, 1)).fit_transform(ori_data)
    return ori_data

def simlutaion_predict():
    x = np.linspace(-1, 1, 1000)
    e4 = np.random.randn(1000)
    x4 = 1 / (1 + np.exp(-x)) + e4
    x4 = scaler(x4.reshape(-1,1))
    e5 = np.random.randn(1000)
    x6 = np.sin(x) +e5
    x6 = scaler(x6.reshape(-1,1))
    y = [x4, x6]
    training_samples = np.load("../dataset/simulation/gen_model_train.npy")[:500]
    tdm_samples = np.load("../result/test/TDM_3000_gen.npy")[:500]
    tdm_pc_direct_samples = np.load("../result/test/TDM_PC_3000_gen.npy")[:500]
    tdm_pc_pre_samples = np.load("../result/test/TDM_PC_pre_3000_gen.npy")[:500]
    def sample_process(samples):
        samples_mean = samples[:, 0, :]
        x_plus = np.sum(samples_mean[:, :3], axis=1)
        y1 = samples_mean[:, 3]
        new_samples1 = np.stack((x_plus, y1), axis=1)
        y2 = samples_mean[:, 5]
        new_samples2 = np.stack((x_plus, y2), axis=1)
        return [new_samples1, new_samples2]
    training_samples1 = sample_process(training_samples)
    tdm_samples1 = sample_process(tdm_samples)
    tdm_pc_direct_samples1 = sample_process(tdm_pc_direct_samples)
    tdm_pc_pre_samples1 = sample_process(tdm_pc_pre_samples)
    for i in range(2):
        fig = plt.figure(figsize=(12,8))
        plt.plot(x, y[i], color='black', alpha=0.7, label='true')
        plt.scatter(training_samples1[i][:, 0], training_samples1[i][:, 1], color='red', label='training_samples')
        # plt.scatter(tdm_samples1[i][:, 0], tdm_samples1[i][:, 1], color='blue', label='TDM')
        #plt.scatter(tdm_pc_direct_samples1[i][:, 0], tdm_pc_direct_samples1[i][:, 1], color='orange', label='TDM_PC(direct train)')
        plt.scatter(tdm_pc_pre_samples1[i][:, 0], tdm_pc_pre_samples1[i][:, 1], color='green', label='TDM_PC(pre_train mode)')
        plt.xlabel('x1 + x2 + x3')
        plt.ylabel('y') 
        plt.legend()
        plt.grid(True, linestyle='--')
        plt.savefig("../result/test/sim_pred{}_TDM_PC_PRE_TRAIN.png".format(i+1))

simlutaion_predict()
# rmse()
# loss()
# curve(3000)

