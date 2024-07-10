import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文显示
plt.rcParams['axes.unicode_minus'] = False

color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
         '#e377c2', '#7f7f7f',
         '#bcbd22', '#17becf']

def curve_visual(data, masks, result, mask_scheme, save_path, rmse = 0.00):
    fig, axes = plt.subplots(6, 5, figsize=(12, 8))
    # sample_ori = np.load('../dataset/seq_64_xy/miss_test_0.4_0.5.npy')[366]
    # sample_ori = np.load('../dataset/seq_64_xy/miss_test_0.4_0.5.npy')[366]
    # mask = np.load('../dataset/seq_64_xy/miss_test_0.4_0.5.npy')[366+2000]
    # mask_test_ = np.int64(1) - mask
    # sample_tdm = np.load("/data/anonym4/zc/ddcls/result/seq_64/testxy/tsdiff_0.4_0.5.npy")[366]
    # sample_tdm = mask_test_ * sample_tdm + sample_ori *mask
    # sample_brits = np.load("/data/anonym4/zc/ddcls/result/seq_64/baseline/brits_0.4_0.5.npy")[366]
    # sample_gpvae = np.load("/data/anonym4/zc/ddcls/result/seq_64/baseline/GPVAE_0.4_0.5.npy")[366]
    # sample_saits = np.load("/data/anonym4/zc/ddcls/result/seq_64/baseline/saits_0.4_0.5.npy")[366]
    fig.suptitle('Result of Imputation (RMSE: {})'.format(rmse), y=0.99)
    x = np.arange(data.shape[1])
    sample_no = [60, 70, 110, 120, 160, 170]
    for i in range(6):
        ori = data[sample_no[i]]
        mask = masks[sample_no[i]]
        cnn_impute = result[sample_no[i]]
        has_zeros = np.any(mask == 0, axis=0)
        masked_feature_no = np.where(has_zeros)[0]
        print(masked_feature_no)
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
            
            if i == 5 and j == 2:
                ax.set_xlabel('timestamp', labelpad=10)  # labelpad调整标签与轴的距离
            if j == 0 and i == 3:
                ax.set_ylabel('y', rotation=0, labelpad=15)  # rotation调整标签旋转角度
            if i == 0 and j == 4:
                ax.legend()
            # ax.set_yticks([])  # 不显示y刻度
    plt.tight_layout()
    # plt.show()
    # plt.legend()
    # plt.savefig(save_path)


def gen_visual(ori_data, generated_data, analysis, save_path):
    """Using PCA or tSNE for generated and original data visualization.
  
  Args:
    - ori_data: original data
    - generated_data: generated synthetic data
    - analysis: tsne or pca
  """
    # Analysis sample size (for faster computation)
    anal_sample_no = min([len(generated_data), len(ori_data)])
    idx = np.random.permutation(anal_sample_no)[:anal_sample_no]

    # Data preprocessing
    ori_data = np.asarray(ori_data)
    # generated_data = np.asarray(generated_data)

    ori_data = ori_data[idx]
    generated_data = generated_data[idx]

    no, seq_len, dim = ori_data.shape

    for i in range(anal_sample_no):
        if (i == 0):
            prep_data = np.reshape(np.mean(ori_data[0, :, :], 1), [1, seq_len])
            prep_data_hat = np.reshape(np.mean(generated_data[0, :, :], 1),
                                       [1, seq_len])
        else:
            prep_data = np.concatenate(
                (prep_data,
                 np.reshape(np.mean(ori_data[i, :, :], 1), [1, seq_len])))
            prep_data_hat = np.concatenate(
                (prep_data_hat,
                 np.reshape(np.mean(generated_data[i, :, :], 1),
                            [1, seq_len])))

    # Visualization parameter
    colors = ["red" for i in range(anal_sample_no)
              ] + ["blue" for i in range(anal_sample_no)]

    if analysis == 'pca':
        # PCA Analysis
        pca = PCA(n_components=2)
        pca.fit(prep_data)
        pca_results = pca.transform(prep_data)
        pca_hat_results = pca.transform(prep_data_hat)

        # Plotting
        f, ax = plt.subplots(1)
        plt.scatter(pca_results[:, 0],
                    pca_results[:, 1],
                    color = 'red',
                    marker='o',
                    alpha=0.7,
                    label="real data")
        plt.scatter(pca_hat_results[:, 0],
                    pca_hat_results[:, 1],
                    color = 'blue',
                    marker = 'o',
                    alpha=0.2,
                    label="generated data")

        ax.legend()
        plt.title('PCA plot')
        plt.xlabel('x-pca')
        plt.ylabel('y_pca')
        plt.savefig(save_path)

    elif analysis == 'tsne':

        # Do t-SNE Analysis together
        prep_data_final = np.concatenate((prep_data, prep_data_hat), axis=0)

        # TSNE anlaysis
        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
        tsne_results = tsne.fit_transform(prep_data_final)

        # Plotting
        f, ax = plt.subplots(1)

        plt.scatter(tsne_results[:anal_sample_no, 0],
                    tsne_results[:anal_sample_no, 1],
                    c='red',
                    alpha=0.7,
                    marker = 'o',
                    label="real data")
        plt.scatter(tsne_results[anal_sample_no:, 0],
                    tsne_results[anal_sample_no:, 1],
                    c='blue',
                    alpha=0.2,
                    marker='o',
                    label="generated data")

        ax.legend()

        plt.title('t-SNE')
        plt.xlabel('x-tsne')
        plt.ylabel('y_tsne')
        plt.savefig(save_path)
        # plt.show()

# sample_ori = np.load('../dataset/seq_64_xy/miss_test_0.4_0.5.npy')[366]
# mask = np.load('../dataset/seq_64_xy/miss_test_0.4_0.5.npy')[366+2000]
# mask_test_ = np.int64(1) - mask
# sample_tdm = np.load("/data/anonym4/zc/ddcls/result/seq_64/testxy/tsdiff_0.4_0.5.npy")[366]
# # sample_tdm = mask_test_ * sample_tdm + sample_ori *mask
# sample_brits = np.load("/data/anonym4/zc/ddcls/result/seq_64/baseline/brits_0.4_0.5.npy")[366]
# sample_gpvae = np.load("/data/anonym4/zc/ddcls/result/seq_64/baseline/GPVAE_0.4_0.5.npy")[366]
# sample_saits = np.load("/data/anonym4/zc/ddcls/result/seq_64/baseline/saits_0.4_0.5.npy")[366]
# # plt.colorbar()
# fig, axs = plt.subplots(1, 6, figsize=(12,1.5))
# for i in range(6):
#     has_zeros = np.any(mask == 0, axis=0)
#     masked_feature_no = np.where(has_zeros)[0]
#     ax = axs[i]
#     x = np.arange(64)
#     ax.plot(x, sample_ori[:, masked_feature_no[i]], linestyle="-", label=f'real')   
#     feature = masked_feature_no[i]
#     mask_feature = mask[:, feature]
#     zero_indices = [index for index, value in enumerate(mask_feature) if value == 0]
#     cnn_ = sample_gpvae[zero_indices, feature]
#     x_mask = np.arange(zero_indices[0], zero_indices[0]+len(zero_indices))
#     ax.plot(x_mask,cnn_, c=color[2], label=f'imputed' )
#     ax.set_xlim(0, 64)
#     # if i == 0:
#     #     ax.set_ylabel('value', rotation=0, labelpad=15)  # rotation调整标签旋转角度
#     if i == 5:
#         ax.legend()
#     ax.set_yticks([])  # 不显示y刻度
#     plt.tight_layout()

# plt.subplot(5,1,1)
# plt.imshow(sample_ori, 
#             # cmap='gray', 
#             aspect='auto')
# plt.subplot(5,1,2)
# plt.imshow(sample_tdm, 
#             # cmap='gray', 
#             aspect='auto')
# plt.subplot(5,1,3)
# plt.imshow(sample_saits, 
#             # cmap='gray', 
#             aspect='auto')
# plt.subplot(5,1,4)
# plt.imshow(sample_brits, 
#             # cmap='gray', 
#             aspect='auto')
# plt.subplot(5,1,5)
# m = plt.imshow(mask, 
#             # cmap='gray', 
#             aspect='auto')
# plt.colorbar(mappable=m)

# fig.suptitle("Original vs Reconstructed Data")
# fig.tight_layout()
# plt.show()

