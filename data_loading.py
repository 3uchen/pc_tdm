import numpy as np
import scipy.io as scio
import random
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import math
# def MinMaxScaler(data):
#     """Min Max normalizer.
  
#   Args:
#     - data: original data
  
#   Returns:
#     - norm_data: normalized data
#   """
#     numerator = data - np.min(data, 0)
#     denominator = np.max(data, 0) - np.min(data, 0)
#     norm_data = numerator / (denominator + 1e-7)
#     return norm_data


def sine_data_generation(no, seq_len, dim):
    """Sine data generation.
  
  Args:
    - no: the number of samples
    - seq_len: sequence length of the time-series
    - dim: feature dimensions
    
  Returns:
    - data: generated data
  """
    # Initialize the output
    data = list()

    # Generate sine data
    for i in range(no):
        # Initialize each time-series
        temp = list()
        # For each feature
        for k in range(dim):
            # Randomly drawn frequency and phase
            freq = np.random.uniform(0, 0.1)
            phase = np.random.uniform(0, 0.1)

            # Generate sine signal based on the drawn frequency and phase
            temp_data = [np.sin(freq * j + phase) for j in range(seq_len)]
            temp.append(temp_data)

        # Align row/column
        temp = np.transpose(np.asarray(temp))
        # Normalize to [0,1]
        temp = (temp + 1) * 0.5
        # Stack the generated data
        data.append(temp)

    return data


def simulation_data(no):
    data = np.ones((no, 10))
    for i in range(no):
        x1 = np.random.normal(0, 1)
        x2 = np.random.normal(0, 1)
        x3 = np.random.normal(0, 1)
        x4 = 1 / (1 + math.exp(-x1-x2-x3))
        x5 = max(1 + x1 + x2 + x3, 0)
        x6 = np.sin(x1 + x2 + x3)
        x7 = x1 + x2 * x2 + x3 ** 3
        x8 = np.tanh(x1 + x2 + x3) + 1
        x9 = np.sin(x1+x2+x3) + np.cos(x4+x5+x6+x7+x8)
        x10 = np.tanh(x4+x5+x6+x7+x8)
        
        data[i, 0] = x1
        data[i, 1] = x2
        data[i, 2] = x3
        data[i, 3] = x4
        data[i, 4] = x5
        data[i, 5] = x6
        data[i, 6] = x7
        data[i, 7] = x8
        data[i, 8] = x9
        data[i, 9] = x10
    return data





def real_data_loading(data_name, seq_len):
    """Load and preprocess real-world datasets.
  
  Args:
    - data_name: stock or energy
    - seq_len: sequence length
    
  Returns:
    - data: preprocessed data.
  """
    assert data_name in ['stock', 'energy', 'soft', 'CO_2', 'furnace', 'traffic', 'simulation']

    if data_name == 'stock':
        ori_data = np.loadtxt('data/stock_data.csv', delimiter=",", skiprows=1)
    elif data_name == 'energy':
        ori_data = np.loadtxt('data/energy_data.csv',
                              delimiter=",",
                              skiprows=1)
    elif data_name == 'soft':
        ori_data = np.load('data/soft_sensor.npy')
    elif data_name == 'CO_2':
        dic_data = scio.loadmat('CO2_Absorption.mat')
        ori_data = dic_data[[*dic_data][-1]]
        ori_data = ori_data[:3000]
        
    elif data_name == 'furnace':
        dic_data = scio.loadmat('../dataset/furnace.mat')
        data_x = dic_data['X1']
        data_y = dic_data['Y1']
        ori_data = np.concatenate([data_x, data_y], axis=1) # all 14dim
    elif data_name == 'traffic':
        data_npy = np.load('/data/anonym4/zc/ddcls/dataset/node_values.npy')
        ori_data = data_npy[1000:12000, 99, :]
    elif data_name == 'simulation':
        ori_data = simulation_data(no=8000)
    

    
    ori_data = StandardScaler().fit_transform(ori_data)
    ori_data = MinMaxScaler(feature_range=(-1, 1)).fit_transform(ori_data)
    np.save('../dataset/simulation/raw_data.npy', ori_data)
    print(np.min(ori_data), np.max(ori_data))
    print(ori_data.shape, ori_data)

    # Flip the data to make chronological data
    # ori_data = ori_data[::-1]
    # Normalize the data
    # print(ori_data.shape)
    # ori_data = MinMaxScaler(ori_data)
    # if train_:
    #     ori_data = ori_data[:1400]
    # else:
    #     ori_data = ori_data[1400:1480]

    # Preprocess the dataset
    temp_data = []
    # Cut data by sequence length
    for i in range(0, len(ori_data) - seq_len):
        _x = ori_data[i:i + seq_len]
        temp_data.append(_x)
    soft_test_data = temp_data[6000:8000]
    test_path = '../dataset/simulation/soft_test.npy'
    np.save(test_path, np.array(soft_test_data))

    soft_train_data = temp_data[:6000]

    # Mix the datasets (to make it similar to i.i.d)
    idx = np.random.permutation(len(soft_train_data))
    data = []
    for i in range(len(soft_train_data)):
        data.append(soft_train_data[idx[i]])
    data = np.array(data)
    
    gen_model_train = data[:4000]
    np.save('../dataset/simulation/gen_model_train.npy', gen_model_train)
    gen_model_test = data[4000:]
    miss_ratios = [[0.2, 1], [0.4, 1], [0.6, 1], [0.8, 1]]
    for miss_ratio in miss_ratios:
        masks, miss_data = masked('RBM', miss_ratio, gen_model_test)
        test_data = np.concatenate([gen_model_test, masks], axis=0)
        test_data = np.concatenate([test_data, miss_data], axis=0) # [gt, masks, miss_data]
        np.save('../dataset/simulation/miss_test_{}_{}.npy'.format(miss_ratio[0], miss_ratio[1]), test_data)


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


def main():
    seq_lens = [64]
    for seq_len in seq_lens:
        real_data_loading('simulation', seq_len=seq_len)

# data = simulation_data(4000)
# print(data, data.shape)

# main()
# dat = np.load('/data/anonym4/zc/ddcls/dataset/node_values.npy')
# print(dat[1000:12000,100,:])
        
        
        