import os
from random import sample, random
from typing import get_args

from data.utils import load_npy_dataset
from prepare_scenario import prepare_and_save_scenario
from scenario_config import ScenarioType, ScenarioConfig
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset, TensorDataset, ConcatDataset



def add_noise(data, features, noise_level=0.9, random_state=42):
    np.random.seed(random_state)
    noisy_data = data.copy()
    for feature in features:
        feature_std = data[feature].std()
        noise = np.random.normal(loc=0, scale=noise_level * feature_std, size=data[feature].shape)
        noisy_data[feature] += noise
    return noisy_data

def oversample_anomalies(data, x_features, ratio=0.1, random_state=42):
    X = data[x_features]
    # print(X.shape)
    # print(X.head())
    y = data['Normal/Attack']
    # print(y.value_counts())
    # print(y.shape)
    # print(y.head())

    smote = SMOTE(sampling_strategy=ratio, random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    # print(y_resampled.shape)
    # print(y_resampled.value_counts())

    resampled_data = pd.DataFrame(X_resampled, columns=x_features)
    resampled_data['Normal/Attack'] = y_resampled
    return resampled_data


def UNSW(random_state=42):
    DATASET_NAME = 'unsw'

    CONFIGS_TO_GENERATE = [
        ScenarioConfig(scenario_type=scenario_type, concepts_no=clusters_no, size_per_concept=size_per_cluster)
        for scenario_type in get_args(ScenarioType)
        for clusters_no in [10]
        for size_per_cluster in [5000]
    ]


    for config in CONFIGS_TO_GENERATE:
        print(
            f"Generating scenario type {config.scenario_type} with {config.concepts_no} clusters and {config.size_per_concept} samples per normal cluster")
        normal_data, anomaly_data = load_npy_dataset('data/UNSW-NB15/full_unsw.npy')
        train_dataset_splits, test_dataset_splits = prepare_and_save_scenario(DATASET_NAME, normal_data, anomaly_data, config)
        return train_dataset_splits, test_dataset_splits

# def SWAT(random_state=42):
#     # 读取SWaT数据集
#     test_file = './data/SWaT/SWaT_Dataset_Attack_v0.csv'
#     train_file = './data/SWaT/SWaT_Dataset_Normal_v1.csv'
#     data = pd.read_csv(test_file, sep=';')
#     data = data.replace(',', '.', regex=True)
#     data.columns = data.columns.str.strip()
#
#     # 转换时间戳格式，提取小时信息
#     data['Timestamp'] = data['Timestamp'].str.strip()
#     data['Timestamp'] = pd.to_datetime(data['Timestamp'], format='%d/%m/%Y %I:%M:%S %p')
#
#     data['Hour'] = data['Timestamp'].dt.hour
#
#     # 将异常样本标注为1，正常样本标注为0
#     data['Normal/Attack'] = data['Normal/Attack'].apply(lambda x: 1 if x == 'Attack' else 0)
#
#     # 分割异常和正常样本
#     anomaly_data = data[data['Normal/Attack'] == 1]
#     normal_data = data[data['Normal/Attack'] == 0]
#
#     # 随机分割异常样本为两部分（训练集和测试集）
#     split_index_anomalies = int(len(anomaly_data) * 0.6)
#     anomaly_data_train = anomaly_data.sample(n=split_index_anomalies, random_state=random_state)
#     anomaly_data_test = anomaly_data.drop(anomaly_data_train.index)
#
#     # 随机分割正常样本为两部分（训练集和测试集）
#     split_index_normals = int(len(normal_data) * 0.7)
#     normal_data_train = normal_data.sample(n=split_index_normals, random_state=random_state)
#     normal_data_test = normal_data.drop(normal_data_train.index)
#
#     # 将部分异常样本加入训练集和测试集
#     train_data_with_anomalies = pd.concat([normal_data_train, anomaly_data_train])
#     test_data_with_anomalies = pd.concat([normal_data_test, anomaly_data_test])
#
#     # 特征选择
#     x_feature = ['FIT101', 'LIT101', 'MV101', 'P101', 'P102', 'AIT201', 'AIT202', 'AIT203',
#                  'FIT201', 'MV201', 'P201', 'P202', 'P203', 'P204', 'P205', 'P206',
#                  'DPIT301', 'FIT301', 'LIT301', 'MV301', 'MV302', 'MV303', 'MV304',
#                  'P301', 'P302', 'AIT401', 'AIT402', 'FIT401', 'LIT401', 'P401', 'P402',
#                  'P403', 'P404', 'UV401', 'AIT501', 'AIT502', 'AIT503', 'AIT504',
#                  'FIT501', 'FIT502', 'FIT503', 'FIT504', 'P501', 'P502', 'PIT501',
#                  'PIT502', 'PIT503', 'FIT601', 'P601', 'P602', 'P603']
#     # 转换特征列为数值类型
#     train_data_with_anomalies[x_feature] = train_data_with_anomalies[x_feature].apply(pd.to_numeric, errors='coerce')
#     test_data_with_anomalies[x_feature] = test_data_with_anomalies[x_feature].apply(pd.to_numeric, errors='coerce')
#
#     # print(train_data_with_anomalies.head())
#     # 给训练集中的异常样本添加噪声
#     anomalous_data_with_noise = train_data_with_anomalies[train_data_with_anomalies['Normal/Attack'] == 1]
#     anomalous_data_with_noise = add_noise(anomalous_data_with_noise, x_feature)
#     train_data_with_anomalies.update(anomalous_data_with_noise)
#
#     # 打印数据集信息
#     print(f"训练集大小: {train_data_with_anomalies.shape}")
#     print(f"测试集大小: {test_data_with_anomalies.shape}")
#     print(train_data_with_anomalies.head())
#
#     # 创建按小时划分的任务
#     hourly_train_data = {}
#     hourly_test_data = {}
#
#     # 按小时分割训练集和测试集
#     for hour in range(24):
#         hourly_train_data[hour] = train_data_with_anomalies[train_data_with_anomalies['Hour'] == hour]
#         hourly_test_data[hour] = test_data_with_anomalies[test_data_with_anomalies['Hour'] == hour]
#
#     # 将数据转换为TensorDataset并返回
#     train_dataset_splits = {}
#     test_dataset_splits = {}
#
#     # 初始化归一化器
#     scaler = MinMaxScaler()
#     scaler.fit(train_data_with_anomalies[x_feature])
#     for hour in range(24):
#         train_features = hourly_train_data[hour][x_feature]
#         train_labels = hourly_train_data[hour]['Normal/Attack']
#         # print(train_labels.size)
#         test_features = hourly_test_data[hour][x_feature]
#         test_labels = hourly_test_data[hour]['Normal/Attack'].values
#
#         num = train_labels.shape[0]
#         attack_num = sum(train_labels[train_labels == 1])
#
#         if hour >= 10 and train_labels.value_counts()[0] != num and attack_num / (num - attack_num) < 0.1:
#             train_task_data = pd.concat([train_features, train_labels], axis=1)
#             train_task_data.columns = x_feature + ['Normal/Attack']
#             # 进行过采样
#             train_data_with_anomalies_resampled = oversample_anomalies(train_task_data, x_feature, ratio=0.3).sample(
#                 frac=1, random_state=42).reset_index(drop=True)
#             # 分离特征和标签
#             train_features = train_data_with_anomalies_resampled[x_feature]
#             train_labels = train_data_with_anomalies_resampled['Normal/Attack']
#
#         # 归一化特征
#         train_features_scaled = scaler.fit_transform(train_features)
#         test_features_scaled = scaler.fit_transform(test_features)
#
#         train_tensor = torch.tensor(train_features_scaled, dtype=torch.float32)
#         train_labels_tensor = torch.tensor(train_labels.values, dtype=torch.long)
#         test_tensor = torch.tensor(test_features_scaled, dtype=torch.float32)
#         test_labels_tensor = torch.tensor(test_labels, dtype=torch.long)
#
#         # 创建TensorDataset
#         train_dataset_splits[hour] = TensorDataset(train_tensor, train_labels_tensor)
#         test_dataset_splits[hour] = TensorDataset(test_tensor, test_labels_tensor)
#
#     return train_dataset_splits, test_dataset_splits

# def process_data():
#     PATH = os.path.dirname(os.path.abspath(__file__))
#     data_path = PATH + '../../data/SWaT'
#     # #Read data
#     normal = pd.read_csv(data_path + "/SWaT_Dataset_Normal_v1.csv",header=1)#, nrows=1000)
#     normal = normal.drop([" Timestamp" , "Normal/Attack" ] , axis = 1)
#
#     for i in list(normal):
#         normal[i]=normal[i].apply(lambda x: str(x).replace("," , "."))
#     normal = normal.astype(float)
#
#     down_rate = 5
#     normal=normal.groupby(np.arange(len(normal.index)) // down_rate).mean()
#     min_max_scaler = preprocessing.MinMaxScaler()
#     x = normal.values
#     x_scaled = min_max_scaler.fit_transform(x)
#     normal = pd.DataFrame(x_scaled)
#     X_times = np.array(normal)
#     np.save(data_path+'/SWAT_train.npy',X_times)
#
#     attack = pd.read_csv(data_path + "/SWaT_Dataset_Attack_v0.csv",header=0)#, nrows=1000)
#     labels = [ float(label!= 'Normal' ) for label  in attack["Normal/Attack"].values]
#     attack = attack.drop([" Timestamp" , "Normal/Attack" ] , axis = 1)
#     for i in list(attack):
#         attack[i]=attack[i].apply(lambda x: str(x).replace("," , "."))
#
#     attack = attack.astype(float)
#     attack=attack.groupby(np.arange(len(attack.index)) // down_rate).mean()
#     #Downsampling the labels
#     labels_down=[]
#
#     for i in range(len(labels)//down_rate):
#         if labels[5*i:5*(i+1)].count(1.0):
#             labels_down.append(1.0) #Attack
#         else:
#             labels_down.append(0.0) #Normal
#
#     #for the last few labels that are not within a full-length window
#     if labels[down_rate*(i+1):].count(1.0):
#         labels_down.append(1.0) #Attack
#     else:
#         labels_down.append(0.0) #Normal
#     from sklearn import preprocessing
#
#     x = attack.values
#
#     x_scaled = min_max_scaler.transform(x)
#     attack = pd.DataFrame(x_scaled)
#     X_times_test = np.array(attack)
#
#     test_labels = np.array(labels_down)
#     np.save(data_path+'/SWAT_test.npy',X_times_test)
#     np.save(data_path+'/SWAT_test_labels.npy',test_labels)

def get_mask(observed_mask, mask_ratio):
    mask = torch.zeros_like(observed_mask)

    original_mask_shape = mask.shape

    mask = mask.reshape(-1)
    total_index_list = list(range(len(mask)))

    selected_number = int(len(total_index_list) * mask_ratio)

    selected_index = sample(total_index_list, selected_number)

    selected_index = torch.LongTensor(selected_index)

    mask[selected_index] = 1

    mask = mask.reshape(original_mask_shape)

    return mask

class TrainData(Dataset):

    def __init__(self, file_path, test_path, window_length=100,split=4,mask_ratio=0.5):
        self.data = pickle.load(
            open(file_path, "rb")
        )
        length = self.data.shape[0]

        self.mask_ratio = mask_ratio
        self.test_data = pickle.load(
            open(test_path, "rb")
        )
        self.data = np.concatenate([self.data, self.test_data])
        self.data = torch.Tensor(self.data)
        # 为了避免高斯噪声造成的影响过大，此处将原有的数值全部乘以20
        self.data = self.data[:length, :] * 20
        self.window_length = window_length
        self.begin_indexes = list(range(0, len(self.data) - 100))
        self.split = split


    def get_mask(self, observed_mask, strategy_type):
        mask = torch.zeros_like(observed_mask)
        length = observed_mask.shape[0]
        if strategy_type == 0:
            # mask_ratio = self.mask_ratio

            skip = length // self.split
            for split_index, begin_index in enumerate(list(
                    range(0, length, skip)
            )):
                if split_index % 2 == 0:
                    mask[begin_index: min(begin_index + skip, length), :] = 1
        else:
            # mask_ratio = 1 - self.mask_ratio
            skip = length // self.split
            for split_index, begin_index in enumerate(list(
                    range(0, length, skip)
            )):
                if split_index % 2 != 0:
                    mask[begin_index: min(begin_index + skip, length), :] = 1

        return mask


    def __len__(self):
        return len(self.begin_indexes)

    def __getitem__(self, item):
        if random.random() < 0.5:
            strategy_type = 0
        else:
            strategy_type = 1

        observed_data = self.data[
            self.begin_indexes[item] :
               self.begin_indexes[item] + self.window_length
        ]
        observed_mask = torch.ones_like(observed_data)
        gt_mask = self.get_mask(observed_mask, strategy_type)
        timepoints = np.arange(self.window_length)
        return {
            "observed_data": observed_data,
            "observed_mask": observed_mask,
            "gt_mask": gt_mask,
            "timepoints": timepoints,
            "strategy_type": strategy_type
        }

def SWAT(random_state=42,mask_ratio=0.5):
    # 读取SWaT数据集
    test_file = './data/SWaT/SWaT_Dataset_Attack_v0.csv'
    train_file = './data/SWaT/SWaT_Dataset_Normal_v1.csv'
    test_data = pd.read_csv(test_file, sep=';')
    test_data = test_data.replace(',', '.', regex=True)
    test_data.columns = test_data.columns.str.strip()

    train_data = pd.read_csv(train_file, sep=',')
    train_data = train_data.replace(',', '.', regex=True)
    train_data.columns = train_data.columns.str.strip()

    # 转换时间戳格式，提取小时信息
    test_data['Timestamp'] = test_data['Timestamp'].str.strip()
    test_data['Timestamp'] = pd.to_datetime(test_data['Timestamp'], format='%d/%m/%Y %I:%M:%S %p')
    train_data['Timestamp'] = train_data['Timestamp'].str.strip()
    train_data['Timestamp'] = pd.to_datetime(train_data['Timestamp'], format='%d/%m/%Y %I:%M:%S %p')

    test_data['Hour'] = test_data['Timestamp'].dt.hour
    train_data['Hour'] = train_data['Timestamp'].dt.hour

    # 将异常样本标注为1，正常样本标注为0
    test_data['Normal/Attack'] = test_data['Normal/Attack'].apply(lambda x: 1 if x == 'Attack' else 0)
    train_data['Normal/Attack'] = train_data['Normal/Attack'].apply(lambda x: 1 if x == 'Attack' else 0)



    # 特征选择
    x_feature = ['FIT101', 'LIT101', 'MV101', 'P101', 'P102', 'AIT201', 'AIT202', 'AIT203',
                 'FIT201', 'MV201', 'P201', 'P202', 'P203', 'P204', 'P205', 'P206',
                 'DPIT301', 'FIT301', 'LIT301', 'MV301', 'MV302', 'MV303', 'MV304',
                 'P301', 'P302', 'AIT401', 'AIT402', 'FIT401', 'LIT401', 'P401', 'P402',
                 'P403', 'P404', 'UV401', 'AIT501', 'AIT502', 'AIT503', 'AIT504',
                 'FIT501', 'FIT502', 'FIT503', 'FIT504', 'P501', 'P502', 'PIT501',
                 'PIT502', 'PIT503', 'FIT601', 'P601', 'P602', 'P603']
    # 转换特征列为数值类型
    train_data[x_feature] = train_data[x_feature].apply(pd.to_numeric, errors='coerce')
    test_data[x_feature] = test_data[x_feature].apply(pd.to_numeric, errors='coerce')


    print(test_data.head())
    print(train_data.head())

    # 给训练集中的异常样本添加噪声
    # anomalous_data_with_noise = train_data_with_anomalies[train_data_with_anomalies['Normal/Attack'] == 1]
    # anomalous_data_with_noise = add_noise(anomalous_data_with_noise, x_feature)
    # train_data_with_anomalies.update(anomalous_data_with_noise)

    # 打印数据集信息
    print(f"训练集大小: {train_data.shape}")
    print(f"测试集大小: {test_data.shape}")

    # 创建按小时划分的任务
    hourly_train_data = {}
    hourly_test_data = {}

    # 按小时分割训练集和测试集
    for hour in range(24):
        hourly_train_data[hour] = train_data[train_data['Hour'] == hour]
        hourly_test_data[hour] = test_data[test_data['Hour'] == hour]

    # 将数据转换为TensorDataset并返回
    train_dataset_splits = {}
    mask_train_dataset_splits = {}
    test_dataset_splits = {}

    # 初始化归一化器
    scaler = MinMaxScaler()
    scaler.fit(train_data[x_feature])
    for hour in range(24):
        train_features = hourly_train_data[hour][x_feature]
        train_labels = hourly_train_data[hour]['Normal/Attack']
        # print(train_labels.size)
        test_features = hourly_test_data[hour][x_feature]
        test_labels = hourly_test_data[hour]['Normal/Attack'].values

        # num = train_labels.shape[0]
        # attack_num = sum(train_labels[train_labels == 1])
        #
        # if hour >= 10 and train_labels.value_counts()[0] != num and attack_num / (num - attack_num) < 0.1:
        #     train_task_data = pd.concat([train_features, train_labels], axis=1)
        #     train_task_data.columns = x_feature + ['Normal/Attack']
        #     # 进行过采样
        #     train_data_with_anomalies_resampled = oversample_anomalies(train_task_data, x_feature, ratio=0.3).sample(
        #         frac=1, random_state=42).reset_index(drop=True)
        #     # 分离特征和标签
        #     train_features = train_data_with_anomalies_resampled[x_feature]
        #     train_labels = train_data_with_anomalies_resampled['Normal/Attack']

        # 归一化特征
        train_features_scaled = scaler.fit_transform(train_features)
        test_features_scaled = scaler.fit_transform(test_features)
        # mask = [random.random() for _ in range(len(train_labels))]
        # mask = mask[mask > 0.3]
        # # 生成随机掩码
        l, w = train_features.shape[0], train_features.shape[1]
        mask = np.random.rand(l, w)<mask_ratio
        # print(mask[:5])
        # print(type(train_features))
        mask_train_features = train_features_scaled * mask - (1 - mask) # 掩码后的输入
        # print(train_features[:5])
        # print(mask_train_features[:5])

        train_tensor = torch.tensor(train_features_scaled, dtype=torch.float32)
        mask_train_tensor = torch.tensor(mask_train_features, dtype=torch.float32)
        train_labels_tensor = torch.tensor(train_labels.values, dtype=torch.long)
        test_tensor = torch.tensor(test_features_scaled, dtype=torch.float32)
        test_labels_tensor = torch.tensor(test_labels, dtype=torch.long)

        # 创建TensorDataset
        train_dataset_splits[hour] = TensorDataset(train_tensor, train_labels_tensor)
        mask_train_dataset_splits[hour] = TensorDataset(mask_train_tensor, train_labels_tensor)
        test_dataset_splits[hour] = TensorDataset(test_tensor, test_labels_tensor)

    return train_dataset_splits, mask_train_dataset_splits, test_dataset_splits, mask

if __name__ == '__main__':
    train_dataset_splits, mask_train_dataset_splits, test_dataset_splits, mask = SWAT(mask_ratio=0.5)
    print(mask)
    print(train_dataset_splits[0][0][1])

