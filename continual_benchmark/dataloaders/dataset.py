import pickle

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
def preprocess_weather(data_path, label_path, window_size=2000, train_ratio=0.7):
    labels = pd.read_csv(label_path, header=None)
    data = pd.read_csv(data_path, header=None)
    df = pd.concat([data, labels], axis=1)
    # print(df.head())
    df = df.reset_index(drop=False)
    df.columns = ['time'] + [f'feature_{i}' for i in range(1, df.shape[1] - 1)] + ['label']

    window_data = {'train': {}, 'test': {}}
    num_samples = df.shape[0]
    print(num_samples)
    num_tasks = num_samples // window_size

    for task_id in range(num_tasks):
        start = task_id * window_size
        end = (task_id + 1) * window_size
        task_data = df.iloc[start:end]
        if task_id == num_tasks - 1 and end < num_samples:
            task_data = df.iloc[start:]

        normal_data = task_data['label'] == 0
        anomaly_data = task_data['label'] == 1
        train = task_data[normal_data][:int(len(task_data[normal_data]) * train_ratio)]
        normal_test = task_data[normal_data][int(len(task_data[normal_data]) * train_ratio):]

        test = pd.concat([normal_test, task_data[anomaly_data]])

        test = test.sort_values(by=test.columns[0])
        test_labels = test['label']
        test_labels = test_labels.values.reshape(-1, 1).astype(np.float32)

        test_feat = test.drop(['time', 'label'], axis=1)
        train_feat = train.drop(['time', 'label'], axis=1)

        # 归一化处理
        # scaler = MinMaxScaler(feature_range=(0, 1), clip=True).fit(train_feat)
        # scaled_train = scaler.transform(train_feat)
        # scaled_test = scaler.transform(test_feat)

        scaler = StandardScaler().fit(train_feat)
        scaled_train = scaler.transform(train_feat)
        scaled_test = scaler.transform(test_feat)

        window_data['train'][task_id] = {
            'features': scaled_train,
            'labels': np.zeros(len(scaled_train))  # 训练集全为正常
        }
        window_data['test'][task_id] = {
            'features': scaled_test,
            'labels': test_labels
        }
    return window_data

def preprocess_swat(path,window_size=20000,train_ratio=0.7):
    """SWaT数据预处理函数，返回按小时组织的完整数据集"""
    # 读取数据
    data = pd.read_csv(path, sep=';')

    # 统一格式处理
    data.columns = data.columns.str.strip()
    data.replace(',', '.', regex=True, inplace=True)

    # 转换时间戳格式，提取小时信息
    data['Timestamp'] = data['Timestamp'].str.strip()
    data['Timestamp'] = pd.to_datetime(data['Timestamp'], format='%d/%m/%Y %I:%M:%S %p')
    data['Hour'] = data['Timestamp'].dt.hour
    # print(data.shape)
    data['Normal/Attack'] = data['Normal/Attack'].apply(lambda x: 1 if x == 'Attack' else 0)

    features = [
        'FIT101', 'LIT101', 'MV101', 'P101', 'P102', 'AIT201', 'AIT202', 'AIT203',
        'FIT201', 'MV201', 'P201', 'P202', 'P203', 'P204', 'P205', 'P206',
        'DPIT301', 'FIT301', 'LIT301', 'MV301', 'MV302', 'MV303', 'MV304',
        'P301', 'P302', 'AIT401', 'AIT402', 'FIT401', 'LIT401', 'P401', 'P402',
        'P403', 'P404', 'UV401', 'AIT501', 'AIT502', 'AIT503', 'AIT504',
        'FIT501', 'FIT502', 'FIT503', 'FIT504', 'P501', 'P502', 'PIT501',
        'PIT502', 'PIT503', 'FIT601', 'P601', 'P602', 'P603'
    ]
    data[features] = data[features].apply(pd.to_numeric, errors='coerce')

    window_data = {'train': {}, 'test': {}}
    for hour in range(24):
        # 提取当前小时的数据并按时间排序
        hour_data = data[data['Hour'] == hour].sort_values(by='Timestamp')
        normal_data = hour_data[hour_data['Normal/Attack'] == 0]
        anomaly_data = hour_data[hour_data['Normal/Attack'] == 1]

        # 训练集只包含正常样本（按时间顺序取前70%）
        split_index = int(len(normal_data) * train_ratio)
        train_data = normal_data.iloc[:split_index].sort_values(by='Timestamp')
        test_normal = normal_data.iloc[split_index:]


        # 测试集包含剩余正常样本+全部异常样本
        test_data = pd.concat([test_normal, anomaly_data]).sort_values(by='Timestamp')

        train_median = train_data[features].median()
        train_data.loc[:, features] = train_data[features].fillna(train_median)
        test_data[features] = test_data[features].fillna(train_median)

        # scaler = MinMaxScaler(feature_range=(0, 1), clip=True).fit(train_data[features])
        # train_features = scaler.transform(train_data[features])
        # test_features = scaler.transform(test_data[features])
        scaler = StandardScaler().fit(train_data[features])
        train_features = scaler.transform(train_data[features])
        test_features = scaler.transform(test_data[features])

        # 存储结果
        window_data['train'][hour] = {
            'features': train_features,
            'labels': train_data['Normal/Attack'].values
        }
        window_data['test'][hour] = {
            'features': test_features,
            'labels': test_data['Normal/Attack'].values
        }

    return window_data

def preprocess_psm(data_path, label_path, window_size=2000, train_ratio=0.7):

    data = pd.read_csv(data_path)
    labels = pd.read_csv(label_path)

    # 合并数据并按时间排序
    merged = pd.merge(data, labels, on='timestamp_(min)', how='inner').sort_values('timestamp_(min)')
    # # features = merged.drop([ 'label'], axis=1).values
    # features = merged.drop(['label'], axis=1)
    # labels = merged['label']

    # 按时间窗口划分任务
    window_data = {'train': {}, 'test': {}}
    n_samples = len(merged)

    for window_id, start_idx in enumerate(range(0, n_samples, window_size)):
        end_idx = min(start_idx + window_size, n_samples)
        window_feat = merged.iloc[:, :-1][start_idx:end_idx]

        window_labels = merged.iloc[:, -1][start_idx:end_idx]

        # 提取当前窗口正常样本
        normal_mask = (window_labels == 0)
        normal_feat = window_feat[normal_mask]

        n_normal = len(normal_feat)

        split_idx = int(n_normal * train_ratio)
        train_feat = normal_feat[:split_idx].sort_values(by='timestamp_(min)')
        train_feat = train_feat.drop(['timestamp_(min)'], axis=1)
        test_normal_feat = normal_feat[split_idx:]

        # 构建测试集（剩余正常+全部异常）
        test_feat = np.concatenate((test_normal_feat, window_feat[~normal_mask]), axis=0)
        test_labels = np.concatenate((np.zeros(len(test_normal_feat)), window_labels[~normal_mask]), axis=0)

        test = np.concatenate((test_feat, test_labels.reshape(-1, 1)), axis=1)

        # 获取第一列的排序索引
        sorted_indices = np.argsort(test[:, 0])
        # 根据排序索引重新排列数组的行
        test = test[sorted_indices]

        test_feat = test[:, 1:-1]
        test_labels = test[:, -1]

        # 归一化处理
        # scaler = MinMaxScaler(feature_range=(0, 1), clip=True).fit(train_feat)
        # scaled_train = scaler.transform(train_feat)
        # scaled_test = scaler.transform(test_feat)

        scaler = StandardScaler().fit(train_feat)
        scaled_train = scaler.transform(train_feat)
        scaled_test = scaler.transform(test_feat)
        # 存储任务数据
        window_data['train'][window_id] = {
            'features': scaled_train,
            'labels': np.zeros(len(scaled_train))  # 训练集全为正常
        }
        window_data['test'][window_id] = {
            'features': scaled_test,
            'labels': test_labels
        }

    return window_data

class Dataset(Dataset):
    def __init__(self, features, labels, mask_ratio=0.3):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.mask_ratio = mask_ratio
        self.num_features = features.shape[1]
    def _get_feature_mask(self):
        mask = torch.zeros(self.num_features)
        masked_indices = torch.randperm(self.num_features)[:int(self.mask_ratio * self.num_features)]
        mask[masked_indices] = 1
        return mask

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        point_features = self.features[idx]
        mask = self._get_feature_mask()
        return {
            'original_data': point_features,
            'masked_data': point_features * (1 - mask),
            'gt_mask': mask,
            'labels': self.labels[idx]
        }

def create_datasets(window_data, mode='train', mask_ratio=0.5):
    task_datasets = {}
    for window_id in window_data[mode].keys():
        data = window_data[mode][window_id]
        dataset = Dataset(
            features=data['features'],
            labels=data['labels'],
            mask_ratio=mask_ratio
        )
        task_datasets[window_id] = dataset
    return task_datasets

def preprocess_smap(data_path, label_path, window_size=1000, train_ratio=0.7):
    data = np.load(data_path).tolist()
    labels = np.load(label_path)


    for idx in range(len(data)):
        data[idx].append(idx)

    data = np.array(data)

    # 按时间窗口划分任务
    window_data = {'train': {}, 'test': {}}
    n_samples = len(data)

    for window_id, start_idx in enumerate(range(0, n_samples, window_size)):
        end_idx = min(start_idx + window_size, n_samples)
        window_feat = data[start_idx:end_idx]

        window_labels = labels[start_idx:end_idx]

        # 提取当前窗口正常样本
        normal_mask = (window_labels == 0)
        normal_feat = window_feat[normal_mask]
        normal_label = window_labels[normal_mask]
        # 异常样本
        abnormal_mask = (window_labels == 1)
        abnormal_feat = window_feat[abnormal_mask]
        abnormal_label = window_labels[abnormal_mask]

        n_normal = len(normal_feat)

        split_idx = int(n_normal * train_ratio)
        train_feat = normal_feat[:split_idx]
        # 训练集排序
        sort_indices = np.argsort(train_feat[:, -1])
        train_feat = train_feat[sort_indices]
        # 去掉时间戳列
        train_feat = train_feat[:, :-1]

        # 测试集
        test_normal_feat = normal_feat[split_idx:]
        test_normal_label = normal_label[split_idx:]
        test_abnormal_feat = abnormal_feat
        test_abnormal_label = abnormal_label
        # 构建测试集（剩余正常+全部异常）
        test_feat = np.concatenate((test_normal_feat, test_abnormal_feat), axis=0)

        test_labels = np.concatenate((test_normal_label, test_abnormal_label), axis=0)
        test = np.concatenate((test_feat, test_labels.reshape(-1, 1)), axis=1)

        sort_indices = np.argsort(test[:, -2])
        # 根据排序索引重新排列数组的行
        test = test[sort_indices]

        test_feat = test[:, :-2]
        test_labels = test[:, -1]

        scaler = StandardScaler().fit(train_feat)
        scaled_train = scaler.transform(train_feat)
        scaled_test = scaler.transform(test_feat)


        # 存储任务数据
        window_data['train'][window_id] = {
            'features': scaled_train,
            'labels': np.zeros(len(scaled_train))  # 训练集全为正常
        }
        window_data['test'][window_id] = {
            'features': scaled_test,
            'labels': test_labels
        }

    return window_data

def preprocess_GECCO(data_path, label_path, window_size=1000, train_ratio=0.7):
    data = np.load(data_path).tolist()
    labels = np.load(label_path)
    # print(data.shape, labels.shape)

    for idx in range(len(data)):
        data[idx].append(idx)

    data = np.array(data)
    print(data.shape, labels.shape)
    # 按时间窗口划分任务
    window_data = {'train': {}, 'test': {}}
    n_samples = len(data)

    for window_id, start_idx in enumerate(range(0, n_samples, window_size)):
        end_idx = min(start_idx + window_size, n_samples)
        window_feat = data[start_idx:end_idx]

        window_labels = labels[start_idx:end_idx]

        # 提取当前窗口正常样本
        normal_mask = (window_labels == 0)
        normal_feat = window_feat[normal_mask]
        normal_label = window_labels[normal_mask]
        print(len(normal_feat))
        # 异常样本
        abnormal_mask = (window_labels == 1)
        abnormal_feat = window_feat[abnormal_mask]
        abnormal_label = window_labels[abnormal_mask]

        n_normal = len(normal_feat)

        split_idx = int(n_normal * train_ratio)
        train_feat = normal_feat[:split_idx]
        # 训练集排序
        sort_indices = np.argsort(train_feat[:, -1])
        train_feat = train_feat[sort_indices]
        # 去掉时间戳列
        train_feat = train_feat[:, :-1]

        # 测试集
        test_normal_feat = normal_feat[split_idx:]
        test_normal_label = normal_label[split_idx:]
        test_abnormal_feat = abnormal_feat
        test_abnormal_label = abnormal_label
        # 构建测试集（剩余正常+全部异常）
        test_feat = np.concatenate((test_normal_feat, test_abnormal_feat), axis=0)

        test_labels = np.concatenate((test_normal_label, test_abnormal_label), axis=0)
        test = np.concatenate((test_feat, test_labels.reshape(-1, 1)), axis=1)

        sort_indices = np.argsort(test[:, -2])
        # 根据排序索引重新排列数组的行
        test = test[sort_indices]

        test_feat = test[:, :-2]
        test_labels = test[:, -1]

        scaler = StandardScaler().fit(train_feat)
        scaled_train = scaler.transform(train_feat)
        scaled_test = scaler.transform(test_feat)


        # 存储任务数据
        window_data['train'][window_id] = {
            'features': scaled_train,
            'labels': np.zeros(len(scaled_train))  # 训练集全为正常
        }
        window_data['test'][window_id] = {
            'features': scaled_test,
            'labels': test_labels
        }

    return window_data

