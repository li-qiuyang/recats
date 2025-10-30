import pandas as pd
from sklearn.preprocessing import StandardScaler

labels = pd.read_csv('../../data/weather/NEweather_class.csv',header=None)
data = pd.read_csv('../../data/weather/NEweather_data.csv',header=None)
# print(data.shape)  (18159,8)
# print(labels.shape)  (18159,1)
# count_0 = (labels == 0).sum()  # 正常样本的数量  12461
# count_1 = (labels == 1).sum()  # 异常样本的数量  5698
df = pd.concat([data, labels], axis=1)

df.columns = [f'feature_{i}' for i in range(data.shape[1])] + ['label']

scaler = StandardScaler()
features = scaler.fit_transform(df.iloc[:, :-1])
df_processed = pd.DataFrame(features, columns=df.columns[:-1])
df_processed['label'] = df['label'].values


# 按连续块划分任务
TASK_SIZE = 2000  # 每个任务样本量
num_samples = len(df_processed)
print(num_samples)
num_tasks = num_samples // TASK_SIZE

for task_id in range(num_tasks):
    start = task_id * TASK_SIZE
    end = (task_id + 1) * TASK_SIZE
    task_data = df_processed.iloc[start:end]

    if task_id == num_tasks - 1 and end < num_samples:
        task_data = df_processed.iloc[start:]

    split_idx = int(len(task_data) * 0.7)
    train = task_data.iloc[:split_idx]
    test = task_data.iloc[split_idx:]