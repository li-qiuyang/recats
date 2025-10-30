
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score, \
    precision_recall_fscore_support
def dict2array(results):
    # 获取runs和tasks数量
    runs = len(results)

    # 获取所有非空任务的任务索引
    valid_task_indices = []
    task_names = []  # 用来保存有效任务名称

    for run, dict_run in results.items():
        for key, val in dict_run.items():
            if val:  # 如果任务对应的值不是空字典
                valid_task_indices.append(key)
                task_names.append(key)  # 记录有效任务的名称

    # 去重并排序（保证任务顺序一致）
    valid_task_indices = sorted(set(valid_task_indices))
    task_names = sorted(set(task_names))  # 对任务名称去重并排序

    # 根据有效任务的数量重新定义array的形状
    valid_task_count = len(valid_task_indices)
    array = np.zeros((runs, valid_task_count, valid_task_count))  # 初始化新的三维数组

    # 填充array
    for run, dict_run in results.items():
        # 对每个run，找到有效任务的值并填充到数组中
        valid_index_map = {valid_task_indices[i]: i for i in range(valid_task_count)}

        for e, (key, val) in enumerate(dict_run.items()):
            if not val:  # 跳过空字典
                continue

            for e1, (k, v) in enumerate(val.items()):
                if k in valid_task_indices and key in valid_task_indices:
                    i = valid_index_map[key]  # 获取对应的行索引
                    j = valid_index_map[k]  # 获取对应的列索引
                    array[int(run), i, j] = round(v, 4)  # 填充值

    return array, task_names


def prediction_adjust(pred, gt):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return pred
# def prediction_adjust(prediction, labels):
#     labels = labels[:len(prediction)]
#     i = 0
#     length = len(labels)
#     while i < length:
#         if labels[i] == True:
#             j = i
#
#             adjust_flag = False
#             while labels[j] == True and j < length:
#                 if prediction[j] == True:
#                     adjust_flag = True
#                 j += 1
#                 if j == length:
#                     break
#             if adjust_flag:
#                 for k in range(i, j):
#                     prediction[k] = True
#             i = j
#         else:
#             i += 1
#     return prediction

def calculate_metrics( y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred,
                                                                          average='binary')
    roc_auc = roc_auc_score(y_true, y_pred)

    return accuracy, precision, recall, f1, roc_auc
def calculate_lifelong_roc_auc(R_matrix):
    N = R_matrix.shape[0]  # 获取任务的数量
    # sum_roc_auc = 0
    # for i in range(N):
    #     for j in range(0, i + 1):
    #         sum_roc_auc += R_matrix[i, j]
    # lifelong_roc_auc = sum_roc_auc / (N * (N + 1) / 2)
    # （最终平均AUC）
    lifelong_roc_auc = [R_matrix[N - 1][j] for j in range(N)]
    return np.mean(lifelong_roc_auc)

def calculate_FM(R_matrix):
    # 负值表示遗忘
    N = R_matrix.shape[0]
    fm_sum = 0.0
    for j in range(N - 1):
        max_diff = -np.inf
        for i in range(N - 1):
            current_diff = R_matrix[i][j] - R_matrix[N - 1][j]
            if current_diff > max_diff:
                max_diff = current_diff
        fm_sum += max_diff

    return fm_sum / (N - 1)

def calculate_bwt(R_matrix):
    # 正值表示积极的反向迁移
    N = R_matrix.shape[0]
    # sum_bwt = 0
    # for i in range(1, N):
    #     for j in range(i-1):
    #         sum_bwt += (R_matrix[i, j] - R_matrix[j, j])
    # bwt = sum_bwt / (N * (N - 1) / 2)
    # return bwt
    backward_sum = 0.0
    for j in range(N - 1):  # 遍历旧任务（0到T-2）
        final_auc = R_matrix[N - 1][j]
        initial_auc = R_matrix[j][j]  # 初始训练时的AUC
        backward_sum += (final_auc - initial_auc)
    return backward_sum / (N - 1)

def calculate_fwt(R_matrix):
    N = R_matrix.shape[0]
    sum_fwt = 0
    for i in range(N):
        for j in range(i + 1, N):
            sum_fwt += R_matrix[i, j]
    fwt = sum_fwt / (N * (N - 1) / 2)
    return fwt
    # forward_sum = 0.0
    # for j in range(1, N):  # 遍历新任务（1到N-1）
    #     # 训练完前j-1个任务后，直接测试任务j的AUC
    #     auc_after_pretrain = R_matrix[j-1][j]
    #     baseline = baseline_auc[j]
    #     forward_sum += (auc_after_pretrain - baseline)
    # return forward_sum / (N - 1)

def remove_zero_rows_and_cols(R_matrix):
    """
    删除 R 矩阵中全0的行以及对应的列
    :param R_matrix: 原始的 n_tasks x n_tasks 矩阵
    :return: 删除全0行和列后的矩阵
    """
    # 找出全0的行
    non_zero_rows = np.any(R_matrix != 0, axis=1)
    # 找出全0的列
    non_zero_cols = np.any(R_matrix != 0, axis=0)
    R_matrix_cleaned = R_matrix[non_zero_rows][:, non_zero_cols]

    return R_matrix_cleaned

if __name__ == '__main__':
    rpath = 'results/'
    name = 'SWaT'
    # name = 'PSM'
    print('Experiment:', name)
    print('\n')

    # baseline_acc = np.load(f"{rpath}{name}/baseline_acc.npy", allow_pickle=True).item()
    # baseline_acc = [v for k, v in sorted(baseline_acc.items())]
    # baseline_pre = np.load(f"{rpath}{name}/baseline_pre.npy", allow_pickle=True).item()
    # baseline_pre = [v for k, v in sorted(baseline_pre.items())]
    # baseline_rec = np.load(f"{rpath}{name}/baseline_rec.npy", allow_pickle=True).item()
    # baseline_rec = [v for k, v in sorted(baseline_rec.items())]
    # baseline_f1 = np.load(f"{rpath}{name}/baseline_f1.npy", allow_pickle=True).item()
    # baseline_f1 = [v for k, v in sorted(baseline_f1.items())]
    # baseline_auc = np.load(f"{rpath}{name}/baseline_auc.npy", allow_pickle=True).item()
    # baseline_auc = [v for k, v in sorted(baseline_auc.items())]
    # print(baseline_acc)
    # print(baseline_pre)
    # print(baseline_rec)
    # print(baseline_f1)
    # print(baseline_auc)

    # based on accuracy
    print('-'*50)
    print('based on accuracy score')
    acc_dict = np.load(f"{rpath}{name}/accuracy.npy", allow_pickle=True).item()
    R_matrix, _ = dict2array(acc_dict)
    lifelong_acc = np.round(calculate_lifelong_roc_auc(R_matrix[0]), 4)
    bwt = np.round(calculate_bwt(R_matrix[0]), 4)
    fwt = np.round(calculate_fwt(R_matrix[0]), 4)
    fm = np.round(calculate_FM(R_matrix[0]), 4)
    print(f"Lifelong Accuracy: {lifelong_acc}")
    print(f"BWT: {bwt}")
    print(f"FWT: {fwt}")
    print(f"FM: {fm}")


    # based on precision
    print('-'*50)
    print('based on precision score')
    pre_dict = np.load(f"{rpath}{name}/precision.npy", allow_pickle=True).item()
    R_matrix, _ = dict2array(pre_dict)
    lifelong_pre = np.round(calculate_lifelong_roc_auc(R_matrix[0]), 4)
    bwt = np.round(calculate_bwt(R_matrix[0]), 4)
    fwt = np.round(calculate_fwt(R_matrix[0]), 4)
    fm = np.round(calculate_FM(R_matrix[0]), 4)
    print(f"Lifelong Precision: {lifelong_pre}")
    print(f"BWT: {bwt}")
    print(f"FWT: {fwt}")
    print(f"FM: {fm}")

    # based on recall
    print('-'*50)
    print('based on recall score')
    rec_dict = np.load(f"{rpath}{name}/recall.npy", allow_pickle=True).item()
    R_matrix, _ = dict2array(rec_dict)
    lifelong_rec = np.round(calculate_lifelong_roc_auc(R_matrix[0]), 4)
    bwt = np.round(calculate_bwt(R_matrix[0]), 4)
    # print(baseline_rec)
    # print(R_matrix[0])
    fwt = np.round(calculate_fwt(R_matrix[0]), 4)
    fm = np.round(calculate_FM(R_matrix[0]), 4)
    print(f"Lifelong Recall: {lifelong_rec}")
    print(f"BWT: {bwt}")
    print(f"FWT: {fwt}")
    print(f"FM: {fm}")


    # based on f1-score
    print('-'*50)
    print('based on f1 score')
    f1_dict = np.load(f"{rpath}{name}/f1_score.npy", allow_pickle=True).item()
    R_matrix, _ = dict2array(f1_dict)
    # print(R_matrix[0])
    lifelong_f1 = np.round(calculate_lifelong_roc_auc(R_matrix[0]), 4)
    bwt = np.round(calculate_bwt(R_matrix[0]), 4)
    fwt = np.round(calculate_fwt(R_matrix[0]), 4)
    fm = np.round(calculate_FM(R_matrix[0]), 4)
    print(f"Lifelong F1-score: {lifelong_f1}")
    print(f"BWT: {bwt}")
    print(f"FWT: {fwt}")
    print(f"FM: {fm}")

    # based on ROC_AUC
    print('-'*50)
    print('based on roc_auc score')
    auc_dict = np.load(f"{rpath}{name}/roc_auc.npy", allow_pickle=True).item()
    R_matrix, _ = dict2array(auc_dict)
    lifelong_roc_auc = np.round(calculate_lifelong_roc_auc(R_matrix[0]), 4)
    bwt = np.round(calculate_bwt(R_matrix[0]), 4)
    fwt = np.round(calculate_fwt(R_matrix[0]), 4)
    fm = np.round(calculate_FM(R_matrix[0]), 4)

    print(f"Lifelong ROC-AUC: {lifelong_roc_auc}")
    print(f"BWT: {bwt}")
    print(f"FWT: {fwt}")
    print(f"FM: {fm}")
