import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import os


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


def calculate_lower_triangular_mean(array):
    new_array = array[0]
    # print(new_array.shape) (20, 20)
    n_tasks = new_array.shape[0]

    # 初始化下三角元素的列表
    lower_triangular_elements = []

    for i in range(n_tasks):
        for j in range(0, i+1):
            # print(i, j)
            lower_triangular_elements.append(new_array[i, j])

    mean_lower_triangular = np.mean(lower_triangular_elements)

    mean_value_rounded = np.round(mean_lower_triangular, 4)

    return mean_value_rounded

def grid_plot(ax, array, exp_name, plot_type, task_names):
    if plot_type == "fid":
        round = 1
    else:
        round = 4
    avg_array = np.around(np.mean(array, axis=0), round)
    num_tasks = array.shape[1]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#287233", "#4c1c24"])
    ax.imshow(avg_array, vmin=50, vmax=300, cmap=cmap)

    for i in range(len(avg_array)):
        for j in range(avg_array.shape[1]):
            # if i >= j:
            ax.text(j, i, avg_array[i, j], va='center', ha='center', c='w', fontsize=70 / num_tasks)

    ax.set_yticks(np.arange(num_tasks))
    ax.set_yticklabels(task_names)  # 设置y轴标签为任务名
    ax.set_ylabel('Number of tasks')

    ax.set_xticks(np.arange(num_tasks))
    ax.set_xticklabels(task_names)  # 设置x轴标签为任务名
    ax.set_xlabel('Tasks finished')

    ax.set_title(f"{plot_type} -- {np.round(np.mean(array[0, -1, :]), 4)}")

def acc_over_time_plot(ax, array):
    num_tasks = array.shape[1]
    acc_over_time = np.sum(array, axis=1) / np.arange(1, num_tasks + 1)
    mean, std = np.mean(acc_over_time, axis=0), np.std(acc_over_time, axis=0)
    ax.fill_between(np.arange(1, num_tasks + 1), mean - std, mean + std, alpha=0.3)
    ax.plot(np.arange(1, num_tasks + 1), mean)


def plot_final_results(names, rpath='results/'):
    fig = plt.figure(figsize=(32, 10 * len(names)))
    gs = GridSpec(len(names), 5)
    fig.suptitle(f"Experiment: {names[0]}\n")

    for e, name in enumerate(names):
        acc_dict = np.load(f"{rpath}{name}/accuracy.npy", allow_pickle=True).item()
        # print('acc_dict')
        # print(acc_dict)
        arr_acc, task_names = dict2array(acc_dict)  # 获取任务名称
        # print('arr_acc')
        # print(arr_acc)

        prec_dict = np.load(f"{rpath}{name}/precision.npy", allow_pickle=True).item()
        arr_prec, _ = dict2array(prec_dict)

        rec_dict = np.load(f"{rpath}{name}/recall.npy", allow_pickle=True).item()
        arr_rec, _ = dict2array(rec_dict)

        f1_dict = np.load(f"{rpath}{name}/f1_score.npy", allow_pickle=True).item()
        arr_f1, _ = dict2array(f1_dict)

        auc_dict = np.load(f"{rpath}{name}/roc_auc.npy", allow_pickle=True).item()
        arr_auc, _ = dict2array(auc_dict)

        ax1 = fig.add_subplot(gs[e, 0])
        ax2 = fig.add_subplot(gs[e, 1])
        ax3 = fig.add_subplot(gs[e, 2])
        ax4 = fig.add_subplot(gs[e, 3])
        ax5 = fig.add_subplot(gs[e, 4])

        grid_plot(ax1, arr_acc, name, "Accuracy", task_names)
        grid_plot(ax2, arr_prec, name, "precision", task_names)
        grid_plot(ax3, arr_rec, name, "recall", task_names)
        grid_plot(ax4, arr_f1, name, "F1 Score", task_names)
        grid_plot(ax5, arr_auc, name, "AUC", task_names)

    # plt.show()
    plt.savefig(rpath + names[0] + f"/results_visualisation", dpi=300)


def plot_anomalies(y_true, y_pred, recon_errors):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    recon_errors = np.array(recon_errors)

    # 生成横坐标：样本索引
    x = np.arange(len(recon_errors))

    plt.figure(figsize=(12, 6))

    # 绘制重构误差折线图
    plt.plot(x, recon_errors, label='Reconstruction Error', color='blue')

    # 标出真实异常点：真实标签为1的点
    true_anomaly_idx = np.where(y_true == 1)[0]
    plt.scatter(true_anomaly_idx, recon_errors[true_anomaly_idx],
                color='red', marker='o', s=80, label='True Anomalies')

    # 标出检测异常点：检测标签为1的点
    detected_anomaly_idx = np.where(y_pred == 1)[0]
    plt.scatter(detected_anomaly_idx, recon_errors[detected_anomaly_idx],
                color='orange', marker='x', s=80, label='Detected Anomalies')

    plt.xlabel('Sample Index')
    plt.ylabel('Reconstruction Error')
    plt.title('Reconstruction Error with Anomaly Points')
    plt.legend()
    plt.grid(True)
    plt.show()


# 示例调用
if __name__ == '__main__':
    # 示例数据
    y_true = [0, 0, 1, 0, 0, 1, 0, 0, 1, 0]
    y_pred = [0, 1, 1, 0, 0, 0, 0, 1, 1, 0]
    recon_errors = [0.1, 0.15, 0.8, 0.2, 0.18, 0.75, 0.12, 0.65, 0.9, 0.2]

    plot_anomalies(y_true, y_pred, recon_errors)
