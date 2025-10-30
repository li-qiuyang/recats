import sys
import argparse
import copy
import random
import torch
import torch.utils.data as data
from collections import OrderedDict

from continual_benchmark.dataloaders.dataset import preprocess_swat, preprocess_psm, \
    preprocess_weather, create_datasets, preprocess_smap
from vae_experiments import multiband_training, training_functions, models_definition

from vae_experiments.validation import calculate_metrics
from visualise import *

def run(args):
    print("Running baseline experiment on", args.dataset)
    if (args.dataset == 'SWaT'):
        # SWaT dataset
        hourly_data = preprocess_swat(
            path='./data/SWaT/SWaT_Dataset_Attack_v0.csv',
            window_size=30000,
            train_ratio=0.7
        )

        train_dataset_splits = create_datasets(hourly_data, mode='train', mask_ratio=0.5)
        test_dataset_splits = create_datasets(hourly_data, mode='test', mask_ratio=0)
    elif (args.dataset == 'PSM'):
        # PSM dataset
        psm_data = preprocess_psm(
            data_path="./data/PSM/test.csv",
            label_path="./data/PSM/test_label.csv",
            window_size=8000,
            train_ratio=0.7
        )
        # 划分数据集
        train_dataset_splits = create_datasets(psm_data, mode='train', mask_ratio=0.5)
        test_dataset_splits = create_datasets(psm_data, mode='test', mask_ratio=0.5)
    elif (args.dataset == 'SMAP'):
        psm_data = preprocess_smap(
            data_path="data/SMAP/SMAP_test.npy",
            label_path="data/SMAP/SMAP_test_label.npy",
            window_size=20000,
            train_ratio=0.7
        )
        # 划分数据集
        train_dataset_splits = create_datasets(psm_data, mode='train', mask_ratio=0.5)
        test_dataset_splits = create_datasets(psm_data, mode='test', mask_ratio=0)
    elif args.dataset == 'weather':
        weather_data = preprocess_weather('./data/weather/NEweather_data.csv', './data/weather/NEweather_class.csv',
                                          window_size=2000, train_ratio=0.7)

        train_dataset_splits = create_datasets(weather_data, mode='train', mask_ratio=0.3)
        test_dataset_splits = create_datasets(weather_data, mode='test', mask_ratio=0)
    elif args.dataset == 'MSL':
        # PSM dataset
        msl_data = preprocess_smap(
            data_path="data/MSL/MSL_test.npy",
            label_path="data/MSL/MSL_test_label.npy",
            window_size=8000,
            train_ratio=0.7
        )
        # 划分数据集
        train_dataset_splits = create_datasets(msl_data, mode='train', mask_ratio=0.5)
        test_dataset_splits = create_datasets(msl_data, mode='test', mask_ratio=0)
    n_tasks = len(train_dataset_splits)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    baseline_model = models_definition.GMVAE(args).to(device)
    train_loaders = []
    test_loaders = []

    for task_name in range(n_tasks):
        # 为每个任务准备训练和验证数据加载器
        train_dataset_loader = data.DataLoader(dataset=train_dataset_splits[task_name],
                                               batch_size=args.gen_batch_size, shuffle=True,
                                               drop_last=False)
        train_loaders.append(train_dataset_loader)
        test_data = test_dataset_splits[task_name]
        test_loader = data.DataLoader(dataset=test_data, batch_size=args.test_batch_size, shuffle=False,
                                      num_workers=args.workers)
        test_loaders.append(test_loader)

    baseline_auc = {}
    baseline_acc = {}
    baseline_f1  = {}
    baseline_pre = {}
    baseline_rec = {}
    # 初始化未训练的模型

    for task_id in range(n_tasks):

        test_dataset_loader = test_loaders[task_id]
        train_dataset_loader = train_loaders[task_id]
        # 计算基线AUC
        y_test = []
        for batch in test_dataset_loader:
            y_test.extend(batch['labels'].numpy())
        y_test = np.array(y_test)
        if sum(y_test) == 0:
            print("No anomaly detected in task", task_id)
        else:

            weighted_alpha = training_functions.compute_threshold(baseline_model, baseline_model.decoder, task_id, train_dataset_loader,
                                                                  test_dataset_loader,
                                                                  window_size=args.window_size,
                                                                  weight_train = args.weight_train,
                                                                  weight_test = args.weight_test,
                                                                  modify=args.modify,
                                                                  global_persent=args.global_persent,
                                                                  clear_persent = args.clear_persent)
            # 进行异常检测
            y_pred_test = training_functions.detect_anomalies(baseline_model, baseline_model.decoder, y_test,
                                                              test_dataset_loader,
                                                              weighted_alpha)
            accuracy,precision,recall,f1, roc_auc = calculate_metrics(y_test, y_pred_test)
            baseline_auc[task_id] = roc_auc
            baseline_acc[task_id] = accuracy
            baseline_f1[task_id] = f1
            baseline_pre[task_id] = precision
            baseline_rec[task_id] = recall
    print("Baseline AUCs:", baseline_auc)
    print("Baseline Accuracies:", baseline_acc)
    print("Baseline F1-scores:", baseline_f1)
    print("Baseline Precision:", baseline_pre)
    print("Baseline Recall:", baseline_rec)
    np.save(f"{args.rpath}{args.experiment_name}/baseline_auc.npy", baseline_auc)
    np.save(f"{args.rpath}{args.experiment_name}/baseline_acc.npy", baseline_acc)
    np.save(f"{args.rpath}{args.experiment_name}/baseline_f1.npy", baseline_f1)
    np.save(f"{args.rpath}{args.experiment_name}/baseline_pre.npy", baseline_pre)
    np.save(f"{args.rpath}{args.experiment_name}/baseline_rec.npy", baseline_rec)


def get_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_train', type=float,default=0.1)
    parser.add_argument('--weight_test', type=float,default=0.9)
    parser.add_argument('--modify', type=float, default=1.77)
    parser.add_argument('--global_persent', type=int, default=98)
    parser.add_argument('--clear_persent', type=int, default=99)
    parser.add_argument('--window_size', type=int, default=270, help='window size for compute local threshold')
    parser.add_argument('--K', type=int, default=10, metavar='N',
                        help='number of Gaussian')
    parser.add_argument('--gen_batch_size', type=int, default=64)
    parser.add_argument('--test_batch_size', type=int, default=200)
    parser.add_argument('--experiment_name', type=str, default='PSM', help='Name of current experiment')
    parser.add_argument('--dataset', type=str, default='PSM', help="dataset name")
    parser.add_argument('--rpath', type=str, default='results/', help='Directory to save results')
    parser.add_argument('--gpuid', nargs="+", type=int, default=[0],
                        help="The list of gpuid, ex:--gpuid 3 1. Negative value means cpu-only")
    parser.add_argument('--seed', type=int, default=13, required=False,
                        help="Random seed. If defined all random operations will be reproducible")
    parser.add_argument('--x-size', type=int, default=200, metavar='N',
                        help='dimension of x')
    parser.add_argument('--workers', type=int, default=0, help="#Thread for dataloader")
    parser.add_argument('--gen_latent_size', type=int, default=200, help="Latent size in VAE")

    args = parser.parse_args(argv)

    return args


if __name__ == '__main__':
    args = get_args(sys.argv[1:])
    torch.cuda.set_device(args.gpuid[0])
    device = torch.device("cuda")
    if args.seed:       #如果设置了 args.seed，则使用它作为随机种子，保证实验的可复现性
        print("Using manual seed = {}".format(args.seed))

        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    run(args)

