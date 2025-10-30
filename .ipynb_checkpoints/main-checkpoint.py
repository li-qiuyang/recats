import sys
import argparse
import copy
import random
import torch
import torch.utils.data as data
from random import shuffle
from collections import OrderedDict

from continual_benchmark.dataloaders.dataset import create_hourly_datasets, preprocess_swat, preprocess_psm, \
    create_window_datasets
from vae_experiments import multiband_training, training_functions

from vae_experiments.validation import prediction_adjust, calculate_metrics
from visualise import *

def run(args):
    if (args.dataset == 'SWaT'):
        # SWaT dataset
        hourly_data = preprocess_swat(path='./data/SWaT/SWaT_Dataset_Attack_v0.csv')

        train_dataset_splits = create_hourly_datasets(hourly_data, mode='train', mask_ratio=0.5)
        test_dataset_splits = create_hourly_datasets(hourly_data, mode='test', mask_ratio=0)
    elif (args.dataset == 'PSM'):
        # PSM dataset
        psm_data = preprocess_psm(
            data_path="./data/PSM/test.csv",
            label_path="./data/PSM/test_label.csv",
            window_size=8000,
            train_ratio=0.7
        )
        # 划分数据集
        train_dataset_splits = create_window_datasets(psm_data, mode='train')
        test_dataset_splits = create_window_datasets(psm_data, mode='test')

    if args.dataset == 'SWaT':
        n_tasks = 24
        task_names = list(range(24))
    elif args.dataset == 'PSM':
        n_tasks = 11  # 任务数
        task_names = list(range(11))

    from vae_experiments import models_definition

    print('Task order:', task_names)
    if args.rand_split_order:
        shuffle(task_names)   # 随机打乱任务顺序
        print('Shuffled task order:', task_names)

    precision_table = OrderedDict()   #字典，用于存储每个任务的精度
    recall_table = OrderedDict()
    accuracy_table = OrderedDict()
    f1_table = OrderedDict()
    auc_table = OrderedDict()

    # Prepare GMVAE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    local_vae = models_definition.GMVAE(args).to(device)
    print(local_vae)

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

    curr_global_decoder = None

    for task_id in range(len(task_names)):
        print("######### Task number {} #########".format(task_id))
        task_name = task_names[task_id]

        # VAE
        print("Train local VAE model")
        train_dataset_loader = train_loaders[task_id]
        test_dataset_loader = test_loaders[task_id]
        curr_global_decoder = multiband_training.train_multiband(args=args,
                                                                 local_vae=local_vae,
                                                                 curr_global_decoder=curr_global_decoder,
                                                                 task_id=task_id,
                                                                 train_dataset_loader=train_dataset_loader,
                                                                 device=device)


        precision_table[task_name] = OrderedDict()
        recall_table[task_name] = OrderedDict()
        accuracy_table[task_name] = OrderedDict()
        f1_table[task_name] = OrderedDict()
        auc_table[task_name] = OrderedDict()

        local_vae.decoder = copy.deepcopy(curr_global_decoder)

        # 获取测试集真实标签 y_test
        y_test = []
        for batch in test_dataset_loader:
            y_test.extend(batch['labels'].numpy())
        y_test = np.array(y_test)
        if sum(y_test) == 0:
            print(f"Task {task_id} (Test) has no anomalies. Skipping evaluation.")
        else:
            # 计算加权阈值
            weighted_alpha = training_functions.compute_threshold(local_vae, curr_global_decoder, task_id, train_dataset_loader,
                                                                  test_dataset_loader,
                                                                  window_size=args.window_size,
                                                                  weight_train = args.weight_train,
                                                                  weight_test = args.weight_test,
                                                                  modify=args.modify,
                                                                  persent=args.persent)
            # 进行异常检测
            anomalies_test = training_functions.detect_anomalies(local_vae, curr_global_decoder, task_id, test_dataset_loader,
                                                                 weighted_alpha)

            y_pred_test = [1 if anomaly else 0 for anomaly in anomalies_test]
            adjust_prediction = prediction_adjust(y_pred_test, y_test)
            # 计算指标
            accuracy, precision, recall, f1, roc_auc = calculate_metrics(y_test, adjust_prediction)
            print(f"\nEvaluation Metrics for Task {task_id} (Test):")
            print('测试样本总数：', len(y_test))
            print('异常样本数量:', np.sum(y_test == 1))
            print('检测出异常样本数量:', adjust_prediction.count(1))
            print('accuracy:', accuracy)
            print('precision:', precision)
            print('recall:', recall)
            print('f1_socre:', f1)
            print('roc_auc:', roc_auc)
            # 保存指标
            accuracy_table[task_name][task_name] = accuracy
            precision_table[task_name][task_name] = precision
            recall_table[task_name][task_name] = recall
            f1_table[task_name][task_name] = f1
            auc_table[task_name][task_name] = roc_auc

            for prev_task_id in range(task_id):
                prev_task_name = task_names[prev_task_id]

                # 获取测试集真实标签 y_test
                y_test = []
                for batch in test_loaders[prev_task_id]:
                    y_test.extend(batch['labels'].numpy())  # 将每个批次的标签添加到 y_test 列表中
                y_test = np.array(y_test)

                if sum(y_test) == 0:
                    print(f"Task {prev_task_id} (Test) has no anomalies. Skipping evaluation.")
                else:
                    # 计算加权阈值
                    weighted_alpha = training_functions.compute_threshold(local_vae, curr_global_decoder, prev_task_id,
                                                                          train_loaders[prev_task_id],
                                                                          test_loaders[prev_task_id],
                                                                          window_size=args.window_size,
                                                                          weight_train=args.weight_train,
                                                                          weight_test=args.weight_test,
                                                                          modify=args.modify,
                                                                          persent=args.persent
                                                                          )

                    # 进行异常检测
                    anomalies_test = training_functions.detect_anomalies(local_vae, curr_global_decoder, prev_task_id,
                                                                         test_loaders[prev_task_id],
                                                                         weighted_alpha)

                    y_pred_test = [1 if anomaly else 0 for anomaly in anomalies_test]
                    adjust_prediction = prediction_adjust(y_pred_test, y_test)
                    # 计算指标
                    accuracy, precision, recall, f1, roc_auc = calculate_metrics(y_test, adjust_prediction)
                    print(f"\nEvaluation Metrics for Task {prev_task_id} (Test):")
                    print('测试样本总数：', len(y_test))
                    print('异常样本数量:', np.sum(y_test == 1))
                    print('检测出异常样本数量:', adjust_prediction.count(1))
                    print('accuracy:', accuracy)
                    print('precision:', precision)
                    print('recall:', recall)
                    print('f1_socre:', f1)
                    print('roc_auc:', roc_auc)
                    # 将用当前任务训练的模型对先前任务的精度添加到 precision_table
                    precision_table[task_name][prev_task_name] = precision
                    recall_table[task_name][prev_task_name] = recall
                    accuracy_table[task_name][prev_task_name] = accuracy
                    f1_table[task_name][prev_task_name] = f1
                    auc_table[task_name][prev_task_name] = roc_auc


            # 评估未来的任务
            for eval_task_id in range(task_id + 1, n_tasks):
                eval_task_name = task_names[eval_task_id]

                # 获取测试集真实标签 y_test
                y_test = []
                for batch in test_loaders[eval_task_id]:
                    y_test.extend(batch['labels'].numpy())  # 将每个批次的标签添加到 y_test 列表中
                y_test = np.array(y_test)

                if sum(y_test) == 0:
                    print(f"Task {eval_task_id} (Test) has no anomalies. Skipping evaluation.")
                else:
                    # 计算加权阈值
                    weighted_alpha = training_functions.compute_threshold(local_vae, curr_global_decoder, eval_task_id,
                                                                          train_loaders[eval_task_id],
                                                                          test_loaders[eval_task_id],
                                                                          window_size=args.window_size,
                                                                          weight_train=args.weight_train,
                                                                          weight_test=args.weight_test,
                                                                          modify=args.modify,
                                                                          persent=args.persent
                                                                          )

                    # 进行异常检测
                    anomalies_test = training_functions.detect_anomalies(local_vae, curr_global_decoder, eval_task_id,
                                                                         test_loaders[eval_task_id],
                                                                         weighted_alpha)

                    y_pred_test = [1 if anomaly else 0 for anomaly in anomalies_test]
                    adjust_prediction = prediction_adjust(y_pred_test, y_test)
                    # 计算指标
                    accuracy, precision, recall, f1, roc_auc = calculate_metrics(y_test, adjust_prediction)
                    print(f"\nEvaluation Metrics for Task {eval_task_id} (Test):")
                    print('测试样本总数：', len(y_test))
                    print('异常样本数量:', np.sum(y_test == 1))
                    print('检测出异常样本数量:', adjust_prediction.count(1))
                    print('accuracy:', accuracy)
                    print('precision:', precision)
                    print('recall:', recall)
                    print('f1_socre:', f1)
                    print('roc_auc:', roc_auc)

                    # 记录评估指标
                    precision_table[task_name][eval_task_name] = precision
                    recall_table[task_name][eval_task_name] = recall
                    accuracy_table[task_name][eval_task_name] = accuracy
                    f1_table[task_name][eval_task_name] = f1
                    auc_table[task_name][eval_task_name] = roc_auc

    return task_names, precision_table, recall_table, accuracy_table, f1_table, auc_table

def get_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_train', type=float,default=0.2)
    parser.add_argument('--weight_test', type=float,default=0.8)
    parser.add_argument('--modify', type=float, default=1.66)
    parser.add_argument('--persent', type=int, default=90)
    parser.add_argument('--window_size', type=int, default=128, help='window size for compute local threshold')
    parser.add_argument('--K', type=int, default=10, metavar='N',
                        help='number of Gaussian')
    parser.add_argument('--x-size', type=int, default=200, metavar='N',
                        help='dimension of x')
    parser.add_argument('--hidden-size', type=int, default=512, metavar='N',
                        help='dimension of hidden layer')
    parser.add_argument('--w-size', type=int, default=32, metavar='N',
                        help='dimension of w')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='learning rate for optimizer')
    # General
    parser.add_argument('--experiment_name', type=str, default='SWaT', help='Name of current experiment')
    parser.add_argument('--rpath', type=str, default='results/', help='Directory to save results')
    parser.add_argument('--gpuid', nargs="+", type=int, default=[0],
                        help="The list of gpuid, ex:--gpuid 3 1. Negative value means cpu-only")
    parser.add_argument('--repeat', type=int, default=1, help="Repeat the experiment N times")
    parser.add_argument('--seed', type=int, default=13, required=False,
                        help="Random seed. If defined all random operations will be reproducible")
    parser.add_argument('--score_on_val', action='store_true', required=False, default=False,
                        help="Compute FID on validation dataset instead of validation dataset")
    parser.add_argument('--test_batch_size', type=int, default=32)
    parser.add_argument('--skip_validation', default=False, action='store_true')
    parser.add_argument('--training_procedure', type=str, default='multiband',
                        help='Training procedure multiband|replay')

    # Data
    parser.add_argument('--dataroot', type=str, default='data', help="The root folder of dataset or downloaded data")
    parser.add_argument('--dataset', type=str, default='SWaT', help="dataset name")
    parser.add_argument('--n_permutation', type=int, default=0, help="Enable permuted tests when >0")
    # parser.add_argument('--first_split_size', type=int, default=2)
    parser.add_argument('--num_batches', type=int, default=5)
    parser.add_argument('--rand_split', dest='rand_split', default=False, action='store_true',
                        help="Randomize the classes in splits")
    parser.add_argument('--rand_split_order', dest='rand_split_order', default=False, action='store_true',
                        help="Randomize the order of splits")
    parser.add_argument('--random_split', dest='random_split', default=False, action='store_true',
                        help="Randomize data in splits")
    parser.add_argument('--limit_data', type=float, default=None,
                        help="limit_data to given %")
    parser.add_argument('--random_shuffle', dest='random_shuffle', default=False, action='store_true',
                        help="Move part of data to next batch")
    parser.add_argument('--no_class_remap', dest='no_class_remap', default=False, action='store_true',
                        help="Avoid the dataset with a subset of classes doing the remapping. Ex: [2,5,6 ...] -> [0,1,2 ...]")
    parser.add_argument('--skip_normalization', action='store_true', help='Loads dataset without normalization')
    parser.add_argument('--train_aug', dest='train_aug', default=False, action='store_true',
                        help="Allow data augmentation during training")
    parser.add_argument('--workers', type=int, default=0, help="#Thread for dataloader")

    # Generative network - multiband vae
    parser.add_argument('--gen_batch_size', type=int, default=32)
    parser.add_argument('--local_lr', type=float, default=0.001)
    parser.add_argument('--local_scheduler_rate', type=float, default=0.99)
    parser.add_argument('--scale_local_lr', default=False, action='store_true',
                        help="Scale lr of local model based on the reconstruction error")
    parser.add_argument('--scale_reconstruction_loss', type=float, default=1)
    parser.add_argument('--global_lr', type=float, default=0.001)
    parser.add_argument('--global_scheduler_rate', type=float, default=0.99)
    parser.add_argument('--gen_n_dim_coding', type=int, default=4,
                        help="Number of bits used to code task id in binary autoencoder")
    parser.add_argument('--gen_p_coding', type=int, default=9,
                        help="Prime number used to calculated codes in binary autoencoder")
    parser.add_argument('--gen_cond_n_dim_coding', type=int, default=0,
                        help="Number of bits used to code task id in binary autoencoder")
    parser.add_argument('--gen_cond_p_coding', type=int, default=9,
                        help="Prime number used to calculated codes in binary autoencoder")
    parser.add_argument('--gen_latent_size', type=int, default=200, help="Latent size in VAE")
    parser.add_argument('--binary_latent_size', type=int, default=4, help="Binary latent size in VAE")
    parser.add_argument('--gen_d', type=int, default=8, help="Size of binary autoencoder")
    parser.add_argument('--gen_ae_epochs', type=int, default=40,  #40
                        help="Number of epochs to train local variational autoencoder")
    parser.add_argument('--global_dec_epochs', type=int, default=10, help="Number of epochs to train global decoder")
    parser.add_argument('--gen_load_pretrained_models', default=False, action='store_true',
                        help="Load pretrained generative models")
    parser.add_argument('--gen_pretrained_models_dir', type=str, default="results/pretrained_models",
                        help="Directory of pretrained generative models")
    parser.add_argument('--standard_embeddings', dest='standard_embeddings', default=False, action='store_true',
                        help="Train multiband with standard embeddings instead of matrix")
    parser.add_argument('--trainable_embeddings', dest='trainable_embeddings', default=False, action='store_true',
                        help="Train multiband with trainable embeddings instead of matrix")
    parser.add_argument('--fc', default=True, action='store_true',
                        help="Use only dense layers in VAE model")
    parser.add_argument('--cosine_sim', default=1.0, type=float,
                        help="Cosine similarity between examples to merge")
    parser.add_argument('--limit_previous', default=1, type=float,
                        help="How much of previous data we want to generate each epoch")
    parser.add_argument('--global_warmup', default=5, type=int,
                        help="Number of epochs for global warmup - only translator training")
    parser.add_argument('--generations_for_switch', default=1000, type=int,
                        help="Number of noise instances we want to create in order to select instances pos")
    parser.add_argument('--visualise_latent', default=False, action='store_true',
                        help="Whether to visualise latent space")

    args = parser.parse_args(argv)

    if args.trainable_embeddings:
        args.standard_embeddings = True

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
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        print("WARNING: Not using manual seed - your experiments will not be reproducible")

    f1_table, acc_table, precision_table, recall_table, auc_table = {}, {}, {}, {}, {}
    os.makedirs(f"{args.rpath}{args.experiment_name}", exist_ok=True)
    with open(f"{args.rpath}{args.experiment_name}/args.txt", "w") as text_file:
        text_file.write(str(args))
    for r in range(args.repeat):
        task_names, precision_table[r], recall_table[r], acc_table[r], f1_table[r], auc_table[r] = run(args)

    np.save(f"{args.rpath}{args.experiment_name}/precision.npy", precision_table)
    np.save(f"{args.rpath}{args.experiment_name}/recall.npy", recall_table)
    np.save(f"{args.rpath}{args.experiment_name}/accuracy.npy", acc_table)
    np.save(f"{args.rpath}{args.experiment_name}/f1_score.npy", f1_table)
    np.save(f"{args.rpath}{args.experiment_name}/roc_auc.npy", auc_table)
    plot_final_results([args.experiment_name])
