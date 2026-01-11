import copy
import math
import time

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from vae_experiments.vae_utils import *
from vae_experiments.validation import prediction_adjust
from visualise import plot_anomalies

def loss_function(args, Y, x, mask, mu_w, logvar_w, qz, mu_x, logvar_x, mu_px, logvar_px):
    # 1. KL( q(w) || p(w) )
    KLD_W = -0.5 * torch.sum(1 + logvar_w - mu_w.pow(2) - logvar_w.exp())  # 计算 KL 散度

    # 2. KL( q(z) || p(z) )
    KLD_Z = torch.sum(qz * torch.log(args.K * qz + 1e-10))

    # 3. E_z_w[KL(q(x) || p(x|z,w))]
    mu_x = mu_x.unsqueeze(-1).expand(-1, args.x_size, args.K)
    logvar_x = logvar_x.unsqueeze(-1).expand(-1, args.x_size, args.K)
    KLD_QX_PX = 0.5 * (
            (logvar_px - logvar_x) +
            ((logvar_x.exp() + (mu_x - mu_px).pow(2)) / logvar_px.exp()) - 1
    )

    E_KLD_QX_PX = torch.sum(torch.bmm(KLD_QX_PX, qz.unsqueeze(-1)))

    # 4. 重构损失
    # recon_loss = torch.mean((Y[mask.bool()]- x[mask.bool()]) ** 2)
    recon_loss = torch.mean((Y- x) ** 2)
    # marginal_loss = nn.MSELoss(reduction="sum")
    # recon_loss = marginal_loss(Y, x) / Y.size(0)

    # 总损失
    loss = recon_loss + 1.0 * KLD_W + 0.001 * KLD_Z + 0.001 * E_KLD_QX_PX

    return loss , KLD_W, KLD_Z, KLD_QX_PX, E_KLD_QX_PX, recon_loss

def train_local_generator(args, local_vae, task_loader, task_id, n_epochs=100,
                          local_start_lr=0.001, scheduler_rate=0.99, scale_local_lr=False):
    # 将局部VAE模型和解码器的翻译器部分设置为训练模式，以确保模型的参数在训练过程中可以被更新
    local_vae.train()

    assert (not scale_local_lr)
    lr = local_start_lr
    print(f"lr set to: {lr}")

    if task_id > 0:  # 如果不是第一个任务  仅训练VAE的编码器部分.使用较小的学习率（lr / 10）
        optimizer = torch.optim.Adam(list(local_vae.encoder.parameters()), lr=lr / 10, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=scheduler_rate)
    else:    # 如果是第一个任务，则对整个VAE模型进行训练，包括编码器和解码器
        optimizer = torch.optim.Adam(list(local_vae.parameters()), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)  # 学习率调度器 在训练过程中逐步衰减学习率

    for epoch in range(n_epochs):

        if (task_id != 0) and (epoch == min(20, max(n_epochs // 10, 5))):
            print("End of local_vae pretraining")
            optimizer = torch.optim.Adam(list(local_vae.parameters()), lr=lr, weight_decay=1e-5)
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=scheduler_rate)


        for iteration, batch in enumerate(task_loader):

            x = batch['masked_data'].to(local_vae.device)
            original_x = batch['original_data'].to(local_vae.device)
            mask = batch['gt_mask'].to(local_vae.device)
            mu_x, logvar_x, mu_px, logvar_px, qz, Y, mu_w, logvar_w = local_vae(x)
            logvar_x = torch.clamp(logvar_x, min=-20.0, max=10.0)
            loss, KLD_W, KLD_Z, KLD_QX_PX, E_KLD_QX_PX, recon_loss = loss_function(
                args, Y, original_x, mask, mu_w, logvar_w, qz, mu_x, logvar_x, mu_px, logvar_px,
            )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_value_(local_vae.parameters(), 4.0)
            optimizer.step()

        scheduler.step()
        print("epoch: " + str(epoch) + " loss: " + str(loss.item()) + " KLD_W: " + str(KLD_W.item()) +
              " KLD_Z: " + str(KLD_Z.item()) + " E_KLD_QX_PX: " + str(E_KLD_QX_PX.item()) +
              " recon_loss: " + str(recon_loss.item()))

def train_global_decoder(curr_global_decoder, local_vae, task_id, n_epochs=100, n_iterations=30, batch_size=1000,
                         train_same_z=False,global_lr=0.0001, scheduler_rate=0.99, limit_previous_examples=1,
                         train_loader=None,):

    global_decoder = copy.deepcopy(curr_global_decoder)
    #  将当前的全局解码器、翻译器和局部VAE模型设为评估模式，以确保不更新它们的参数。
    #  将新创建的全局解码器设为训练模式, 允许参数更新
    curr_global_decoder.eval()
    # curr_global_decoder.translator.eval()
    local_vae.eval()
    global_decoder.train()

    # criterion = nn.BCELoss(reduction="sum")
    criterion = nn.MSELoss(reduction="sum")

    n_prev_examples = int(batch_size * min(task_id, 5) * limit_previous_examples)


    # 将当前全局解码器存储到 tmp_decoder
    tmp_decoder = curr_global_decoder
    # attn = nn.MultiheadAttention(embed_dim=200, num_heads=4, batch_first=False)
    for epoch in range(n_epochs):
        losses = []
        start = time.time()

        optimizer = torch.optim.Adam(list(global_decoder.parameters()), lr=global_lr)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=scheduler_rate)

        # 迭代训练
        for iteration in range(n_iterations):  # n_iterations：每轮训练的迭代次数
            # recon_prev是由全局解码器生成的先前任务的样本，z_prev是由全局解码器生成的先前任务的潜在变量
            recon_prev, z_prev, task_ids_prev = generate_previous_data(
                tmp_decoder,
                n_tasks=task_id,
                n_prev_examples=n_prev_examples,
                num_local=batch_size,
                return_z=True,
                translate_noise=True,
                same_z=train_same_z)
            

            with torch.no_grad():
                recon_local = next(iter(train_loader))
                task_ids_local = torch.zeros([len(recon_local)])
                recon_local = recon_local['original_data'].to(local_vae.device)
                # means, log_var = local_vae.encoder(recon_local)
                qz, means, log_var, mu_w, logvar_w = local_vae.encoder(recon_local)
                log_var = torch.clamp(log_var, min=-20.0, max=10.0)
                std = torch.exp(0.5 * log_var)
                eps  = torch.randn_like(std).to(local_vae.device)
                z_local = eps * std + means

            # 将来自先前任务的潜在变量 (z_prev) 与当前任务的潜在变量 (z_local) 进行拼接
            z_concat = torch.cat([z_prev, z_local])
            task_ids_concat = torch.cat([task_ids_prev, task_ids_local])

            recon_concat = torch.cat([recon_prev, recon_local])

            n_mini_batches = math.ceil(len(z_concat) / batch_size)
            # print(n_mini_batches)  #3
            # print(batch_size) #32
            # print(len(recon_prev))  #35
            # print(len(recon_local)) #32
            # print(len(task_ids_concat))39
            shuffle = torch.randperm(len(task_ids_concat))
            z_concat = z_concat[shuffle]
            task_ids_concat = task_ids_concat[shuffle]
            recon_concat = recon_concat[shuffle]

            for batch_id in range(n_mini_batches):
                global_decoder.zero_grad()
                start_point = batch_id * batch_size
                end_point = min(len(task_ids_concat), (batch_id + 1) * batch_size)
                if start_point >= end_point:
                    continue
                global_recon = global_decoder(z_concat[start_point:end_point],mode="global")
                # gloal_recon是当前全局解码器生成的样本，recon_concat是由先前任务和当前任务的样本拼接得到的样本
                loss = criterion(global_recon, recon_concat[start_point:end_point]) #.sum(dim=-1).mean(dim=-1)
 
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

        scheduler.step()
        if (epoch % 1 == 0):
            print("Epoch: {}/{}, loss: {}, took: {} s".format(epoch + 1, n_epochs, np.mean(losses), time.time() - start))
    return global_decoder

def compute_threshold(local_vae, curr_global_decoder, task_id, train_loader, test_loader, window_size=270, weight_train=0.1, weight_test=0.9,q=1.77, global_persent=98, clear_persent=99):
    local_vae.eval()
    curr_global_decoder.eval()
    if train_loader is not None:
        train_reconstruction_errors = []
        with torch.no_grad():
            for batch in train_loader:
                x = batch

                x = x['original_data'].to(local_vae.device, dtype=torch.float32)
                # print(x.shape[0])
                _, means, log_var, _, _ = local_vae.encoder(x)
                log_var = torch.clamp(log_var, min=-20.0, max=10.0)
                std = torch.exp(0.5 * log_var)
                eps = torch.randn_like(std)
                z_local = eps * std + means
                recon_x = curr_global_decoder(z_local)
                # loss = F.binary_cross_entropy(recon_x,x,reduction='none').mean(dim=-1).cpu().numpy()

                mse_loss_fn = nn.MSELoss(reduction='none')
                loss = mse_loss_fn(recon_x,x).mean(dim=-1).cpu().numpy()
                train_reconstruction_errors.extend(_ for _ in loss)

        alpha = np.percentile(train_reconstruction_errors, global_persent)*q


    thresholds = []
    test_reconstruction_errors = []
    with torch.no_grad():
        for batch in test_loader:
            x = batch
            x = x['original_data'].to(local_vae.device)

            # print(x.shape[0])
            _, means, log_var, _, _ = local_vae.encoder(x)
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std).to(local_vae.device)
            # print(std.shape)
            z_local = eps * std + means
            recon_x = curr_global_decoder(z_local)
            mse_loss_fn = nn.MSELoss(reduction='none')
            loss = mse_loss_fn(recon_x, x).mean(dim=-1).cpu().numpy()
            test_reconstruction_errors.extend(_ for _ in loss)

    for idx in range(len(test_reconstruction_errors)):
        start_idx = max(0, idx - window_size // 2)
        end_idx = min(len(test_reconstruction_errors), idx + window_size // 2)
        local_errors = np.array(test_reconstruction_errors[start_idx:end_idx])

        local_errors_cleaned = local_errors[local_errors < np.percentile(local_errors, clear_persent)]  # 95

        mean = np.mean(local_errors_cleaned)
        std = np.std(local_errors_cleaned)
        local_threshold = mean + 3 * std

        if train_loader is not None:
            window_threshold = weight_train * alpha + weight_test * local_threshold
        else:
            window_threshold = local_threshold
        thresholds.append(window_threshold)
    return thresholds

def detect_anomalies(local_vae, curr_global_decoder, y_true, test_loader, thresholds):
    local_vae.eval()
    curr_global_decoder.eval()
    anomalies = []
    reconstruction_errors = []

    with torch.no_grad():

        for idx, batch in enumerate(test_loader):
            x = batch

            x = x['original_data'].to(local_vae.device, dtype=torch.float32)
            _, mu_x, logvar_x, _, _ = local_vae.encoder(x)
            std_x = torch.exp(0.5 * logvar_x)
            eps = torch.randn_like(std_x)
            z = mu_x + eps * std_x
            reconstructed_x = curr_global_decoder(z)
            mse_loss_fn = nn.MSELoss(reduction='none')

            loss = mse_loss_fn(reconstructed_x, x).mean(dim=-1).cpu().numpy()
            reconstruction_errors.extend(_ for _ in loss)

    for idx, error in enumerate(reconstruction_errors):
        window_threshold = thresholds[idx]
        flag = error > window_threshold
        anomalies.append(flag)

    y_pred_test = [1 if anomaly else 0 for anomaly in anomalies]
    y_pred_test = prediction_adjust(y_pred_test, y_true)
    return y_pred_test
