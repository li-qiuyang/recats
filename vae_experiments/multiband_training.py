import torch
from vae_experiments import training_functions
import copy

# 局部和全局 VAE 模型的训练过程
def train_multiband(args,local_vae, curr_global_decoder, task_id, train_dataset_loader):


    if task_id == 0:
        n_epochs = args.gen_ae_epochs + args.global_dec_epochs
    else:
        n_epochs = args.gen_ae_epochs
    training_functions.train_local_generator(args, local_vae, task_loader=train_dataset_loader,
                                                             task_id=task_id,
                                                             n_epochs=n_epochs, local_start_lr=args.local_lr,
                                                             scheduler_rate=args.local_scheduler_rate,
                                                             scale_local_lr=args.scale_local_lr)

    print("Done training local VAE model")

    if not task_id:
        # 对于第一个任务，全局解码器还没有初始化，所以直接使用第一个任务的局部VAE解码器作为全局解码器的初始状态
        # First task, initializing global decoder as local_vae's decoder
        curr_global_decoder = copy.deepcopy(local_vae.decoder)
    else:
        print("Train global VAE model")

        curr_global_decoder = training_functions.train_global_decoder(curr_global_decoder=curr_global_decoder,
                                                                      local_vae=local_vae,
                                                                      task_id=task_id,
                                                                      n_iterations=len(train_dataset_loader),
                                                                      n_epochs=args.global_dec_epochs,
                                                                      batch_size=args.gen_batch_size,
                                                                      train_same_z=True,
                                                                      global_lr=args.global_lr,
                                                                      scheduler_rate=args.global_scheduler_rate,
                                                                      limit_previous_examples=args.limit_previous,
                                                                      train_loader=train_dataset_loader)
    torch.cuda.empty_cache()

    return curr_global_decoder
