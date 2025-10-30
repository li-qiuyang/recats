
import numpy as np
import torch


class BitUnpacker:
    results_map = {}

    @classmethod
    def unpackbits(cls, y, num_bits):
        with torch.no_grad():
            x = y + 1
            if num_bits == 0:
                return torch.Tensor([])

            if num_bits in cls.results_map:
                mask = cls.results_map[num_bits]
            else:
                print("Mask for num_bits={} does not exist, calculating one.".format(num_bits))

                mask = 2 ** (num_bits - 1 - torch.arange(num_bits).view([1, num_bits])).long()
                cls.results_map[num_bits] = mask

            x = x.view(-1, 1).long()

            return (x & mask).bool().float() * 2 - 1


def prepare_class_samplres(task_id, class_table):
    class_samplers = []
    for task_id in range(task_id):
        local_probs = class_table[task_id] * 1.0 / torch.sum(class_table[task_id])
        class_samplers.append(torch.distributions.categorical.Categorical(probs=local_probs))
    return class_samplers


def generate_images(curr_global_decoder, z, task_ids):
    example = curr_global_decoder(z)
    return example


def generate_noise_for_previous_data(n_prev_examples, n_task, latent_size, tasks_dist, device, num_local=0, same_z=False):
    if same_z:
        tasks_dist_tensor = torch.tensor(tasks_dist, device=device)  # 将 tasks_dist 转换为张量
        # 获取最大值并将其转换为整数
        max_value = max(torch.cat([tasks_dist_tensor, torch.tensor([num_local], device=device)]).tolist())
        max_value = int(max_value)
        z_max = torch.randn((max_value, latent_size)).to(device)
        z = []

        for task_id, n_prev_examples in enumerate(tasks_dist):
            n_prev_examples = n_prev_examples.item() if isinstance(n_prev_examples, torch.Tensor) else n_prev_examples
            # print(n_prev_examples)
            # print(type(n_prev_examples))  float
            z.append(z_max[:int(n_prev_examples)])
        z = torch.cat(z)
        return z, z_max
    else:
        z = torch.randn([n_prev_examples, latent_size]).to(device)
        return z




def generate_previous_data(curr_global_decoder, n_tasks, n_prev_examples, num_local=0, translate_noise=True,
                           same_z=False, return_z=False):
    # 生成来自先前任务的样本数据
    with torch.no_grad():
        # 计算每个任务的样本分配
        tasks_dist = torch.ones(n_tasks) * n_prev_examples // n_tasks  # 初始化每个任务的样本数量

        remaining = n_prev_examples - tasks_dist.sum().int()  # 计算剩余的样本数
        # print("n_prev_examples: ", n_prev_examples)  32
        tasks_dist[0:remaining] += 1
        assert sum(tasks_dist) == n_prev_examples  # 确保样本总数正确

        task_ids = []
        for task_id in range(n_tasks):
            if tasks_dist[task_id] > 0:
                task_ids.append([task_id] * int(tasks_dist[task_id]))  #将当前任务的ID（task_id）重复 tasks_dist[task_id] 次

        task_ids = torch.from_numpy(np.concatenate(task_ids)).float()
        assert len(task_ids) == n_prev_examples

        # 生成来自先前任务的噪声
        z_combined = generate_noise_for_previous_data(n_prev_examples, n_tasks, curr_global_decoder.latent_size,tasks_dist,
                                                      device=curr_global_decoder.device, num_local=num_local,
                                                      same_z=same_z)

        if same_z:
            z, _ = z_combined

        if return_z:  ######

            example = generate_images(curr_global_decoder, z, task_ids)
            return example, z_combined, task_ids

