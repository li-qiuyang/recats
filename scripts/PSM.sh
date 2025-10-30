# for window_size in 50 100 200 240 270 300
# do
#     echo "正在运行window_size的参数: $window_size"
#     python main.py \
#         --weight_train 0.1 \
#         --weight_test 0.9 \
#         --modify 1.77 \
#         --global_persent 98 \
#         --clear_persent 99 \
#         --window_size $window_size \
#         --test_batch_size 200 \
#         --gen_batch_size 64 \
#         --experiment_name PSM \
#         --dataset PSM \
#         --gen_ae_epochs 50 \
#         --global_dec_epochs 20

#     python vae_experiments/validation.py > results/PSM/PSM_$window_size.log
# done

for K in 1 3 5 10 15 20
do
    echo "正在运行K的参数: $K"
    python main.py \
        --weight_train 0.1 \
        --weight_test 0.9 \
        --modify 1.77 \
        --global_persent 98 \
        --clear_persent 99 \
        --window_size 270 \
        --test_batch_size 200 \
        --gen_batch_size 64 \
        --experiment_name PSM \
        --dataset PSM \
        --gen_ae_epochs 50 \
        --global_dec_epochs 20 \
        --K $K
    python vae_experiments/validation.py > results/PSM/PSM_K_$K.log
done
