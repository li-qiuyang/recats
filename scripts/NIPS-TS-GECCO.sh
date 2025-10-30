# for window_size in 50 100 200 240 270 300 350
# do
#     echo "正在运行window_size的参数: $window_size"
#     python main.py \
#         --weight_train 0.5 \
#         --weight_test 0.5 \
#         --modify 1.4 \
#         --global_persent 99 \
#         --clear_persent 98 \
#         --window_size $window_size \
#         --test_batch_size 200 \
#         --gen_batch_size 64 \
#         --experiment_name GECCO \
#         --dataset GECCO \
#         --gen_ae_epochs 50 \
#         --global_dec_epochs 30
        
#     python vae_experiments/validation.py > results/GECCO/GECCO_$window_size.log
# done



for K in 1 3 5 10 15 20
do
    echo "正在运行K的参数: $K"
    python main.py \
        --weight_train 0.5 \
        --weight_test 0.5 \
        --modify 1.4 \
        --global_persent 99 \
        --clear_persent 98 \
        --window_size 270 \
        --test_batch_size 200 \
        --gen_batch_size 64 \
        --experiment_name GECCO \
        --dataset GECCO \
        --gen_ae_epochs 50 \
        --global_dec_epochs 30 \
        --K $K
        
    python vae_experiments/validation.py > results/GECCO/GECCO_K_$K.log
done
