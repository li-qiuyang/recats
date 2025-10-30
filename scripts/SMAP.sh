# for window_size in 50 100 200 240 270 300
# do
#     echo "正在运行window_size的参数: $window_size"
#     python main.py \
#         --weight_train 0.4 \
#         --weight_test 0.6 \
#         --modify 1.7 \
#         --global_persent 95 \
#         --clear_persent 78 \
#         --window_size $window_size \
#         --test_batch_size 200 \
#         --gen_batch_size 128 \
#         --experiment_name SMAP \
#         --dataset SMAP \
#         --gen_ae_epochs 60 \
#         --global_dec_epochs 20

#     python vae_experiments/validation.py > results/SMAP/SMAP_$window_size.log
# done



for K in 1 3 5 10 15 20
do
    echo "正在运行K的参数: $K"
    python main.py \
        --weight_train 0.4 \
        --weight_test 0.6 \
        --modify 1.7 \
        --global_persent 95 \
        --clear_persent 78 \
        --window_size 270 \
        --test_batch_size 200 \
        --gen_batch_size 128 \
        --experiment_name SMAP \
        --dataset SMAP \
        --K $K \
        --gen_ae_epochs 60 \
        --global_dec_epochs 20

    python vae_experiments/validation.py > results/SMAP/SMAP_K_$K.log
done
