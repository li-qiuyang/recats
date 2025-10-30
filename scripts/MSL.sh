for window_size in 270 300
do
    echo "正在运行window_size的参数: $window_size"
    python main.py \
        --weight_train 0.6 \
        --weight_test 0.4 \
        --modify 1.95 \
        --global_persent 98 \
        --clear_persent 98 \
        --window_size $window_size \
        --test_batch_size 200 \
        --gen_batch_size 64 \
        --experiment_name MSL \
        --dataset MSL \
        --gen_ae_epochs 50 \
        --global_dec_epochs 20

    python vae_experiments/validation.py > results/MSL/MSL_$window_size.log
done




# python main.py \
#     --weight_train 0.6 \
#     --weight_test 0.4 \
#     --modify 1.95 \
#     --global_persent 98 \
#     --clear_persent 98 \
#     --window_size 256 \
#     --test_batch_size 200 \
#     --gen_batch_size 64 \
#     --experiment_name MSL \
#     --dataset MSL \
#     --gen_ae_epochs 50 \
#     --global_dec_epochs 20
