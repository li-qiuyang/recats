for window_size in 50 100 200 240 270 300
do
    echo "正在运行window_size的参数: $window_size"
    python main.py \
        --weight_train 0.5 \
        --weight_test 0.5 \
        --modify 1 \
        --global_persent 70 \
        --clear_persent 95 \
        --window_size $window_size \
        --test_batch_size 40 \
        --gen_batch_size 50 \
        --experiment_name weather \
        --dataset weather \
        --gen_ae_epochs 50 \
        --global_dec_epochs 10

    python vae_experiments/validation.py > results/weather/weather_$window_size.log
done

# python main.py \
#     --weight_train 0.5 \
#     --weight_test 0.5 \
#     --modify 1 \
#     --global_persent 70 \
#     --clear_persent 95 \
#     --window_size 100 \
#     --test_batch_size 40 \
#     --gen_batch_size 50 \
#     --experiment_name weather \
#     --dataset weather \
#     --gen_ae_epochs 50 \
#     --global_dec_epochs 10
