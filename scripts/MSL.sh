python main.py \
    --weight_train 0.6 \
    --weight_test 0.4 \
    --modify 1.95 \
    --global_persent 98 \
    --clear_persent 98 \
    --window_size 240 \
    --test_batch_size 200 \
    --gen_batch_size 64 \
    --experiment_name MSL \
    --dataset MSL \
    --gen_ae_epochs 50 \
    --global_dec_epochs 20

python vae_experiments/validation.py 