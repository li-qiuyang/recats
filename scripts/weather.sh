python main.py \
    --weight_train 0.5 \
    --weight_test 0.5 \
    --q 1 \
    --global_persent 70 \
    --clear_persent 95 \
    --window_size 100 \
    --test_batch_size 40 \
    --gen_batch_size 50 \
    --experiment_name weather \
    --dataset weather \
    --gen_ae_epochs 50 \
    --global_dec_epochs 10

python vae_experiments/validation.py