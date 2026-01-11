python main.py \
    --weight_train 0.4 \
    --weight_test 0.6 \
    --q 1.7 \
    --global_persent 95 \
    --clear_persent 78 \
    --window_size 270 \
    --test_batch_size 200 \
    --gen_batch_size 128 \
    --experiment_name SMAP \
    --dataset SMAP \
    --K 10 \
    --gen_ae_epochs 60 \
    --global_dec_epochs 20

python vae_experiments/validation.py 