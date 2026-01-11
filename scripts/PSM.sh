python main.py \
    --weight_train 0.1 \
    --weight_test 0.9 \
    --q 1.77 \
    --global_persent 98 \
    --clear_persent 99 \
    --window_size 270 \
    --test_batch_size 200 \
    --gen_batch_size 64 \
    --experiment_name PSM \
    --dataset PSM \
    --gen_ae_epochs 50 \
    --global_dec_epochs 20 \
    --K 10
python vae_experiments/validation.py 
