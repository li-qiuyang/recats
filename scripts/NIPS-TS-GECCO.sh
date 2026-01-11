python main.py \
    --weight_train 0.5 \
    --weight_test 0.5 \
    --q 1.4 \
    --global_persent 99 \
    --clear_persent 98 \
    --window_size 270 \
    --test_batch_size 200 \
    --gen_batch_size 64 \
    --experiment_name GECCO \
    --dataset GECCO \
    --gen_ae_epochs 50 \
    --global_dec_epochs 30 \
    --K 10
    
python vae_experiments/validation.py
