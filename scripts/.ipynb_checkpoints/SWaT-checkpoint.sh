export CUDA_VISIBLE_DEVICES=0

# python main.py --experiment_name SWaT --gen_ae_epochs 40 --dataset SWaT --window_size 128 --test_batch_size 32 --weight_train 0.1 --weight_test 0.9 --modify 1.5 --persent 90 | tee main_SWaT_result.txt


python main.py --experiment_name SWaT --dataset SWaT --weight_train 0.2 --weight_test 0.8 --modify 1.7 --persent 92 | tee main_SWaT_result.txt --limit_previous 1.1