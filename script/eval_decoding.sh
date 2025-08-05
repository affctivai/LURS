python3 eval_decoding.py \
    --checkpoint_path checkpoints/best/ldm_task1_task2_taskNRv2_2steptraining_b32_50_20_2e-05_2e-05.pt \
    --config_path config/decoding/ldm_task1_task2_taskNRv2_2steptraining_b32_50_20_2e-05_2e-05.json \
    --test_input EEG \
    --train_input EEG \
    -cuda cuda:0

