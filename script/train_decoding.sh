python3 pretrain.py \
    --task_name task1_task2_taskNRv2 \
    --num_epoch 100 \
    --train_input EEG \
    -lr 0.00002 \
    -b 32 \
    --cuda cuda:0 \
    -s ./checkpoints/autoencoder

python3 train_diffusion.py --model_name BrainTranslator \
    --task_name task1_task2_taskNRv2 \
    --two_step \
    --num_epoch_step1 100 \
    --num_epoch_step2 50 \
    --train_input EEG \
    -lr1 0.00002 \
    -lr2 0.00002 \
    -b 32 \
    --cuda cuda:0 \
    -s ./checkpoints \
    --eeg_encoder checkpoints/encoder/best/pretrained_module_task1_task2_taskNRv2_32_100_2e-05.pt