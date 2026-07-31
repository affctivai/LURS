import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_config(case):
    if case == 'pretrain_autoencoder': 
        # args config for training EEG-To-Text decoder
        parser = argparse.ArgumentParser(description='Specify config args for training EEG-To-Text decoder')

        parser.add_argument('-t', '--task_name', help='choose from {task1,task1_task2, task1_task2_task3,task1_task2_taskNRv2}', default = "task1", required=True)
        parser.add_argument('-s', '--save_path', help='checkpoint save path', default = './checkpoints/decoding', required=True)
        parser.add_argument('-eeg', '--eeg_type', help='choose from {GD, FFD, TRT}', default = 'GD', required=False)
        parser.add_argument('-band', '--eeg_bands', nargs='+', help='specify freqency bands', default = ['_t1','_t2','_a1','_a2','_b1','_b2','_g1','_g2'] , required=False)        
        
        parser.add_argument('-epoch', '--num_epoch', type = int, help='num_epoch_step', default = 20, required=True)
        parser.add_argument('-lr', '--learning_rate', type = float, help='learning_rate_step1', default = 0.00005, required=True)
        parser.add_argument('-b', '--batch_size', type = int, help='batch_size', default = 32, required=True)
        
        parser.add_argument('-cuda', '--cuda', help='specify cuda device name, e.g. cuda:0, cuda:1, etc', default = 'cuda:0')
        parser.add_argument('--save_name', help='save name' ,required=False)

        parser.add_argument('-m', '--model_name', help='choose from pretrain, wo_diffusion_pretrain, DConv_pretrain, EEGNet_pretrain, MLP_pretrain', default = "DELTA", required=True)
        parser.add_argument('--seed', type=int, help='set seed', default=312, required=False)

        parser.add_argument('-n', '--noise_scheduler', help='select noise scheduler', default = 'linear' , required=False)
        parser.add_argument('-ts', '--time_step', type = int, help='selecttime step', default = 1000 , required=False)

        parser.add_argument('-con', '--is_con', help='use Contrastive loss?', action='store_true')
        parser.add_argument('-geo', '--is_geo', help='use Geometric loss?', action='store_true')
        parser.add_argument('-kl', '--is_kl', help='use kl loss?', action='store_true')
        parser.add_argument('-subj', '--subjects', help='leave-one-subject-out target; omit for ALL', default='ALL', required=False)

        parser.add_argument('-inputs', '--test_input', help = 'choose dataset setting from eeg or noise', default = 'eeg', required=False)
        parser.add_argument('-setting', '--dataset_setting', help = 'choose dataset setting from unique_sent, unique_subject', default = 'unique_sent', required=False)

        args = vars(parser.parse_args())

    elif case == 'train_diffusion':
        # args config for training EEG-To-Text decoder
        # 데이터/모델 세팅(task, eeg, subject, test_input 등)은 pretrained config에서 자동으로 읽음
        parser = argparse.ArgumentParser(description='Specify config args for training EEG-To-Text decoder')

        parser.add_argument('-conf', '--config_path', help='path to pretrained model config json (full path, e.g. save_model/config/con_unique_sent_linear_1000_pretrain-module.json)', required=True)
        parser.add_argument('-m', '--model_name', help='choose from DELTA, wo_diffusion, DConv, EEGNet, MLP, wo_pretrained, main_diffusion, full_diffusion', default='DELTA', required=True)
        parser.add_argument('-s', '--save_path', help='checkpoint save path', default='./save_model/checkpoints/DELTA', required=True)

        parser.add_argument('-ne1', '--num_epoch_step1', type=int, help='num_epoch_step1', default=20, required=True)
        parser.add_argument('-ne2', '--num_epoch_step2', type=int, help='num_epoch_step2', default=30, required=True)
        parser.add_argument('-lr1', '--learning_rate_step1', type=float, help='learning_rate_step1', default=0.00002, required=True)
        parser.add_argument('-lr2', '--learning_rate_step2', type=float, help='learning_rate_step2', default=0.000005, required=True)
        parser.add_argument('-b', '--batch_size', type=int, help='batch_size', default=32, required=True)

        parser.add_argument('-cuda', '--cuda', help='cuda device name, e.g. cuda:0', default='cuda:0')
        parser.add_argument('--save_name', help='save name', required=False)

        parser.add_argument('--apply_lora_to_denoising', help='wrap denoising_module attn1/attn2 with LoRA', action='store_true')
        parser.add_argument('--lora_r', type=int, help='LoRA rank', default=8, required=False)
        parser.add_argument('--lora_alpha', type=int, help='LoRA alpha', default=16, required=False)
        parser.add_argument('--lora_dropout', type=float, help='LoRA dropout', default=0.05, required=False)

        args = vars(parser.parse_args())

    elif case == 'eval_decoding':
        # args config for evaluating EEG-To-Text decoder
        parser = argparse.ArgumentParser(description='Specify config args for evaluate EEG-To-Text decoder')
        parser.add_argument('-checkpoint', '--checkpoint_path', help='specify model checkpoint' ,required=True)
        parser.add_argument('-conf', '--config_path', help='specify training model config json' ,required=True)
        parser.add_argument('-cuda', '--cuda', help='specify cuda device name, e.g. cuda:0, cuda:1, etc', default = 'cuda:0')
        parser.add_argument('--seed', type=int, help='set seed', default=20, required=False)
        parser.add_argument('-setting', '--dataset_setting', help = 'choose dataset setting from unique_sent, unique_subject', default = 'unique_sent', required=False)
        parser.add_argument('-subj', '--subjects', help='leave-one-subject-out target; omit for ALL', default='ALL', required=False)
        parser.add_argument('--save_name', help='save name', required=False)
        parser.add_argument('-b', '--batch_size', type=int, help='batch_size', default=1, required=False)

        args = vars(parser.parse_args())


    return args
