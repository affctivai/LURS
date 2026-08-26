import os
import numpy as np
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import pickle
import json
import time
import copy
from tqdm import tqdm
from transformers import BartTokenizer, BartForConditionalGeneration
from dataclasses import asdict

from module.ldm.autoencoder import AutoencoderKL
from module.ldm.openaimodel import UNetModel
from model import *
from utils import *
from module.compare_model.EEGNet import EEGNet
from module.compare_model.DeepConvNet import DeepConvNet
from module.compare_model.MLP import MLP
from evaluation import eval_model
from train import train_model
from torch.optim.swa_utils import AveragedModel
import distutils.version  # torch 1.9's tensorboard shim needs this submodule pre-loaded (newer setuptools no longer auto-imports it)
from torch.utils.tensorboard import SummaryWriter

# from data.data import ZuCoNpyDataset
# ---- preprocessed ----
from data.data import ZuCo_dataset, get_input_sample, load_preprocessed


# import os, tempfile
# TMPDIR = "/home/pymp_tmp"
# os.environ["TMPDIR"] = TMPDIR
# os.environ["TEMP"] = TMPDIR
# os.environ["TMP"] = TMPDIR
# tempfile.tempdir = TMPDIR

def train(dataloaders, device, model, optimizer, scheduler, num_epochs=25, checkpoint_path_best = './checkpoints/decoding/best/temp_decoding.pt', checkpoint_path_last = './checkpoints/decoding/last/temp_decoding.pt', patience = 5, writer=None):
    # modified from: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 100000000000
    early_stopper = EarlyStopping(patience=patience)
    # ema_model = AveragedModel(
    #     model,
    #     avg_fn=lambda avg, param, n: 0.999 * avg + 0.001 * param
    # )

    for epoch in range(1, num_epochs+1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'dev']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()

            running_loss = 0.0

            
            # Iterate over data.
            # for input_embeddings, seq_len, input_masks, input_mask_invert, target_ids, target_mask, sent_level_EEG, rawEEG in tqdm(dataloaders[phase]):
            # for input_embeddings, seq_len, input_masks, input_mask_invert, target_ids, target_mask, sentiment_label, sent_level_EEG in tqdm(dataloaders[phase]):
            for input_embeddings, seq_len, input_masks, input_mask_invert, target_ids, target_mask, sentiment_labels, sent_level_EEG, mask_indices, same_target, word_ids in tqdm(dataloaders[phase]):

                # load in batch
                input_embeddings_batch = input_embeddings.to(device).float()
                input_masks_batch = input_masks.to(device)
                input_mask_invert_batch = input_mask_invert.to(device)
                target_ids_batch = target_ids.to(device)
                """replace padding ids in target_ids with -100"""
                context = target_ids_batch.clone()
                target_ids_batch[target_ids_batch == tokenizer.pad_token_id] = -100

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
    	        # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # warm_up_rate = get_linear_annealing_weight(epoch, 10, 30)  # Update the annealing weight
                    out, loss = model(input_embeddings_batch, context, target_ids_batch, input_masks_batch)
                    """calculate loss"""

                    if phase == 'train':
                        loss.sum().backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                        optimizer.step()
                        # ema_model.update_parameters(model)
                # statistics
                
                running_loss += loss.sum().item() * input_embeddings_batch.size()[0] # batch loss
                
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]

            print('{} Loss: {:.4f}'.format(phase, epoch_loss))

            if writer is not None:
                writer.add_scalar(f'Loss/{phase}', epoch_loss, epoch)
                if phase == 'train':
                    writer.add_scalar('lr', scheduler.get_last_lr()[0], epoch)

            # deep copy the model
            if phase == 'dev':
                if  epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    # best_model_wts = copy.deepcopy(ema_model.module.state_dict())
                    '''save checkpoint'''
                    torch.save(best_model_wts, checkpoint_path_best)
                    print(f'update best on dev checkpoint: {checkpoint_path_best}')
                # ④ Early Stopping 조건 확인
                if early_stopper(epoch_loss):
                    print(f'\nEarly stopping triggered (patience={patience}) at epoch {epoch}')
                    torch.save(model.state_dict(), checkpoint_path_last)
                    model.load_state_dict(best_model_wts)   # 최적 가중치 복원
                    # torch.save(model.state_dict(), checkpoint_path_best)

                    time_elapsed = time.time() - since
                    print(f'Training stopped in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s')
                    print(f'update last checkpoint: {checkpoint_path_last}')
                    return model
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    torch.save(model.state_dict(), checkpoint_path_last)
    model.load_state_dict(best_model_wts)
    # torch.save(model.state_dict(), checkpoint_path_best)
    print(f'update last checkpoint: {checkpoint_path_last}')
    return model


if __name__ == '__main__':
    args = get_config('pretrain_autoencoder')

    ''' config param'''

    is_con = args['is_con']
    is_geo = args['is_geo']
    is_kl = args['is_kl']

    noise_scheduler = args['noise_scheduler']
    time_step = args['time_step']
    
    num_epoch = args['num_epoch']
    lr = args['learning_rate']
    
    batch_size = args['batch_size']
    model_name = args['model_name'] # LURS, w/o_diffusion, DConv, EEGNet ...
    models = ['pretrain', 'wo_diffusion_pretrain', 'DConv_pretrain', 'EEGNet_pretrain', 'MLP_pretrain']
    if model_name not in models:
        raise ValueError(f'model_name should be chosen from {models}')
    
    # task_name = 'task1'
    # task_name = 'task1_task2'
    # task_name = 'task1_task2_task3'
    # task_name = 'task1_task2_taskNRv2'
    task_name = args['task_name']
    save_path = args['save_path']
    # train_input = args['train_input'] # word-level or rawEEG
    # print("train_input is:", train_input)   
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    experiment_name = f"{noise_scheduler}_{time_step}_{model_name}" if args['save_name'] is None else args['save_name']

    save_name = f"{args['subjects']}_{experiment_name}-module" if args['save_name'] is None else args['save_name']
 
    if args['is_con']:
        save_name = 'con_' + save_name
    if args['is_geo']:
        save_name = 'geo_' + save_name
    if args['is_kl']:
        save_name = 'kl_' + save_name

    save_path_best = os.path.join(save_path, 'best')
    if not os.path.exists(save_path_best):
        os.makedirs(save_path_best)

    output_checkpoint_name_best = os.path.join(save_path_best, f'{save_name}.pt')

    save_path_last = os.path.join(save_path, 'last')
    if not os.path.exists(save_path_last):
        os.makedirs(save_path_last)

    output_checkpoint_name_last = os.path.join(save_path_last, f'{save_name}.pt')

    # subject_choice = 'ALL
    # subject_choice = args['subjects']
    # print(f'![Debug]using {subject_choice}')
    # eeg_type_choice = 'GD
    eeg_type_choice = args['eeg_type']
    print(f'[INFO]eeg type {eeg_type_choice}')
    # bands_choice = ['_t1'] 
    # bands_choice = ['_t1','_t2','_a1','_a2','_b1','_b2','_g1','_g2'] 
    bands_choice = args['eeg_bands']
    print(f'[INFO]using bands {bands_choice}')


    
    ''' set random seeds '''
    # https://pytorch.org/docs/stable/notes/randomness.html
    seed_val = args['seed']
    set_seed(seed_val)

    ''' set up device '''
    # use cuda
    if torch.cuda.is_available():  
        # dev = "cuda:3" 
        dev = args['cuda'] 
    else:  
        dev = "cpu"
    # CUDA_VISIBLE_DEVICES=0,1,2,3  
    device = torch.device(dev)
    print(f'[INFO]using device {dev}')
    print()


    print()


    
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')



    pretrained_LM = BartForConditionalGeneration.from_pretrained('facebook/bart-large').to(device)

    vocab_size = tokenizer.vocab_size
    

    
    if model_name == 'EEGNet_pretrain':
        denoising_module_statte_dict = {'n_classes': 2,
            'input_shape': (1, 1, 512, 56),
            'F1': 16,                # 첫 번째 Temporal Conv 필터 수
            'D': 2,                  # Spatial Conv 깊이 승수
            'F2': 'auto',            # F1 * D = 32
            'T1': 32,                # 첫 번째 Temporal Conv 커널 크기 (sequence_length보다 작아야 함)
            'T2': 16,                # 두 번째 Temporal Conv 커널 크기
            'P1': 4,                 # 첫 번째 Pooling 크기
            'P2': 8,                 # 두 번째 Pooling 크기
            'drop_out': 0.5,         # 드롭아웃 비율
            'pool_mode': 'mean',     # 풀링 방식
            'weight_init_method': 'xavier_uniform'}
        denoising_module = EEGNet(
            **denoising_module_statte_dict
        ).to(device)
        
    elif model_name == 'DConv_pretrain':
        denoising_module_statte_dict = {
            'input_shape': [32, 1, 512, 56], # [Batch, In-channels, Electrodes, Samples]
            'first_conv_length': 5,
            # 'block_out_channels': [1024, 256, 128, 256, 512],
            'block_out_channels': [32, 64, 128, 256, 512],
            'pool_size': 3,
            }
        denoising_module = DeepConvNet(
            **denoising_module_statte_dict
        ).to(device)

    elif model_name == 'MLP_pretrain':
        denoising_module_statte_dict = {
            'input_shape': (1, 1, 512, 56),
            'hidden_dim': 6144,
            'num_layers': 10,
            'drop_out': 0.5,
            'weight_init_method': 'xavier_uniform',
            }
        denoising_module = MLP(
            **denoising_module_statte_dict
        ).to(device)
    else:
        denoising_module_statte_dict = {
            'image_size': 512, # 256
            # 'image_size': 256,
            'dims' : 1, 
            'in_channels': 56, 
            'out_channels': 56, 
            'model_channels': 320, # 224
            # 'model_channels': 224,
            'attention_resolutions': [4, 2, 1], 
            # 'attention_resolutions': [2, 1], 
            'num_res_blocks': 2, 
            'channel_mult': [1, 2, 4, 4],
            # 'channel_mult': [1, 2],
            'num_heads': 8, 
            'use_spatial_transformer': True, 
            'transformer_depth': 1, 
            'context_dim': 512, 
            'use_checkpoint': False, 
            'legacy': False,
            'dropout': 0.0
            }
        denoising_module = UNetModel(
            **denoising_module_statte_dict
            ).to(device)
        
    vocab_size = tokenizer.vocab_size


    autoencoder_state_dict = {
        'embed_dim': 56,
        'monitor': "val/rec_loss",
        'ddconfig': {
            "double_z": True,
            "z_channels": 58, 
            "resolution": 256, # 128
            # "resolution": 128,
            "in_channels": 56,
            "out_ch": 56,
            "ch": 128, # 64
            # "ch": 64,
            "ch_mult": [1, 2, 4, 4],# [1, 2, 4],
            # "ch_mult": [1, 2, 4],
            "num_res_blocks": 2,
            "attn_resolutions": [],
            "dropout": 0.0
        },
        'lossconfig': {
            "target": "torch.nn.Identity"
        },
    }
    autoencoder = AutoencoderKL(
        **autoencoder_state_dict
    ).to(device)
    
    if model_name == 'wo_diffusion_pretrain' or model_name in ('EEGNet_pretrain', 'DConv_pretrain', 'MLP_pretrain'):
        is_diffusion = False
    else :
        is_diffusion = True

    # is_compare_model = model_name in ('EEGNet_pretrain', 'DConv_pretrain', 'MLP_pretrain')

    pretrain_opts = BuildOptions(model_name="pretrain", is_con=is_con, is_geo=is_geo, is_diffusion=is_diffusion,
                                 is_kl=is_kl, autoencoder=autoencoder, denoising_module=denoising_module, pretrained_LM=pretrained_LM, device = device, noise_scheduler = noise_scheduler, time_step = time_step)


    """save config"""
    cfg_dir = './save_model/config'

    if not os.path.exists(cfg_dir):
        os.makedirs(cfg_dir)
    

    pretrain_model = make_model(pretrain_opts).to(device)
    state_dict = {
        "args":args,
        "module_state_dict": {"denoising_module": denoising_module_statte_dict, "autoencoder": autoencoder_state_dict},
        "loss_weights": {'alpha': pretrain_model.alpha if is_kl else None, 'beta': pretrain_model.beta if is_con else None, 'theta': pretrain_model.theta if is_geo else None, 'is_diffusion': is_diffusion},
    }
    with open(os.path.join(cfg_dir,f'{save_name}.json'), 'w') as out_config:
        json.dump(state_dict, out_config, indent = 4)

    # optimizer = optim.SGD(pretrain_model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5)
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    optimizer = optim.AdamW(pretrain_model.parameters(), lr=lr, weight_decay=0.01)
    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epoch)

    test_input = args['test_input']
    dataset_setting = args['dataset_setting']
    subject_choice = args['subjects']

    whole_dataset_dicts = []
    if 'task1' in task_name:
        whole_dataset_dicts.append(load_preprocessed("task1-SR"))
    if 'task2' in task_name:
        whole_dataset_dicts.append(load_preprocessed("task2-NR"))
    if 'task3' in task_name:
        whole_dataset_dicts.append(load_preprocessed("task3-TSR"))
    if 'taskNRv2' in task_name:
        whole_dataset_dicts.append(load_preprocessed("task2-NR-2.0"))

    train_set = ZuCo_dataset(whole_dataset_dicts, phase='train', tokenizer=tokenizer, eeg_type=eeg_type_choice, bands=bands_choice, setting=dataset_setting)
    dev_set   = ZuCo_dataset(whole_dataset_dicts, phase='dev',   tokenizer=tokenizer, eeg_type=eeg_type_choice, bands=bands_choice, setting=dataset_setting)
    test_set  = ZuCo_dataset(whole_dataset_dicts, phase='test',  tokenizer=tokenizer, eeg_type=eeg_type_choice, bands=bands_choice, setting=dataset_setting)



    # test dataset
    # test_set = ZuCo_dataset(whole_dataset_dicts, 'test', tokenizer, eeg_type = eeg_type_choice, bands = bands_choice, setting = dataset_setting)

    
    dataset_sizes = {'train': len(train_set), 'dev': len(dev_set)}
    print('[INFO]train_set size: ', len(train_set))
    print('[INFO]dev_set size: ', len(dev_set))
    # print('[INFO]test_set size: ', len(test_set))
    
    # train dataloader
    train_dataloader = DataLoader(train_set, batch_size = batch_size, shuffle=True, num_workers=4)
    # dev dataloader
    val_dataloader = DataLoader(dev_set, batch_size = batch_size, shuffle=False, num_workers=4)
    # test_dataloader = DataLoader(test_set, batch_size = batch_size, shuffle=False, num_workers=4)
    # dataloaders
    # dataloaders = {'train':train_dataloader, 'dev':val_dataloader, 'test':test_dataloader}
    dataloaders = {'train':train_dataloader, 'dev':val_dataloader}

    ''' set up tensorboard '''
    tb_log_dir = os.path.join(save_path, 'tensorboard', save_name)
    writer = SummaryWriter(log_dir=tb_log_dir)
    print(f'[Info] tensorboard logging to {tb_log_dir}')

    ''' training loop '''
    since = time.time()
    model = train(dataloaders, device, pretrain_model, optimizer, exp_lr_scheduler, num_epochs=num_epoch, checkpoint_path_best = output_checkpoint_name_best, checkpoint_path_last = output_checkpoint_name_last, writer=writer)
    total_time = time.time() - since
    print(f'Training complete in {total_time//60:.0f}m {total_time%60:.0f}s')
    writer.close()
