import os
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

from data.data import ZuCoNpyDataset
from model import *
from utils import *
from module.ldm.autoencoder import AutoencoderKL
from module.ldm.openaimodel import UNetModel
from evaluation import eval_model
from dataclasses import asdict
from module.compare_model.EEGNet import EEGNet
from module.compare_model.DeepConvNet import DeepConvNet

from torch.optim.swa_utils import AveragedModel

# import os, tempfile
# TMPDIR = "/home/pymp_tmp"
# os.environ["TMPDIR"] = TMPDIR
# os.environ["TEMP"] = TMPDIR
# os.environ["TMP"] = TMPDIR
# tempfile.tempdir = TMPDIR



def train_model(dataloaders, device, model, optimizer, scheduler, num_epochs=25, checkpoint_path_best = './checkpoints/decoding/best/temp_decoding.pt', checkpoint_path_last = './checkpoints/decoding/last/temp_decoding.pt', stage = None):
    # modified from: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')

    # contrastive with reg 할 때 15에서 20으로 바꿨음
    early_stopper = EarlyStopping(patience=20) 
    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0

            
            # Iterate over data.
            # for input_embeddings, seq_len, input_masks, input_mask_invert, target_ids, target_mask, sent_level_EEG, rawEEG in tqdm(dataloaders[phase]):
            for input_embeddings, seq_len, input_masks, input_mask_invert, target_ids, target_mask, sentiment_labels, sent_level_EEG in tqdm(dataloaders[phase]):

                # load in batch
                input_embeddings_batch = input_embeddings.to(device).float()
                input_masks_batch = input_masks.to(device)
                input_mask_invert_batch = input_mask_invert.to(device)
                target_ids_batch = target_ids.to(device)
                """replace padding ids in target_ids with -100"""
                context = target_ids_batch.clone()

                target_ids_batch[target_ids_batch == tokenizer.pad_token_id] = -100

                with torch.set_grad_enabled(phase == 'train'):
                    optimizer.zero_grad()
                    out, loss = model(input_embeddings_batch, input_masks_batch, target_ids_batch, context)

                    """calculate loss"""
                 
                    # backward + optimize only if in training phase
                    loss_sum = loss.sum()
                    if phase == 'train':
                        # with torch.autograd.detect_anomaly():
                        loss_sum.backward()
                        optimizer.step()
                        # ema_model.update_parameters(model)

                # statistics
                running_loss += loss_sum.detach().item() * input_embeddings_batch.size()[0] # batch loss
                

            if phase == 'train':
                scheduler.step()
                

            epoch_loss = running_loss / dataset_sizes[phase]

            print('{} Loss: {:.4f}'.format(phase, epoch_loss))
            
            # deep copy the model
            if phase == 'dev':
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    '''save checkpoint'''
                    torch.save(best_model_wts, checkpoint_path_best)
                    print(f'update best on dev checkpoint: {checkpoint_path_best}')
                # ④ Early Stopping 조건 확인
                if early_stopper(epoch_loss):
                    print(f'\nEarly stopping triggered (patience={20}) at epoch {epoch}')
                    torch.save(model.state_dict(), checkpoint_path_last)
                    model.load_state_dict(best_model_wts)   # 최적 가중치 복원
                    torch.save(model.state_dict(), checkpoint_path_best)
                    
                    time_elapsed = time.time() - since
                    print(f'Training stopped in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s')
                    print(f'update last checkpoint: {checkpoint_path_best}')
                    return model
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    torch.save(model.state_dict(), checkpoint_path_last)
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), checkpoint_path_best)
    
    print(f'update last checkpoint: {checkpoint_path_last}')
    return model


if __name__ == '__main__':

    args = get_config('train_diffusion')

    ''' config param'''
    num_epochs_step1 = args['num_epoch_step1']
    num_epochs_step2 = args['num_epoch_step2']
    step1_lr = args['learning_rate_step1']
    step2_lr = args['learning_rate_step2']
    batch_size = args['batch_size']
    model_name = args['model_name']

    models = ['DELTA', 'wo_diffusion', 'DConv', 'EEGNet', 'wo_pretrained','main_diffusion', 'full_diffusion']
    if model_name not in models:
        raise ValueError(f'model_name should be chosen from {models}')

    save_path = args['save_path']
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print(f'[INFO]using model: {model_name}')

    # pretrained config를 직접 로드 (경로에 con_/geo_/kl_ 접두사 포함해서 전달)
    module_config_path = args['config_path']
    with open(module_config_path, 'r') as f:
        configs = json.load(f)

    pretrained_args = configs['args']

    # 데이터/모델 관련 세팅은 모두 pretrained config에서 읽음
    task_name       = pretrained_args['task_name']
    eeg_type_choice = pretrained_args['eeg_type']
    bands_choice    = pretrained_args['eeg_bands']
    dataset_setting = pretrained_args['dataset_setting']
    subject_choice  = pretrained_args['subjects']
    test_input      = pretrained_args['test_input']
    noise_scheduler = pretrained_args['noise_scheduler']
    time_step       = pretrained_args['time_step']
    is_con          = pretrained_args['is_con']
    is_geo          = pretrained_args['is_geo']
    is_kl           = pretrained_args['is_kl']
    seed_val        = pretrained_args['seed']

    pretrained_model_name = pretrained_args['model_name']

    print(f'[INFO]eeg type {eeg_type_choice}')
    print(f'[INFO]using bands {bands_choice}')
    print(f'[INFO]task: {task_name} | subject: {subject_choice} | test_input: {test_input}')

    base_name = os.path.basename(module_config_path)
    save_name = f"{base_name.split('_pretrain')[0]}_{model_name}" if args['save_name'] is None else args['save_name']

    save_path_best = os.path.join(save_path, 'best')
    if not os.path.exists(save_path_best):
        os.makedirs(save_path_best)
    output_checkpoint_name_best = os.path.join(save_path_best, f'{save_name}.pt')

    save_path_last = os.path.join(save_path, 'last')
    if not os.path.exists(save_path_last):
        os.makedirs(save_path_last)
    output_checkpoint_name_last = os.path.join(save_path_last, f'{save_name}.pt')

    set_seed(seed_val)
    ''' set random seeds '''
    # https://pytorch.org/docs/stable/notes/randomness.html


    ''' set up device '''
    # use cuda
    if torch.cuda.is_available():  
        # dev = "cuda:3" 
        dev = args['cuda'] 
    else:  
        dev = "cpu"
    # CUDA_VISIBLE_DEVICES=0,1
    device = torch.device(dev)
    print(f'[INFO]using device {dev}')
    print()

    ''' set up dataloader '''
    # whole_dataset_dicts = []
    # if 'task1' in task_name:
    #     dataset_path_task1 = './dataset/ZuCo/task1-SR/pickle/task1-SR-dataset.pickle'
    #     with open(dataset_path_task1, 'rb') as handle:
    #         whole_dataset_dicts.append(pickle.load(handle))
    #         # zuco_v1.append(pickle.load(handle))
    # if 'task2' in task_name:
    #     dataset_path_task2 = './dataset/ZuCo/task2-NR/pickle/task2-NR-dataset.pickle'
    #     with open(dataset_path_task2, 'rb') as handle:
    #         whole_dataset_dicts.append(pickle.load(handle))
    #         # zuco_v1.append(pickle.load(handle))
    # if 'task3' in task_name:
    #     dataset_path_task3 = './dataset/ZuCo/task3-TSR/pickle/task3-TSR-dataset.pickle'
    #     with open(dataset_path_task3, 'rb') as handle:
    #         whole_dataset_dicts.append(pickle.load(handle))
    #         # zuco_v1.append(pickle.load(handle))
    # if 'taskNRv2' in task_name:
    #     dataset_path_taskNRv2 = './dataset/ZuCo/task2-NR-2.0/pickle/task2-NR-2.0-dataset.pickle'
    #     with open(dataset_path_taskNRv2, 'rb') as handle:
    #         whole_dataset_dicts.append(pickle.load(handle))
    #         # zuco_v2.append(pickle.load(handle))

    print()



    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')


    # # train dataset
    # train_set = ZuCo_dataset(whole_dataset_dicts, 'train', tokenizer, subject = subject_choice, eeg_type = eeg_type_choice, bands = bands_choice, setting = dataset_setting, test_input=train_input)
    # # dev dataset
    # dev_set = ZuCo_dataset(whole_dataset_dicts, 'dev', tokenizer, subject = subject_choice, eeg_type = eeg_type_choice, bands = bands_choice, setting = dataset_setting, test_input=train_input)
    # # test dataset
    # test_set = ZuCo_dataset(whole_dataset_dicts, 'test', tokenizer, subject = subject_choice, eeg_type = eeg_type_choice, bands = bands_choice, setting = dataset_setting, test_input=train_input)
    

    # del whole_dataset_dicts
    # import gc; gc.collect()

    ''' set up model '''
    pretrained_LM = BartForConditionalGeneration.from_pretrained('facebook/bart-large').to(device)
    vocab_size = tokenizer.vocab_size

    # load denoising module and autoencoder state dict
    if model_name == 'w/o_pretrained':
        with open('save_model/config/Default_setting.json') as f:
            configs = json.load(f)
        denoising_module_state_dict = configs["UNet"]
        autoencoder_state_dict = configs["autoencoder"]
    else:
        denoising_module_state_dict = configs["module_state_dict"]["denoising_module"]
        autoencoder_state_dict = configs["module_state_dict"]["autoencoder"]



    
    if model_name == 'EEGNet':
        denoising_module = EEGNet(
            **denoising_module_state_dict
        ).to(device)
        
    elif model_name == 'DConv':
        denoising_module = DeepConvNet(
            **denoising_module_state_dict
        ).to(device)
    else: 
        denoising_module = UNetModel(
            **denoising_module_state_dict
            ).to(device)
        
    vocab_size = tokenizer.vocab_size

    autoencoder = AutoencoderKL(
        **autoencoder_state_dict
    ).to(device)
    
    # if model_name == 'wo_diffusion' or model_name == 'EEGNet':
    #     is_diffusion = [False, False]
    # elif model_name == 'DConv':
    #     is_diffusion = [False, True]
    # else :
    #     is_diffusion = [True, False]

    if model_name == 'wo_diffusion':
        is_diffusion = False
    else :
        is_diffusion = True


    # if pretrained_model_name == "pretrain":
    #     pretrained_model_name = pretrained_model_name
    # else:
    #     pretrained_model_name = pretrained_model_name+"_pretrain"
    pretrain_opts = BuildOptions(model_name=pretrained_model_name, is_con=is_con, is_geo=is_geo,is_kl=is_kl, is_diffusion=is_diffusion,
                                 autoencoder=autoencoder, denoising_module=denoising_module, pretrained_LM=pretrained_LM, device = device)
    


    pt_name = base_name.split('.')[0]
    pt_path = f'./save_model/checkpoints/autoencoder/best/{pt_name}.pt'
    pt_module = make_model(pretrain_opts)
    print(f'[Info] loading {pt_name}')

    if model_name != 'wo_pretrained':
        state_dict = torch.load(pt_path, map_location=device)
        pt_module.load_state_dict(state_dict)
    

    #     models = ['DELTA', 'wo_diffusion', 'DConv', 'EEGNet', 'wo_pretrained','main_diffusion', 'full_diffusion']

    if pretrained_model_name != 'pretrain' and model_name == 'full_diffusion':
        Exception('full_diffusion should use pretrainedmodule as pretrained_model_name')
    
    if pretrained_model_name != 'wo_diffusion_pretrain' and model_name == 'main_diffusion':
        Exception('main_diffusion should use wo_diffusion_pretrain as pretrained_model_name')

    # if model_name == 'DConv':
    #     is_diffusion_delta = [False, True]
    # elif model_name == 'main_diffusion' or model_name == 'full_diffusion':
    #     is_diffusion_delta = [True, False]
    # else :
    #     is_diffusion_delta = [False, False]

    opts = BuildOptions(model_name=model_name, pretrained_module=pt_module, is_diffusion=is_diffusion, noise_scheduler=noise_scheduler, time_step=time_step, device=device)

    model = make_model(opts)

    
    ''' training loop '''
    ######################################################
    '''step one trainig: freeze most of BART params'''
    ######################################################

    # closely follow BART paper
    for name, param in model.named_parameters():
        if param.requires_grad and 'pretrained' in name:
            # 이것도 멈추는거 한번 보자
            # pretrian에서 BART embedding시키니까 한번 그쪽도 손봐야 되기도 싶고
            if ('shared' in name) or ('embed_positions' in name) or ('encoder.layers.0' in name):
                continue
            else:
                param.requires_grad = False

# subjects=( 'ZAB' 'ZDM' 'ZGW' 'ZJM' 'ZJN' 'ZJS' 'ZKB' 'ZKH' 'ZKW' 'ZMG' 'ZPH' )
    


    ''' set up optimizer and scheduler'''
    print('=== start Step1 training ... ===')
    optimizer_step1 = optim.SGD(model.parameters(), lr=step1_lr, momentum=0.9)
    exp_lr_scheduler_step1 = lr_scheduler.StepLR(optimizer_step1, step_size=20, gamma=0.1)

    since = time.time()

    state_dict = {
        "args": args,
        "pretrained_args": pretrained_args,
        "module_state_dict": {"denoising_module": denoising_module_state_dict, "autoencoder": autoencoder_state_dict},
        "pretrained_module_loss_weights": {'alpha': pt_module.alpha if pt_module else None, 'beta': pt_module.beta if pt_module else None, 'theta': pt_module.theta if pt_module else None, 'is_diffusion': is_diffusion},
    }

    npy_root = "/mnt/Members/speech/jd_data/zuco_decoding_npy/"
    task_npy_dirs = []
    if 'task1' in task_name:
        task_npy_dirs.append(os.path.join(npy_root, "task1-SR"))
    if 'task2' in task_name:
        task_npy_dirs.append(os.path.join(npy_root, "task2-NR"))
    if 'task3' in task_name:
        task_npy_dirs.append(os.path.join(npy_root, "task3-TSR"))
    if 'taskNRv2' in task_name:
        task_npy_dirs.append(os.path.join(npy_root, "task2-NR-2.0"))

    train_set = ZuCoNpyDataset(task_npy_dirs, phase='train', tokenizer=tokenizer, subject=subject_choice, eeg_type=eeg_type_choice, bands=bands_choice, setting=dataset_setting, test_input=test_input)
    dev_set   = ZuCoNpyDataset(task_npy_dirs, phase='dev',   tokenizer=tokenizer, subject=subject_choice, eeg_type=eeg_type_choice, bands=bands_choice, setting=dataset_setting, test_input=test_input)
    test_set  = ZuCoNpyDataset(task_npy_dirs, phase='test',  tokenizer=tokenizer, subject=subject_choice, eeg_type=eeg_type_choice, bands=bands_choice, setting=dataset_setting, test_input=test_input)
    # train_set = ZuCo_dataset(whole_dataset_dicts, 'train', tokenizer, eeg_type = eeg_type_choice, bands = bands_choice, setting = dataset_setting)
    # # dev dataset
    # dev_set = ZuCo_dataset(whole_dataset_dicts, 'dev', tokenizer, eeg_type = eeg_type_choice, bands = bands_choice, setting = dataset_setting)
    # # test dataset
    # test_set = ZuCo_dataset(whole_dataset_dicts, 'test', tokenizer, eeg_type = eeg_type_choice, bands = bands_choice, setting = dataset_setting)

    
    dataset_sizes = {'train': len(train_set), 'dev': len(dev_set)}
    print('[INFO]train_set size: ', len(train_set))
    print('[INFO]dev_set size: ', len(dev_set))
    print('[INFO]test_set size: ', len(test_set))
    
    # train dataloader
    train_dataloader = DataLoader(train_set, batch_size = batch_size, shuffle=True, num_workers=4)
    # dev dataloader
    val_dataloader = DataLoader(dev_set, batch_size = 1, shuffle=False, num_workers=4)
    test_dataloader = DataLoader(test_set, batch_size = 1, shuffle=False, num_workers=4)
    # dataloaders
    dataloaders = {'train':train_dataloader, 'dev':val_dataloader, 'test':test_dataloader}


    model = train_model(dataloaders, device, model, optimizer_step1, exp_lr_scheduler_step1, num_epochs=num_epochs_step1, 
                        checkpoint_path_best = output_checkpoint_name_best, checkpoint_path_last = output_checkpoint_name_last, stage = 'first_stage')
    first_step_time = time.time() - since
    print(f'Step 1 Training complete in {first_step_time//60:.0f}m {first_step_time%60:.0f}s')

    """save config"""
    cfg_dir = './save_model/config'

    if not os.path.exists(cfg_dir):
        os.makedirs(cfg_dir)


    with open(os.path.join(cfg_dir,f'{save_name}.json'), 'w') as out_config:
        json.dump(state_dict, out_config, indent = 4)
    ######################################################
    '''step two trainig: update whole model for a few iterations'''
    ######################################################
    for name, param in model.named_parameters():
        param.requires_grad = True

    ''' set up optimizer and scheduler'''
    # optimizer_step2 = optim.Adam(model.parameters(), lr=step2_lr)# (B, 56, vocab_size)
    optimizer_step2 = optim.SGD(model.parameters(), lr=step2_lr, momentum=0.9)
    exp_lr_scheduler_step2 = lr_scheduler.StepLR(optimizer_step2, step_size=30, gamma=0.1)

    print()
    print('=== start Step2 training ... ===')
    # print training layers
    # show_require_grad_layers(model)
    
    since = time.time()
    '''main loop'''
    model = train_model(dataloaders, device, model, optimizer_step2, exp_lr_scheduler_step2, num_epochs=num_epochs_step2, 
                                    checkpoint_path_best = output_checkpoint_name_best, checkpoint_path_last = output_checkpoint_name_last, stage = 'second_stage')

    second_step_time = time.time() - since
    print(f'Step 2 Training complete in {second_step_time//60:.0f}m {second_step_time%60:.0f}s')
    
    if not os.path.exists('./results'):
        os.makedirs('./results')
    if not os.path.exists('./score_results'):
        os.makedirs('./score_results')
    eval_model(dataloaders, device, tokenizer, model, output_all_results_path = f'./results/best_{save_name}.txt' , score_results=f'./score_results/best_{save_name}.txt')
