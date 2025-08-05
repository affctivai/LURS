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
from data import ZuCo_dataset
from config import get_config
# import wandb
from ldm.autoencoder import AutoencoderKL
from ldm.openaimodel import UNetModel
from model_decoding import pretrainedmodule

class EarlyStopping:
    def __init__(self, patience: int = 10, delta: float = 0.0):
        """
        Args:
            patience (int): 개선이 없다고 판단할 연속 epoch 수
            delta (float): '개선'으로 간주할 최소 손실 감소 폭
        """
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        score = -val_loss                       # 손실이 작을수록 좋은 모델
        if self.best_score is None:
            self.best_score = score
            return False

        if score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
        return self.early_stop
    
def train(dataloaders, device, model, optimizer, scheduler, num_epochs=25, checkpoint_path_best = './checkpoints/decoding/best/temp_decoding.pt', checkpoint_path_last = './checkpoints/decoding/last/temp_decoding.pt'):
    # modified from: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 100000000000
    early_stopper = EarlyStopping(patience=15) 

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
            for input_embeddings, seq_len, input_masks, input_mask_invert, target_ids, target_mask, rawEEG in tqdm(dataloaders[phase]):
                
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
                    out, loss = model(input_embeddings_batch, context, target_ids_batch)
                    """calculate loss"""

                    if phase == 'train':
                        # with torch.autograd.detect_anomaly():
                        loss.sum().backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                        optimizer.step()
                # statistics
                
                running_loss += loss.sum().item() * input_embeddings_batch.size()[0] # batch loss
                
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]

            print('{} Loss: {:.4f}'.format(phase, epoch_loss))
            
            # deep copy the model
            if phase == 'dev':
                if  epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    '''save checkpoint'''
                    torch.save(model.state_dict(), checkpoint_path_best)
                    print(f'update best on dev checkpoint: {checkpoint_path_best}')
                # ④ Early Stopping 조건 확인
                if early_stopper(epoch_loss):
                    print(f'\nEarly stopping triggered (patience={10}) at epoch {epoch}')
                    model.load_state_dict(best_model_wts)   # 최적 가중치 복원
                    time_elapsed = time.time() - since
                    print(f'Training stopped in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s')
                    torch.save(model.state_dict(), checkpoint_path_last)
                    print(f'update last checkpoint: {checkpoint_path_last}')
                    return model
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), checkpoint_path_last)
    print(f'update last checkpoint: {checkpoint_path_last}')
    return model


def show_require_grad_layers(model):
    print()
    print(' require_grad layers:')
    # sanity check
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(' ', name)


if __name__ == '__main__':
    args = get_config('pretrain_autoencoder')

    ''' config param'''
    dataset_setting = 'unique_sent'
    
    num_epoch = args['num_epoch']
    lr = args['learning_rate']
    
    batch_size = args['batch_size']
    
    # task_name = 'task1'
    # task_name = 'task1_task2'
    # task_name = 'task1_task2_task3'
    # task_name = 'task1_task2_taskNRv2'
    task_name = args['task_name']
    save_path = args['save_path']
    train_input = args['train_input']
    print("train_input is:", train_input)   
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    experiment_name = args['save_name']
    save_name = f'pretrained_module_{experiment_name}_{task_name}_{batch_size}_{num_epoch}_{lr}'

    save_path_best = os.path.join(save_path, 'best')
    if not os.path.exists(save_path_best):
        os.makedirs(save_path_best)

    output_checkpoint_name_best = os.path.join(save_path_best, f'{save_name}.pt')

    save_path_last = os.path.join(save_path, 'last')
    if not os.path.exists(save_path_last):
        os.makedirs(save_path_last)

    output_checkpoint_name_last = os.path.join(save_path_last, f'{save_name}.pt')

    # subject_choice = 'ALL
    subject_choice = args['subjects']
    print(f'![Debug]using {subject_choice}')
    # eeg_type_choice = 'GD
    eeg_type_choice = args['eeg_type']
    print(f'[INFO]eeg type {eeg_type_choice}')
    # bands_choice = ['_t1'] 
    # bands_choice = ['_t1','_t2','_a1','_a2','_b1','_b2','_g1','_g2'] 
    bands_choice = args['eeg_bands']
    print(f'[INFO]using bands {bands_choice}')


    
    ''' set random seeds '''
    # https://pytorch.org/docs/stable/notes/randomness.html
    seed_val = 312

    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

    ''' set up dataloader '''
    whole_dataset_dicts = []
    if 'task1' in task_name:
        dataset_path_task1 = '[path]/task1-SR-dataset.pickle'
        with open(dataset_path_task1, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))
    if 'task2' in task_name:
        dataset_path_task2 = '[path]/task2-NR-dataset.pickle'
        with open(dataset_path_task2, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))
    if 'task3' in task_name:
        dataset_path_task3 = '[path]/task3-TSR-dataset.pickle'
        with open(dataset_path_task3, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))
    if 'taskNRv2' in task_name:
        dataset_path_taskNRv2 = '[path]/task2-NR-2.0-dataset.pickle'
        with open(dataset_path_taskNRv2, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))

    print()

    """save config"""
    cfg_dir = './config/decoding/'

    if not os.path.exists(cfg_dir):
        os.makedirs(cfg_dir)

    with open(os.path.join(cfg_dir,f'{save_name}.json'), 'w') as out_config:
        json.dump(args, out_config, indent = 4)

    
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

    # train dataset
    train_set = ZuCo_dataset(whole_dataset_dicts, 'train', tokenizer, subject = subject_choice, eeg_type = eeg_type_choice, bands = bands_choice, setting = dataset_setting, test_input=train_input)
    # dev dataset
    dev_set = ZuCo_dataset(whole_dataset_dicts, 'dev', tokenizer, subject = subject_choice, eeg_type = eeg_type_choice, bands = bands_choice, setting = dataset_setting, test_input=train_input)
    # test dataset
    test_set = ZuCo_dataset(whole_dataset_dicts, 'test', tokenizer, subject = subject_choice, eeg_type = eeg_type_choice, bands = bands_choice, setting = dataset_setting, test_input=train_input)

    
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

    pretrained = BartForConditionalGeneration.from_pretrained('facebook/bart-large').to(device)

    vocab_size = tokenizer.vocab_size
    eeg_encoder = AutoencoderKL(
        embed_dim=56,
        monitor="val/rec_loss",
        ddconfig={
            "double_z": True,
            "z_channels": 58, 
            "resolution": 256,
            "in_channels": 56,
            "out_ch": 56,
            "ch": 128,
            "ch_mult": [1, 2, 4, 4],
            "num_res_blocks": 2,
            "attn_resolutions": [],
            "dropout": 0.0
        },
        lossconfig={
            "target": "torch.nn.Identity"
        },
        vocab_size = vocab_size,
        latent_dim=1024,
        type = 'eeg'
    ).to(device)
    
    unet = UNetModel(
        image_size=512,
        dims = 1, 
        in_channels=56, 
        out_channels=56, 
        model_channels=320, 
        attention_resolutions=[4, 2, 1], 
        num_res_blocks=2, 
        channel_mult=[1, 2, 4, 4], 
        num_heads=8, 
        use_spatial_transformer=True, 
        transformer_depth=1, 
        context_dim=512, 
        use_checkpoint=False, 
        legacy=False
        ).to(device)
    
    model = pretrainedmodule(eeg_encoder, unet, pretrained, device).to(device)
    optimizer = optim.SGD(eeg_encoder.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5)
    # optimizer = optim.SGD(eeg_encoder.parameters(), lr=lr, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)


    model = train(dataloaders, device, model, optimizer, exp_lr_scheduler, num_epochs=num_epoch, checkpoint_path_best = output_checkpoint_name_best, checkpoint_path_last = output_checkpoint_name_last)