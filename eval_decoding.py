from metrics import compute_metrics
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import pickle
import json
import matplotlib.pyplot as plt
from glob import glob
import time
import copy
from tqdm import tqdm
import torch.nn.functional as F
import time
from transformers import BertLMHeadModel, BartTokenizer, BartForConditionalGeneration, BartConfig, BartForSequenceClassification, BertTokenizer, BertConfig, BertForSequenceClassification, RobertaTokenizer, RobertaForSequenceClassification, PegasusForConditionalGeneration, PegasusTokenizer, T5Tokenizer, T5ForConditionalGeneration, BertGenerationDecoder
from data import ZuCo_dataset
from model_decoding import LDMTranslator, pretrainedmodule
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from rouge import Rouge
from config import get_config
import evaluate
from evaluate import load
from ldm.autoencoder import AutoencoderKL
from ldm.openaimodel import UNetModel
import re


def compute_rouge(preds, labels):

    metrics={}
    for decoded_label, decoded_pred in zip(labels, preds):
        metric=rouge_score(decoded_pred,decoded_label)
        for key in metric.keys():
            metrics[key]=metrics.get(key, 0) + metric[key]
    for key in metrics.keys():
        metrics[key]=metrics[key]/len(labels)*100

    return metrics

metric = evaluate.load("sacrebleu")
cer_metric = load("cer")
wer_metric = load("wer")


def valid_text(s: str) -> bool:
    """알파벳·숫자 토큰이 하나 이상 남는 문장인지 확인"""
    if re.search(r'\w', s) is None:
        return ''
    return s

def remove_text_after_token(text, token='</s>'):
    # 특정 토큰 이후의 텍스트를 찾아 제거
    token_index = text.find(token)
    if token_index != -1:  # 토큰이 발견된 경우
        return text[:token_index]  # 토큰 이전까지의 텍스트 반환
    return text  # 토큰이 없으면 원본 텍스트 반환

def eval_model(dataloaders, device, tokenizer, model, output_all_results_path = './results/temp.txt' , score_results='./score_results/task.txt'):
    # modified from: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    target_string_list = []
    start_time = time.time()
    model.eval()   # Set model to evaluate mode
    
    target_tokens_list = []
    pred_tokens_list = []
    pred_string_list = []
    pred_tokens_list_previous = []
    pred_string_list_previous = []


    with open(output_all_results_path,'w') as f:
        vocab_size = tokenizer.vocab_size
        for input_embeddings, seq_len, input_masks, input_mask_invert, target_ids, target_mask, rawEEG in tqdm(dataloaders['test']):
            # load in batch
            input_embeddings_batch = input_embeddings.to(device).float() # B, 56, 840
            # input_embeddings_batch = sent_level_EEG.unsqueeze(1).to(device).float()
            input_masks_batch = input_masks.to(device) # B, 56
            target_ids_batch = target_ids.to(device) # B, 56
            input_mask_invert_batch = input_mask_invert.to(device) # B, 56
            context = target_ids_batch.clone()

            target_tokens = tokenizer.convert_ids_to_tokens(target_ids_batch[0].tolist(), skip_special_tokens = True)
            target_string = tokenizer.decode(target_ids_batch[0], skip_special_tokens = True)
            # print('target ids tensor:',target_ids_batch[0])
            # print('target ids:',target_ids_batch[0].tolist())
            # print('target tokens:',target_tokens)
            # print('target string:',target_strininvert.to(device) # B, 56
            
            f.write(f'target string: {target_string}\n')

            # add to list for later calculate bleu metric
            target_tokens_list.append([target_tokens])
            target_string_list.append(target_string)
            
            """replace padding ids in target_ids with -100"""
            target_ids_batch[target_ids_batch == tokenizer.pad_token_id] = -100 

            # target_ids_batch_label = target_ids_batch.clone().detach()
            # target_ids_batch_label[target_ids_batch_label == tokenizer.pad_token_id] = -100
            
            # Original code 
            # seq2seqLMoutput,_ = model(input_embeddings_batch, input_masks_batch, input_mask_invert_batch, target_ids_batch, context) # (batch, time, n_class)
            seq2seqLMoutput = model.evaluation(input_embeddings_batch, input_masks_batch, input_mask_invert_batch, target_ids_batch, context) # (batch, time, n_class)
            logits_previous = seq2seqLMoutput.logits
            probs_previous = logits_previous[0].softmax(dim = 1)
            values_previous, predictions_previous = probs_previous.topk(1)
            predictions_previous = torch.squeeze(predictions_previous)
            predicted_string_previous = remove_text_after_token(tokenizer.decode(predictions_previous).split('</s></s>')[0].replace('<s>',''))
            
            predicted_string_previous = valid_text(predicted_string_previous)
            f.write(f'predicted string with tf: {predicted_string_previous}\n')
            predictions_previous = predictions_previous.tolist()
            truncated_prediction_previous = []
            for t in predictions_previous:
                if t != tokenizer.eos_token_id:
                    truncated_prediction_previous.append(t)
                else:
                    break
            pred_tokens_previous = tokenizer.convert_ids_to_tokens(truncated_prediction_previous, skip_special_tokens = True)
            pred_tokens_list_previous.append(pred_tokens_previous)
            pred_string_list_previous.append(predicted_string_previous)
            

            # Modify code
            dummy_decoder_input_ids = torch.tensor([[tokenizer.pad_token_id]]).to(device) # B, 1
            predictions=model.generate(input_embeddings_batch, input_masks_batch, input_mask_invert_batch, target_ids_batch,dummy_decoder_input_ids,
                                       max_length=56,
                                       num_beams=5,
                                       do_sample=True,
                                       repetition_penalty= 5.0,
                                       no_repeat_ngram_size = 2,
                                       )
            
            predicted_string=tokenizer.batch_decode(predictions, skip_special_tokens=True)[0]
            # predicted_string=predicted_string.squeeze()
            
            predictions=tokenizer.encode(predicted_string)
            # print('predicted string:',predicted_string)
            predicted_string = valid_text(predicted_string)
            f.write(f'predicted string: {predicted_string}\n')
            f.write(f'################################################\n\n\n')

            # convert to int list
            # predictions = predictions.tolist() # 이미 list 형식이다. 
            truncated_prediction = []
            for t in predictions:
                if t != tokenizer.eos_token_id:
                    truncated_prediction.append(t)
                else:
                    break
            pred_tokens = tokenizer.convert_ids_to_tokens(truncated_prediction, skip_special_tokens = True)
            # print('predicted tokens:',pred_tokens)
            pred_tokens_list.append(pred_tokens)
            pred_string_list.append(predicted_string)
            # pred_tokens_list.extend(pred_tokens)
            # pred_string_list.extend(predicted_string)
            # print('################################################')
            # print()
    # print(f"pred_string_list : {pred_string_list}")
    
    """ calculate corpus bleu score """
    weights_list = [(1.0,),(0.5,0.5),(1./3.,1./3.,1./3.),(0.25,0.25,0.25,0.25)]
    corpus_bleu_scores = []
    corpus_bleu_scores_previous = []
    for weight in weights_list:
        # print('weight:',weight)
        corpus_bleu_score = corpus_bleu(target_tokens_list, pred_tokens_list, weights = weight)
        corpus_bleu_score_previous = corpus_bleu(target_tokens_list, pred_tokens_list_previous, weights = weight)
        corpus_bleu_scores.append(corpus_bleu_score)
        corpus_bleu_scores_previous.append(corpus_bleu_score_previous)
        print(f'corpus BLEU-{len(list(weight))} score:', corpus_bleu_score)
        print(f'corpus BLEU-{len(list(weight))} score with tf:', corpus_bleu_score_previous)


    """ calculate sacre bleu score """
    
    reference_list = [[item] for item in target_string_list]

    #print(f'ref: {reference_list}')
    #print(f'pred: {prediction_list}')
    sacre_blue = metric.compute(predictions=pred_string_list, references=reference_list)
    sacre_blue_previous = metric.compute(predictions=pred_string_list_previous, references=reference_list)
    print("sacreblue score: ", sacre_blue, '\n')
    print("sacreblue score with tf: ", sacre_blue_previous)
    
    print()
    """ calculate rouge score """
    def preprocess(text: str) -> str:
        tokens = tokenizer.tokenize(text)
        return " ".join(tokens)          # 공백 단위 토큰열
    
    pred_proc   = [preprocess(t) for t in pred_string_list]
    pred_previous_proc   = [preprocess(t) for t in pred_string_list_previous]
    target_proc = [preprocess(t) for t in target_string_list]
    rouge = Rouge()

    try:
        rouge_scores = rouge.get_scores(pred_proc, target_proc, avg = True, ignore_empty=True)
    except ValueError as e:
        rouge_scores = 'Hypothesis is empty'

    try:
        rouge_scores_previous = rouge.get_scores(pred_previous_proc, target_proc, avg = True, ignore_empty=True)
    except ValueError as e:
        rouge_scores_previous = 'Hypothesis is empty'

    print()
    """ calculate WER score """
    #wer = WordErrorRate()
    wer_scores = wer_metric.compute(predictions=pred_string_list, references=target_string_list)
    wer_scores_previous = wer_metric.compute(predictions=pred_string_list_previous, references=target_string_list)
    print("WER score:", wer_scores)
    print("WER score with tf:", wer_scores_previous)
    

    """ calculate CER score """
    cer_scores = cer_metric.compute(predictions=pred_string_list, references=target_string_list)
    cer_scores_previous = cer_metric.compute(predictions=pred_string_list_previous, references=target_string_list)
    print("CER score:", cer_scores)
    print("CER score with tf:", cer_scores_previous)


    end_time = time.time()
    print(f"Evaluation took {(end_time-start_time)/60} minutes to execute.")

     # score_results (only fix teacher-forcing)
    file_content = [
    f'corpus_bleu_score = {corpus_bleu_scores}',
    f'sacre_blue_score = {sacre_blue}',
    f'rouge_scores = {rouge_scores}',
    f'wer_scores = {wer_scores}',
    f'cer_scores = {cer_scores}',
    f'corpus_bleu_score_with_tf = {corpus_bleu_scores_previous}',
    f'sacre_blue_score_with_tf = {sacre_blue_previous}',
    f'rouge_scores_with_tf = {rouge_scores_previous}',
    f'wer_scores_with_tf = {wer_scores_previous}',
    f'cer_scores_with_tf = {cer_scores_previous}',
    ]
    if not os.path.exists(score_results):
        with open(score_results, 'w') as f:
            f.write("")
    with open(score_results, "a") as file_results:
        for line in file_content:
            if isinstance(line, list):
                for item in line:
                    file_results.write(str(item) + "\n")
            else:
                file_results.write(str(line) + "\n")

    print(f'[INFO]score results saved in {score_results}')
    



if __name__ == '__main__': 
    batch_size = 1
    ''' get args'''
    args = get_config('eval_decoding')

    
    test_input = args['test_input']
    print("test_input is:", test_input)
    train_input = args['train_input']
    print("train_input is:", train_input)
    ''' load training config'''
    training_config = json.load(open(args['config_path']))
    save_name = args['config_path'].split('/')[-1].split('.')[0]

    subject_choice = training_config['subjects']
    print(f'[INFO]subjects: {subject_choice}')
    eeg_type_choice = training_config['eeg_type']
    print(f'[INFO]eeg type: {eeg_type_choice}')
    bands_choice = training_config['eeg_bands']
    print(f'[INFO]using bands: {bands_choice}')
    
    dataset_setting = 'unique_sent'

    task_name = training_config['task_name']
    model_name = training_config['model_name']
    if not os.path.exists('./results'):
        os.makedirs('./results')
    if not os.path.exists('./score_results'):
        os.makedirs('./score_results')
        
    if test_input == 'EEG' and train_input=='EEG':
        print("EEG and EEG")
        output_all_results_path = f'./results/{save_name}.txt'
        score_results = f'./score_results/{save_name}.txt'
    else:
        output_all_results_path = f'./results/{save_name}.txt'
        score_results = f'./score_results/{save_name}.txt'


    ''' set random seeds '''
    seed_val = 20 #500
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    ''' set up device '''
    # use cuda
    if torch.cuda.is_available():  
        dev = args['cuda'] 
    else:  
        dev = "cpu"
    # CUDA_VISIBLE_DEVICES=0,1,2,3  
    device = torch.device(dev)
    print(f'[INFO]using device {dev}')

    # task_name = 'task1_task2_task3'


    ''' set up dataloader '''
    whole_dataset_dicts = []
    if 'task1' in task_name:
        dataset_path_task1 = '/home/saul_park/workspace/code/Baseline/dataset/ZuCo/task1-SR/pickle/task1-SR-dataset.pickle'
        with open(dataset_path_task1, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))
    if 'task2' in task_name:
        dataset_path_task2 = '/home/saul_park/workspace/code/Baseline/dataset/ZuCo/task2-NR/pickle/task2-NR-dataset.pickle'
        with open(dataset_path_task2, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))
    if 'task3' in task_name:
        dataset_path_task3 = '/home/saul_park/workspace/data/dataset/ZuCo/task3-TSR/pickle/task3-TSR-dataset.pickle'
        with open(dataset_path_task3, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))
    if 'taskNRv2' in task_name:
        dataset_path_taskNRv2 = '/home/saul_park/workspace/code/Baseline/dataset/ZuCo/task2-NR-2.0/pickle/task2-NR-2.0-dataset.pickle'
        with open(dataset_path_taskNRv2, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))
    print()
    
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

    # test dataset
    test_set = ZuCo_dataset(whole_dataset_dicts, 'test', tokenizer, subject = subject_choice, eeg_type = eeg_type_choice, bands = bands_choice, setting = dataset_setting, test_input=test_input)

    dataset_sizes = {"test_set":len(test_set)}
    print('[INFO]test_set size: ', len(test_set))
    
    # dataloaders
    test_dataloader = DataLoader(test_set, batch_size = batch_size, shuffle=False, num_workers=4)

    dataloaders = {'test':test_dataloader}

    ''' set up model '''
    checkpoint_path = args['checkpoint_path']
    

    pretrained_bart = BartForConditionalGeneration.from_pretrained('facebook/bart-large')

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

    pretrained_module = pretrainedmodule(eeg_encoder, unet, pretrained_bart, device).to(device)
    
    model = LDMTranslator(pretrained=pretrained_module)

    state_dict = torch.load(checkpoint_path)
    # new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    
    # model.load_state_dict(torch.load(checkpoint_path))
    model.to(device)

    ''' eval '''
    eval_model(dataloaders, device, tokenizer, model, output_all_results_path = output_all_results_path, score_results=score_results)
