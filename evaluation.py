import os
from data.data import ZuCoNpyDataset
import numpy as np
import torch
from torch.utils.data import DataLoader
import pickle
import json
import time
from tqdm import tqdm
import torch.nn.functional as F
import time
from transformers import BartTokenizer, BartForConditionalGeneration
# from data.data_for_rawEEG import ZuCo_dataset
from model import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu
from rouge import Rouge
import sys
from metrics import compute_bertscore, compute_bleurt, compute_entailment, compute_entity_preservation
# import evaluate
# from evaluate import load
from module.ldm.autoencoder import AutoencoderKL
from module.ldm.openaimodel import UNetModel
from dataclasses import asdict
from module.compare_model.EEGNet import EEGNet
from module.compare_model.DeepConvNet import DeepConvNet
from module.compare_model.MLP import MLP
import re
from torch.optim.swa_utils import AveragedModel

# import os, tempfile
# TMPDIR = "/home/pymp_tmp"
# os.environ["TMPDIR"] = TMPDIR
# os.environ["TEMP"] = TMPDIR
# os.environ["TMP"] = TMPDIR
# tempfile.tempdir = TMPDIR


# def compute_rouge(preds, labels):

#     metrics={}
#     for decoded_label, decoded_pred in zip(labels, preds):
#         metric=rouge_score(decoded_pred,decoded_label)
#         for key in metric.keys():
#             metrics[key]=metrics.get(key, 0) + metric[key]
#     for key in metrics.keys():
#         metrics[key]=metrics[key]/len(labels)*100

#     return metrics

# metric = evaluate.load("sacrebleu")
# cer_metric = load("cer")
# wer_metric = load("wer")


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

def truncate_at_eos(ids, eos_token_id):
    if eos_token_id in ids:
        return ids[:ids.index(eos_token_id)]
    return ids

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

    # ema_model = AveragedModel(
    #     model,
    #     avg_fn=lambda avg, param, n: 0.999 * avg + 0.001 * param
    # )
    with open(output_all_results_path,'w') as f:
        vocab_size = tokenizer.vocab_size
        for input_embeddings, seq_len, input_masks, input_mask_invert, target_ids, target_mask, sentiment_label, sent_level_EEG in tqdm(dataloaders['test'], file=sys.stderr):
            # load in batch
            input_embeddings_batch = input_embeddings.to(device).float() # B, 56, 840
            # input_embeddings_batch = sent_level_EEG.unsqueeze(1).to(device).float()
            input_masks_batch = input_masks.to(device) # B, 56
            target_ids_batch = target_ids.to(device) # B, 56
            input_mask_invert_batch = input_mask_invert.to(device) # B, 56
            context = target_ids_batch.clone()
            cur_batch_size = target_ids_batch.shape[0]

            # target strings/tokens: one batch_decode call instead of per-sample decode
            target_strings = tokenizer.batch_decode(target_ids_batch, skip_special_tokens=True)
            target_tokens_batch = [tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=True) for ids in target_ids_batch.tolist()]
            f.write(''.join(f'target string: {s}\n' for s in target_strings))
            target_tokens_list.extend([[t] for t in target_tokens_batch])
            target_string_list.extend(target_strings)

            """replace padding ids in target_ids with -100"""
            target_ids_batch[target_ids_batch == tokenizer.pad_token_id] = -100

            # sample latent once (reverse diffusion), reuse for both teacher-forced and free-running below
            embedding_out = model.encode_and_sample(input_embeddings_batch)

            # Original code (teacher-forced)
            seq2seqLMoutput = model.evaluate(input_embeddings_batch, input_masks_batch, target_ids_batch, input_embeddings_batch, embedding_out=embedding_out) # (batch, time, n_class)
            predictions_previous_ids = seq2seqLMoutput.logits.argmax(dim=-1) # (B, time) -- argmax == softmax-then-top1
            raw_strings_previous = tokenizer.batch_decode(predictions_previous_ids, skip_special_tokens=False)
            predicted_strings_previous = [
                valid_text(remove_text_after_token(s.split('</s></s>')[0].replace('<s>', '')))
                for s in raw_strings_previous
            ]
            f.write(''.join(f'predicted string with tf: {s}\n' for s in predicted_strings_previous))
            pred_tokens_list_previous.extend([
                tokenizer.convert_ids_to_tokens(truncate_at_eos(ids, tokenizer.eos_token_id), skip_special_tokens=True)
                for ids in predictions_previous_ids.tolist()
            ])
            pred_string_list_previous.extend(predicted_strings_previous)

            # Modify code (free-running generation)
            dummy_decoder_input_ids = torch.full((cur_batch_size, 1), tokenizer.pad_token_id, dtype=torch.long, device=device) # B, 1
            predictions=model.generate(input_embeddings_batch, input_masks_batch, dummy_decoder_input_ids, context,
                                       embedding_out=embedding_out,
                                       max_length=56,
                                       num_beams=5,
                                       do_sample=True,
                                       repetition_penalty= 5.0,
                                       no_repeat_ngram_size = 2,
                                       # num_beams=5,encoder_no_repeat_ngram_size =1,
                                       # do_sample=True, top_k=15,temperature=0.5,num_return_sequences=5,
                                       # early_stopping=True
                                       )

            predicted_strings = [valid_text(s) for s in tokenizer.batch_decode(predictions, skip_special_tokens=True)]
            f.write(''.join(f'predicted string: {s}\n################################################\n\n\n' for s in predicted_strings))
            pred_tokens_list.extend([
                tokenizer.convert_ids_to_tokens(truncate_at_eos(tokenizer.encode(s), tokenizer.eos_token_id), skip_special_tokens=True)
                for s in predicted_strings
            ])
            pred_string_list.extend(predicted_strings)
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
    """ calculate BERTScore / BLEURT / entailment / entity preservation """
    bertscore_scores = compute_bertscore(pred_string_list, target_string_list, device=device)
    bertscore_scores_previous = compute_bertscore(pred_string_list_previous, target_string_list, device=device)
    print('BERTScore F1:', bertscore_scores['f1'])
    print('BERTScore F1 with tf:', bertscore_scores_previous['f1'])

    bleurt_score = compute_bleurt(pred_string_list, target_string_list, device=device)
    bleurt_score_previous = compute_bleurt(pred_string_list_previous, target_string_list, device=device)
    print('BLEURT score:', bleurt_score)
    print('BLEURT score with tf:', bleurt_score_previous)

    entailment_scores = compute_entailment(pred_string_list, target_string_list, device=device)
    entailment_scores_previous = compute_entailment(pred_string_list_previous, target_string_list, device=device)
    print('entailment accuracy:', entailment_scores['entailment_accuracy'])
    print('entailment accuracy with tf:', entailment_scores_previous['entailment_accuracy'])

    entity_preservation_score = compute_entity_preservation(pred_string_list, target_string_list)
    entity_preservation_score_previous = compute_entity_preservation(pred_string_list_previous, target_string_list)
    print('entity preservation accuracy:', entity_preservation_score)
    print('entity preservation accuracy with tf:', entity_preservation_score_previous)

    print()
    """ calculate WER score """
    #wer = WordErrorRate()
    # wer_scores = wer_metric.compute(predictions=pred_string_list, references=target_string_list)
    # wer_scores_previous = wer_metric.compute(predictions=pred_string_list_previous, references=target_string_list)
    # print("WER score:", wer_scores)
    # print("WER score with tf:", wer_scores_previous)
    

    """ calculate CER score """
    # cer_scores = cer_metric.compute(predictions=pred_string_list, references=target_string_list)
    # cer_scores_previous = cer_metric.compute(predictions=pred_string_list_previous, references=target_string_list)
    # print("CER score:", cer_scores)
    # print("CER score with tf:", cer_scores_previous)


    end_time = time.time()
    print(f"Evaluation took {(end_time-start_time)/60} minutes to execute.")

     # score_results (only fix teacher-forcing)
    file_content = [
    f'corpus_bleu_score = {corpus_bleu_scores}',
    f'rouge_scores = {rouge_scores}',
    f'bertscore = {bertscore_scores}',
    f'bleurt_score = {bleurt_score}',
    f'entailment_scores = {entailment_scores}',
    f'entity_preservation_accuracy = {entity_preservation_score}',
    # f'wer_scores = {wer_scores}',
    # f'cer_scores = {cer_scores}',
    f'corpus_bleu_score_with_tf = {corpus_bleu_scores_previous}',
    f'rouge_scores_with_tf = {rouge_scores_previous}',
    f'bertscore_with_tf = {bertscore_scores_previous}',
    f'bleurt_score_with_tf = {bleurt_score_previous}',
    f'entailment_scores_with_tf = {entailment_scores_previous}',
    f'entity_preservation_accuracy_with_tf = {entity_preservation_score_previous}',
    # f'wer_scores_with_tf = {wer_scores_previous}',
    # f'cer_scores_with_tf = {cer_scores_previous}',
    ]
    if not os.path.exists(score_results):
        with open(score_results, 'w') as f:
            f.write("")
    with open(score_results, "w") as file_results:
        for line in file_content:
            if isinstance(line, list):
                for item in line:
                    file_results.write(str(item) + "\n")
            else:
                file_results.write(str(line) + "\n")
    print(f'[INFO]score results saved in {score_results}', file=sys.stderr)
    print(f'[INFO]score results saved in {score_results}')


if __name__ == '__main__':
    ''' get args'''
    args = get_config('eval_decoding')
    batch_size = args['batch_size']


    

    
    # test_input = args['test_input'] # word-level or rawEEG
    # print("test_input is:", test_input)
    ''' load training config'''
    with open(args['config_path'], 'r') as f:
        configs = json.load(f)
    # training_config = configs['args']
    training_config = configs['pretrained_args']
    module_state_dict_config = configs['module_state_dict']
    # save_name = args['config_path'].split('/')[-1].split('.')[0]
    save_name = args['config_path'].split('/')[-1].split('.')[0] if args['save_name'] is None else args['save_name']
    # subject_choice = training_config['subjects']
    # print(f'[INFO]subjects: {subject_choice}')

    is_con = training_config['is_con']
    is_geo = training_config['is_geo']
    is_kl = training_config['is_kl']


    eeg_type_choice = training_config['eeg_type']
    print(f'[INFO]eeg type: {eeg_type_choice}')
    bands_choice = training_config['eeg_bands']
    print(f'[INFO]using bands: {bands_choice}')
    subject_choice  = training_config['subjects']
    print(f'[INFO]subjects: {subject_choice}')
    test_input      = training_config['test_input']
    print(f'[INFO]test_input: {test_input}')


    task_name = training_config['task_name']
    model_name = configs['args']['model_name']
    if not os.path.exists('./results'):
        os.makedirs('./results')
    if not os.path.exists('./score_results'):
        os.makedirs('./score_results')
        
    output_all_results_path = f'./results/{save_name}.txt'
    score_results = f'./score_results/{save_name}.txt'


    ''' set random seeds '''
    seed_val = args['seed']
    set_seed(seed_val)

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


    
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')


    dataset_setting = args['dataset_setting']

    # test dataset

    ''' set up dataloader '''
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

    test_set  = ZuCoNpyDataset(task_npy_dirs, phase='test',  tokenizer=tokenizer, subject=subject_choice, eeg_type=eeg_type_choice, bands=bands_choice, setting=dataset_setting, test_input=test_input)
    print()
    

    dataset_sizes = {"test_set":len(test_set)}
    print('[INFO]test_set size: ', len(test_set))
    
    # dataloaders
    test_dataloader = DataLoader(test_set, batch_size = batch_size, shuffle=False, num_workers=4)

    dataloaders = {'test':test_dataloader}

    ''' set up model '''
    checkpoint_path = args['checkpoint_path']
    

    pretrained_LM = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
    vocab_size = tokenizer.vocab_size

    denoising_module_statte_dict = module_state_dict_config["denoising_module"]
    autoencoder_state_dict = module_state_dict_config["autoencoder"]

    if model_name == 'EEGNet':
        denoising_module = EEGNet(
            **denoising_module_statte_dict
        ).to(device)

    elif model_name == 'DConv':
        denoising_module = DeepConvNet(
            **denoising_module_statte_dict
        ).to(device)

    elif model_name == 'MLP':
        denoising_module = MLP(
            **denoising_module_statte_dict
        ).to(device)
    else:
        denoising_module = UNetModel(
            **denoising_module_statte_dict
            ).to(device)
        
    vocab_size = tokenizer.vocab_size

    autoencoder = AutoencoderKL(
        **autoencoder_state_dict
    ).to(device)
    if model_name == 'main_diffusion' or model_name == 'full_diffusion':
        is_diffusion = True
    else:
        is_diffusion = False

    is_compare_model = model_name in ('EEGNet', 'DConv', 'MLP')

    # pretrained_model_name = configs['args']['config_path'].split('/')[-1].split('.')[0].split('-')[0]
    pretrained_model_name = training_config['model_name']
    pretrain_opts = BuildOptions(model_name=pretrained_model_name, is_con=is_con, is_geo=is_geo,is_kl=is_kl, is_diffusion=is_diffusion, is_compare_model=is_compare_model,
                                 autoencoder=autoencoder, denoising_module=denoising_module, pretrained_LM=pretrained_LM, device = device)
    pretrained_module = make_model(pretrain_opts)

    opts = BuildOptions(model_name=model_name, is_con=is_con, is_geo=is_geo, is_kl=is_kl, device = device,pretrained_module=pretrained_module, is_diffusion=is_diffusion, is_compare_model=is_compare_model)
    model = make_model(opts)

    state_dict = torch.load(checkpoint_path)
    # new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)


    # model.load_state_dict(torch.load(checkpoint_path))
    model.to(device)

    ''' eval '''
    eval_model(dataloaders, device, tokenizer, model, output_all_results_path = output_all_results_path, score_results=score_results)
