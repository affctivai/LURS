#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
preprocessed ZuCo npy 데이터 로더.

baseline_original 의 data.py 를 기준으로 하되, **pickle 대신 struct-npy 를 읽는다.**
get_input_sample / ZuCo_dataset 본문은 원본과 동일하다 — struct-npy 는 로드 이후
원본 pickle 과 완전히 같은 dict 를 돌려주므로 손댈 이유가 없다.

    # before
    with open(pickle_path, 'rb') as f:
        whole_dataset_dicts.append(pickle.load(f))

    # after
    whole_dataset_dicts.append(load_preprocessed('task1-SR'))

    train_set = ZuCo_dataset(whole_dataset_dicts, 'train', tokenizer,
                             eeg_type='GD', bands=bands, setting='unique_sent')

원본 data.py 대비 유일한 차이는 모듈 최상단에서 matplotlib / fuzzy_match 를
import 하지 않는다는 점이다(이 파일의 어떤 함수도 쓰지 않으며, 환경에 따라
설치돼 있지 않아 import 자체가 실패하기 때문).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .zuco_npy_store import load_zuco_npy

DEFAULT_NPY_ROOT = "/mnt/Members/speech/EEGToText/_data/zuco_npy"
TASKS = ["task1-SR", "task2-NR", "task3-TSR", "task2-NR-2.0"]

__all__ = [
    "normalize_1d",
    "get_input_sample",
    "ZuCo_dataset",
    "load_preprocessed",
    "load_preprocessed_many",
]


def load_preprocessed(task: str, npy_root: str = DEFAULT_NPY_ROOT, **kw):
    """pickle.load() 자리에 그대로 넣는다. task1-SR 등 task 이름을 준다."""
    return load_zuco_npy(Path(npy_root) / "preprocessed" / task, **kw)


def load_preprocessed_many(tasks, npy_root: str = DEFAULT_NPY_ROOT, **kw):
    """ZuCo_dataset 에 넘길 dict 리스트를 만든다."""
    return [load_preprocessed(t, npy_root, **kw) for t in tasks]


# --------------------------------------------------------------------------- #
# 이하 baseline_original/data.py 와 동일
# --------------------------------------------------------------------------- #
def normalize_1d(input_tensor):
    # normalize a 1d tensor
    mean = torch.mean(input_tensor)
    std = torch.std(input_tensor)
    input_tensor = (input_tensor - mean) / std
    return input_tensor


def bart_word_ids(tokenizer, ids):
    """BPE subword token -> word index. byte-level BPE는 새 단어 시작 토큰에 'Ġ' prefix가
    붙는다는 성질을 이용(fast tokenizer 없이도 동작). special token(<s>,</s>,<pad>)은 -1.
    EEG word embedding(word 단위)과 text hidden state(subword 단위)의 index를 맞추는 데 씀."""
    tokens = tokenizer.convert_ids_to_tokens(ids)
    special = set(tokenizer.all_special_tokens)
    word_ids = []
    cur = -1
    started = False
    for tok in tokens:
        if tok in special:
            word_ids.append(-1)
            continue
        if (not started) or tok.startswith('Ġ') or tok.startswith('Ċ'):
            cur += 1
            started = True
        word_ids.append(cur)
    return word_ids


def get_input_sample(sent_obj, tokenizer, eeg_type='GD',
                     bands=['_t1', '_t2', '_a1', '_a2', '_b1', '_b2', '_g1', '_g2'],
                     max_len=56, add_CLS_token=False):

    def get_word_embedding_eeg_tensor(word_obj, eeg_type, bands):
        frequency_features = []
        for band in bands:
            frequency_features.append(word_obj['word_level_EEG'][eeg_type][eeg_type + band])
        word_eeg_embedding = np.concatenate(frequency_features)
        if len(word_eeg_embedding) != 105 * len(bands):
            return None
        # assert len(word_eeg_embedding) == 105*len(bands)
        return_tensor = torch.from_numpy(word_eeg_embedding)
        return normalize_1d(return_tensor)

    def get_sent_eeg(sent_obj, bands):
        sent_eeg_features = []
        for band in bands:
            key = 'mean' + band
            sent_eeg_features.append(sent_obj['sentence_level_EEG'][key])
        sent_eeg_embedding = np.concatenate(sent_eeg_features)
        assert len(sent_eeg_embedding) == 105 * len(bands)
        return_tensor = torch.from_numpy(sent_eeg_embedding)
        return normalize_1d(return_tensor)

    if sent_obj is None:
        return None

    input_sample = {}
    # get target label
    target_string = sent_obj['content']
    target_tokenized = tokenizer(target_string, padding='max_length', max_length=max_len,
                                 truncation=True, return_tensors='pt', return_attention_mask=True)

    input_sample['target_ids'] = target_tokenized['input_ids'][0]

    # for JEPA
    mask_indices = [i for i, token in enumerate(sent_obj['word_tokens_with_mask']) if token == '[MASK]']
    mask_indices = mask_indices[:56] + [-1] * max(0, 56 - len(mask_indices))
    input_sample['mask_indices'] = torch.tensor(mask_indices)

    same_target_string = ' '.join([t for t in sent_obj['word_tokens_with_mask'] if t != '[MASK]'])
    same_target = tokenizer(same_target_string, padding='max_length', max_length=56,
                                    truncation=True, return_tensors='pt', return_attention_mask=True)

    input_sample['same_target'] = same_target['input_ids'][0]

    # same_target(subword) 위치 -> word(=input_embeddings) 위치 매핑. -1은 special/pad.
    word_ids = bart_word_ids(tokenizer, same_target['input_ids'][0].tolist())
    input_sample['word_ids'] = torch.tensor(word_ids, dtype=torch.long)

    # get sentence level EEG features
    sent_level_eeg_tensor = get_sent_eeg(sent_obj, bands)
    if torch.isnan(sent_level_eeg_tensor).any():
        return None
    input_sample['sent_level_EEG'] = sent_level_eeg_tensor

    # get sentiment label
    # handle some wierd case
    if 'emp11111ty' in target_string:
        target_string = target_string.replace('emp11111ty', 'empty')
    if 'film.1' in target_string:
        target_string = target_string.replace('film.1', 'film.')

    input_sample['sentiment_label'] = torch.tensor(-100)  # dummy value

    # get input embeddings
    word_embeddings = []

    """add CLS token embedding at the front"""
    if add_CLS_token:
        word_embeddings.append(torch.ones(105 * len(bands)))

    for word in sent_obj['word']:
        # add each word's EEG embedding as Tensors
        word_level_eeg_tensor = get_word_embedding_eeg_tensor(word, eeg_type, bands=bands)
        # check none, for v2 dataset
        if word_level_eeg_tensor is None:
            return None
        # check nan:
        if torch.isnan(word_level_eeg_tensor).any():
            return None

        word_embeddings.append(word_level_eeg_tensor)
    # pad to max_len
    while len(word_embeddings) < max_len:
        word_embeddings.append(torch.zeros(105 * len(bands)))

    input_sample['input_embeddings'] = torch.stack(word_embeddings)  # max_len * (105*num_bands)

    # mask out padding tokens
    input_sample['input_attn_mask'] = torch.zeros(max_len)  # 0 is masked out

    if add_CLS_token:
        input_sample['input_attn_mask'][:len(sent_obj['word']) + 1] = torch.ones(len(sent_obj['word']) + 1)
    else:
        input_sample['input_attn_mask'][:len(sent_obj['word'])] = torch.ones(len(sent_obj['word']))

    # mask out padding tokens reverted: handle different use case: this is for pytorch transformers
    input_sample['input_attn_mask_invert'] = torch.ones(max_len)  # 1 is masked out

    if add_CLS_token:
        input_sample['input_attn_mask_invert'][:len(sent_obj['word']) + 1] = torch.zeros(len(sent_obj['word']) + 1)
    else:
        input_sample['input_attn_mask_invert'][:len(sent_obj['word'])] = torch.zeros(len(sent_obj['word']))

    # mask out target padding for computing cross entropy loss
    input_sample['target_mask'] = target_tokenized['attention_mask'][0]
    input_sample['seq_len'] = len(sent_obj['word'])

    # clean 0 length data
    if input_sample['seq_len'] == 0:
        return None

    return input_sample


class ZuCo_dataset(Dataset):
    def __init__(self, input_dataset_dicts, phase, tokenizer, eeg_type='GD',
                 bands=['_t1', '_t2', '_a1', '_a2', '_b1', '_b2', '_g1', '_g2'],
                 setting='unique_sent', is_add_CLS_token=False):
        self.inputs = []
        self.tokenizer = tokenizer

        if not isinstance(input_dataset_dicts, list):
            input_dataset_dicts = [input_dataset_dicts]
        print(f'[INFO]loading {len(input_dataset_dicts)} task datasets')

        for input_dataset_dict in input_dataset_dicts:

            subjects = list(input_dataset_dict.keys())

            total_num_sentence = len(input_dataset_dict[subjects[0]])

            if setting == 'unique_sent':
                train_divider = int(0.8 * total_num_sentence)
                dev_divider = train_divider + int(0.1 * total_num_sentence)
                # take first 80% as trainset, 10% as dev and 10% as test
                if phase == 'train':
                    print('[INFO]initializing a train set...')
                    for key in subjects:
                        for i in range(train_divider):
                            input_sample = get_input_sample(input_dataset_dict[key][i], self.tokenizer, eeg_type, bands=bands, add_CLS_token=is_add_CLS_token)
                            if input_sample is not None:
                                self.inputs.append(input_sample)
                elif phase == 'dev':
                    print('[INFO]initializing a dev set...')
                    for key in subjects:
                        for i in range(train_divider, dev_divider):
                            input_sample = get_input_sample(input_dataset_dict[key][i], self.tokenizer, eeg_type, bands=bands, add_CLS_token=is_add_CLS_token)
                            if input_sample is not None:
                                self.inputs.append(input_sample)
                elif phase == 'test':
                    print('[INFO]initializing a test set...')
                    for key in subjects:
                        for i in range(dev_divider, total_num_sentence):
                            input_sample = get_input_sample(input_dataset_dict[key][i], self.tokenizer, eeg_type, bands=bands, add_CLS_token=is_add_CLS_token)
                            if input_sample is not None:
                                self.inputs.append(input_sample)
            # elif setting == 'unique_subj'
            else:
                # subject ['ZAB', 'ZDM', 'ZGW', 'ZJM', 'ZJN', 'ZJS', 'ZKB', 'ZKH', 'ZKW'] for train
                # subject ['ZMG'] for dev
                # subject ['ZPH'] for test
                subjects.remove(setting)
                train_divider = int(0.9 * total_num_sentence)
                dev_divider = train_divider + int(0.1 * total_num_sentence)
                if phase == 'test':
                    print(f'[INFO]initializing a test set using {setting} setting...')
                    for i in range(total_num_sentence):
                        input_sample = get_input_sample(input_dataset_dict[setting][i], self.tokenizer, eeg_type, bands=bands, add_CLS_token=is_add_CLS_token)
                        if input_sample is not None:
                            self.inputs.append(input_sample)
                        else:
                            ValueError(f'no valid test sample for {setting}, please check your setting!')
                if phase == 'train':
                    for key in subjects:
                        for i in range(train_divider):
                            input_sample = get_input_sample(input_dataset_dict[key][i], self.tokenizer, eeg_type, bands=bands, add_CLS_token=is_add_CLS_token)
                            if input_sample is not None:
                                self.inputs.append(input_sample)
                if phase == 'dev':
                    for i in range(train_divider, dev_divider):
                        for key in subjects:
                            input_sample = get_input_sample(input_dataset_dict[key][i], self.tokenizer, eeg_type, bands=bands, add_CLS_token=is_add_CLS_token)
                            if input_sample is not None:
                                self.inputs.append(input_sample)

        print('[INFO]input tensor size:', self.inputs[0]['input_embeddings'].size())
        print()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_sample = self.inputs[idx]
        return (
            input_sample['input_embeddings'],
            input_sample['seq_len'],
            input_sample['input_attn_mask'],
            input_sample['input_attn_mask_invert'],
            input_sample['target_ids'],
            input_sample['target_mask'],
            input_sample['sentiment_label'],
            input_sample['sent_level_EEG'],
            input_sample['mask_indices'],
            input_sample['same_target'],
            input_sample['word_ids'],
        )
