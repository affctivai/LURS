#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Raw EEG ZuCo npy 데이터 로더.

baseline_original 계열의 raw 로더(get_input_sample / ZuCo_dataset)를 기준으로 하되,
pickle 대신 struct-npy 를 읽는다. 로드 이후 문장 객체는 원본 pickle 과 완전히 같은
dict 이므로 get_input_sample 본문은 손대지 않았다.

    # before
    with open(raw_pickle_path, 'rb') as f:
        whole_dataset_dicts.append(pickle.load(f))

    # after
    whole_dataset_dicts.append(load_raw('task1-SR'))

    train_set = ZuCo_dataset(whole_dataset_dicts, 'train', tokenizer, bands=bands)

rawEEG 표현
-----------
같은 raw pickle 을 쓰면서도 리포지토리마다 rawData 를 다르게 가공한다.
  segment=False (기본, baseline_original)  : rawEEG = rawData 그대로 (105, 5500)
  segment=True  (DeWave)                   : window/stride 로 56조각 → (105, 56, 100)
필요한 쪽을 인자로 고르면 된다. 어느 쪽이든 원본 rawData 에서 파생되므로
데이터 자체는 동일하다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .zuco_npy_store import load_zuco_npy

DEFAULT_NPY_ROOT = "/mnt/Members/speech/EEGToText/_data/zuco_npy"
TASKS = ["task1-SR", "task2-NR", "task2-NR-2.0"]

__all__ = [
    "normalize_1d",
    "get_input_sample",
    "ZuCo_dataset",
    "load_raw",
    "load_raw_many",
]


def load_raw(task: str, npy_root: str = DEFAULT_NPY_ROOT, **kw):
    """pickle.load() 자리에 그대로 넣는다.

    copy_raw=False 를 주면 rawData 를 memmap 뷰로 그대로 돌려줘 문장당 2.3MB 복사를
    피할 수 있다(읽기 전용이 된다).
    """
    return load_zuco_npy(Path(npy_root) / "raw" / task, **kw)


def load_raw_many(tasks, npy_root: str = DEFAULT_NPY_ROOT, **kw):
    return [load_raw(t, npy_root, **kw) for t in tasks]


# --------------------------------------------------------------------------- #
# 이하 baseline_original 의 raw 로더와 동일
# --------------------------------------------------------------------------- #
def normalize_1d(input_tensor):
    # normalize a 1d tensor
    mean = torch.mean(input_tensor)
    std = torch.std(input_tensor)
    input_tensor = (input_tensor - mean) / std
    return input_tensor


def get_input_sample(sent_obj, tokenizer,
                     bands=['_t1', '_t2', '_a1', '_a2', '_b1', '_b2', '_g1', '_g2'],
                     max_len=56, segment=False, window=100, stride=50, num_segments=56):

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
    input_sample['target_mask'] = target_tokenized['attention_mask'][0]

    # get sentence level EEG features
    sent_level_eeg_tensor = get_sent_eeg(sent_obj, bands)
    if torch.isnan(sent_level_eeg_tensor).any():
        return None
    input_sample['sent_level_EEG'] = sent_level_eeg_tensor

    # handle some wierd case
    if 'emp11111ty' in target_string:
        target_string = target_string.replace('emp11111ty', 'empty')
    if 'film.1' in target_string:
        target_string = target_string.replace('film.1', 'film.')

    if sent_obj['rawData'] is None:
        return None
    if segment:
        segments = []
        for i in range(num_segments):
            start = i * stride
            end = start + window
            segments.append(sent_obj['rawData'][:, start:end])
        input_sample['rawEEG'] = np.transpose(np.array(segments), (1, 0, 2))
    else:
        input_sample['rawEEG'] = sent_obj['rawData']

    input_sample['input_attn_mask'] = torch.zeros(max_len)  # 0 is masked out
    input_sample['input_attn_mask'][:len(sent_obj['word'])] = torch.ones(len(sent_obj['word']))

    # mask out padding tokens reverted: handle different use case: this is for pytorch transformers
    input_sample['input_attn_mask_invert'] = torch.ones(max_len)  # 1 is masked out
    input_sample['input_attn_mask_invert'][:len(sent_obj['word'])] = torch.zeros(len(sent_obj['word']))

    input_sample['length'] = sent_obj['current_length']

    return input_sample


class ZuCo_dataset(Dataset):
    def __init__(self, input_dataset_dicts, phase, tokenizer, subject='ALL',
                 bands=['_t1', '_t2', '_a1', '_a2', '_b1', '_b2', '_g1', '_g2'],
                 setting='unique_sent', segment=False):
        self.inputs = []
        self.tokenizer = tokenizer

        if not isinstance(input_dataset_dicts, list):
            input_dataset_dicts = [input_dataset_dicts]
        print(f'[INFO]loading {len(input_dataset_dicts)} task datasets')

        def _add(d, key, rng):
            for i in rng:
                s = get_input_sample(d[key][i], self.tokenizer, bands=bands, segment=segment)
                if s is not None:
                    self.inputs.append(s)

        count = 0
        for input_dataset_dict in input_dataset_dicts:
            subjects = list(input_dataset_dict.keys())
            total_num_sentence = len(input_dataset_dict[subjects[0]])

            train_divider = int(0.8 * total_num_sentence)
            dev_divider = train_divider + int(0.1 * total_num_sentence)

            if setting == 'unique_sent':
                if phase == 'train':
                    print('[INFO]initializing a train set...')
                    for key in subjects:
                        _add(input_dataset_dict, key, range(train_divider))
                elif phase == 'dev':
                    print('[INFO]initializing a dev set...')
                    for key in subjects:
                        _add(input_dataset_dict, key, range(train_divider, dev_divider))
                elif phase == 'test':
                    print('[INFO]initializing a test set...')
                    for key in subjects:
                        _add(input_dataset_dict, key, range(dev_divider, total_num_sentence))
            elif setting == 'unique_subject':
                if subject == 'ALL':
                    raise ValueError('for unique_subject setting, please specify a subject name')
                test_subject = subject
                print(f'[INFO]using {test_subject} as test subject')

                if test_subject not in subjects:
                    count += 1
                    if phase == 'train':
                        for key in subjects:
                            _add(input_dataset_dict, key, range(dev_divider))
                    if phase == 'dev':
                        for key in subjects:
                            _add(input_dataset_dict, key, range(dev_divider, total_num_sentence))
                else:
                    subjects.remove(test_subject)
                    if phase == 'train':
                        for key in subjects:
                            _add(input_dataset_dict, key, range(dev_divider))
                    if phase == 'dev':
                        for key in subjects:
                            _add(input_dataset_dict, key, range(dev_divider, total_num_sentence))
                    if phase == 'test':
                        _add(input_dataset_dict, test_subject, range(total_num_sentence))
            print('++ adding task to dataset, now we have:', len(self.inputs))

        if count > 3:
            raise ValueError(f'{subject} not found in any dataset, please check your setting!')
        print()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_sample = self.inputs[idx]
        return (
            input_sample['sent_level_EEG'],
            input_sample['rawEEG'],
            input_sample['input_attn_mask'],
            input_sample['input_attn_mask_invert'],
            input_sample['target_ids'],
            input_sample['target_mask'],
            input_sample['length']
        )
