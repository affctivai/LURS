import os
import numpy as np
import torch
import pickle
from torch.utils.data import Dataset, DataLoader
import json
# import matplotlib.pyplot as plt
from glob import glob
from transformers import BartTokenizer, BertTokenizer
from tqdm import tqdm
from fuzzy_match import match
from fuzzy_match import algorithims

# macro
#ZUCO_SENTIMENT_LABELS = json.load(open('./dataset/ZuCo/task1-SR/sentiment_labels/sentiment_labels.json'))
#SST_SENTIMENT_LABELS = json.load(open('./dataset/stanfordsentiment/ternary_dataset.json'))

def normalize_1d(input_tensor):
    # normalize a 1d tensor
    mean = torch.mean(input_tensor)
    std = torch.std(input_tensor)
    input_tensor = (input_tensor - mean)/std
    return input_tensor 

def get_input_sample(sent_obj, tokenizer, eeg_type = 'GD', bands = ['_t1','_t2','_a1','_a2','_b1','_b2','_g1','_g2'], max_len = 56, add_CLS_token = False):
    
    def get_word_embedding_eeg_tensor(word_obj, eeg_type, bands):
        frequency_features = []
        for band in bands:
            frequency_features.append(word_obj['word_level_EEG'][eeg_type][eeg_type+band])
        word_eeg_embedding = np.concatenate(frequency_features)
        if len(word_eeg_embedding) != 105*len(bands):
            return None
        # assert len(word_eeg_embedding) == 105*len(bands)
        return_tensor = torch.from_numpy(word_eeg_embedding)
        return normalize_1d(return_tensor)

    def get_sent_eeg(sent_obj, bands):
        sent_eeg_features = []
        for band in bands:
            key = 'mean'+band
            sent_eeg_features.append(sent_obj['sentence_level_EEG'][key])
        sent_eeg_embedding = np.concatenate(sent_eeg_features)
        assert len(sent_eeg_embedding) == 105*len(bands)
        return_tensor = torch.from_numpy(sent_eeg_embedding)
        return normalize_1d(return_tensor)

    if sent_obj is None:
        return None

    input_sample = {}
    # get target label
    target_string = sent_obj['content']
    target_tokenized = tokenizer(target_string, padding='max_length', max_length=max_len, truncation=True, return_tensors='pt', return_attention_mask = True)
    
    input_sample['target_ids'] = target_tokenized['input_ids'][0]
    
    # get sentence level EEG features
    sent_level_eeg_tensor = get_sent_eeg(sent_obj, bands)
    if torch.isnan(sent_level_eeg_tensor).any():
        return None
    input_sample['sent_level_EEG'] = sent_level_eeg_tensor

    # get sentiment label
    # handle some wierd case
    if 'emp11111ty' in target_string:
        target_string = target_string.replace('emp11111ty','empty')
    if 'film.1' in target_string:
        target_string = target_string.replace('film.1','film.')
    
    #if target_string in ZUCO_SENTIMENT_LABELS:
    #    input_sample['sentiment_label'] = torch.tensor(ZUCO_SENTIMENT_LABELS[target_string]+1) # 0:Negative, 1:Neutral, 2:Positive
    #else:
    #    input_sample['sentiment_label'] = torch.tensor(-100) # dummy value
    input_sample['sentiment_label'] = torch.tensor(-100) # dummy value

    # get input embeddings
    word_embeddings = []

    """add CLS token embedding at the front"""
    if add_CLS_token:
        word_embeddings.append(torch.ones(105*len(bands)))

    for word in sent_obj['word']:
        # add each word's EEG embedding as Tensors
        word_level_eeg_tensor = get_word_embedding_eeg_tensor(word, eeg_type, bands = bands)
        # check none, for v2 dataset
        if word_level_eeg_tensor is None:
            return None
        # check nan:
        if torch.isnan(word_level_eeg_tensor).any():
            return None
            

        word_embeddings.append(word_level_eeg_tensor)
    # pad to max_len
    while len(word_embeddings) < max_len:
        word_embeddings.append(torch.zeros(105*len(bands)))

    input_sample['input_embeddings'] = torch.stack(word_embeddings) # max_len * (105*num_bands)

    # mask out padding tokens
    input_sample['input_attn_mask'] = torch.zeros(max_len) # 0 is masked out

    if add_CLS_token:
        input_sample['input_attn_mask'][:len(sent_obj['word'])+1] = torch.ones(len(sent_obj['word'])+1) # 1 is not masked
    else:
        input_sample['input_attn_mask'][:len(sent_obj['word'])] = torch.ones(len(sent_obj['word'])) # 1 is not masked
    

    # mask out padding tokens reverted: handle different use case: this is for pytorch transformers
    input_sample['input_attn_mask_invert'] = torch.ones(max_len) # 1 is masked out

    if add_CLS_token:
        input_sample['input_attn_mask_invert'][:len(sent_obj['word'])+1] = torch.zeros(len(sent_obj['word'])+1) # 0 is not masked
    else:
        input_sample['input_attn_mask_invert'][:len(sent_obj['word'])] = torch.zeros(len(sent_obj['word'])) # 0 is not masked

    

    # mask out target padding for computing cross entropy loss
    input_sample['target_mask'] = target_tokenized['attention_mask'][0]
    input_sample['seq_len'] = len(sent_obj['word'])
    
    # clean 0 length data
    if input_sample['seq_len'] == 0:
        # print('discard length zero instance: ', target_string)
        return None

    return input_sample

class ZuCo_dataset(Dataset):
    def __init__(self, input_dataset_dicts, phase, tokenizer, subject='ALL', eeg_type='GD', bands=['_t1','_t2','_a1','_a2','_b1','_b2','_g1','_g2'], setting='unique_sent', is_add_CLS_token=False, seed=42):
        self.inputs = []
        self.tokenizer = tokenizer

        if not isinstance(input_dataset_dicts, list):
            input_dataset_dicts = [input_dataset_dicts]
        print(f'[INFO]loading {len(input_dataset_dicts)} task datasets')

        # LOSO mode: subject is a specific subject ID (not "ALL" and not a sentence-split setting)
        loso_mode = (subject != 'ALL' and subject != setting)

        if loso_mode:
            # target subject → test, remaining subjects → 8:2 train:dev (shuffled)
            test_keys = []
            other_keys = []

            for di, input_dataset_dict in enumerate(input_dataset_dicts):
                subjects = list(input_dataset_dict.keys())
                if subject not in subjects:
                    print(f"[WARN] Subject '{subject}' not found in dataset {di}; "
                          f"all samples from this dataset go to train/dev pool")
                for key in subjects:
                    for si in range(len(input_dataset_dict[key])):
                        if key == subject:
                            test_keys.append((di, key, si))
                        else:
                            other_keys.append((di, key, si))

            rng = np.random.default_rng(seed)
            perm = rng.permutation(len(other_keys))
            split = int(len(perm) * 0.8)
            train_keys = [other_keys[i] for i in perm[:split]]
            dev_keys = [other_keys[i] for i in perm[split:]]

            if phase == 'train':
                print('[INFO]initializing a train set (LOSO)...')
                selected_keys = train_keys
            elif phase == 'dev':
                print('[INFO]initializing a dev set (LOSO)...')
                selected_keys = dev_keys
            elif phase == 'test':
                print('[INFO]initializing a test set (LOSO)...')
                selected_keys = test_keys
            else:
                raise ValueError(f"Unknown phase: {phase}")

            for di, key, si in selected_keys:
                input_sample = get_input_sample(input_dataset_dicts[di][key][si], tokenizer, eeg_type, bands=bands, add_CLS_token=is_add_CLS_token)
                if input_sample is not None:
                    self.inputs.append(input_sample)

        else:
            for input_dataset_dict in input_dataset_dicts:

                subjects = list(input_dataset_dict.keys())
                total_num_sentence = len(input_dataset_dict[subjects[0]])

                if setting == 'unique_sent':
                    train_divider = int(0.8*total_num_sentence)
                    dev_divider = train_divider + int(0.1*total_num_sentence)
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
                # elif setting == 'unique_subj':
                else:
                    # subject ['ZAB', 'ZDM', 'ZGW', 'ZJM', 'ZJN', 'ZJS', 'ZKB', 'ZKH', 'ZKW'] for train
                    # subject ['ZMG'] for dev
                    # subject ['ZPH'] for test
                    subjects.remove(setting)
                    train_divider = int(0.8*total_num_sentence)
                    dev_divider = train_divider + int(0.2*total_num_sentence)
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

            # print('++ adding task to dataset, now we have:', len(self.inputs))

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
            input_sample['sent_level_EEG']
        )
        # keys: input_embeddings, input_attn_mask, input_attn_mask_invert, target_ids, target_mask, 
# =========================================================
# Memmap-friendly NPY Dataset (RAM efficient)
# =========================================================

def _resolve_memmap_root(task_dir: str) -> str:
    """
    task_dir는 보통:
      /mnt/Members/speech/jd_data/zuco_decoding_npy/task1-SR
    converter가 만든 실제 memmap root는:
      <task_dir>/memmap_v1
    """
    task_dir = str(task_dir)
    # 1) task_dir 자체가 memmap_v1 일 수도 있음
    cand1 = os.path.join(task_dir, "meta.json")
    if os.path.isfile(cand1):
        return task_dir

    cand2 = os.path.join(task_dir, "memmap_v1", "meta.json")
    if os.path.isfile(cand2):
        return os.path.join(task_dir, "memmap_v1")

    # 그 외: 하위에서 meta.json 탐색
    for sub in ["memmap", "memmap_v0", "memmap_v1", "memmap_v2"]:
        cand = os.path.join(task_dir, sub, "meta.json")
        if os.path.isfile(cand):
            return os.path.join(task_dir, sub)

    raise FileNotFoundError(
        f"meta.json not found. Expected one of:\n"
        f"  - {task_dir}/meta.json\n"
        f"  - {task_dir}/memmap_v1/meta.json\n"
        f"(did you run convert_zuco_pickle_to_memmap_npy.py?)"
    )


class _ZuCoMemmapPart:
    """
    하나의 task(=하나의 pickle) + 하나의 phase(train/dev/test) 에 해당하는 memmap 조각.
    subject='ALL' 또는 특정 subject slice만 참조하도록 구성 가능.
    """
    def __init__(self, memmap_root: str, phase: str, subject: str,
                 expected: dict,
                 mmap_mode: str = "r",
                 require_meta: bool = True,
                 noise = False):
        self.memmap_root = memmap_root
        self.phase = phase
        self.subject = subject
        self.mmap_mode = mmap_mode
        self.require_meta = require_meta
        self.noise = noise
        meta_path = os.path.join(memmap_root, "meta.json")
        if not os.path.isfile(meta_path):
            if require_meta:
                raise FileNotFoundError(f"meta.json not found: {meta_path}")
            else:
                self.meta = None
                return

        with open(meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)

        # config 일치 검사 (정확히 동일한 역할 보장용)
        # (meta가 없으면 검사 불가)
        for k, v in expected.items():
            if k not in self.meta:
                continue
            if self.meta[k] != v:
                raise ValueError(
                    f"[CONFIG MISMATCH] {k}: requested={v} but file_meta={self.meta[k]}\n"
                    f"  memmap_root={memmap_root}\n"
                    f"  (re-convert with the same config)"
                )

        if phase not in self.meta["phases"]:
            raise ValueError(f"phase '{phase}' not in meta phases: {list(self.meta['phases'].keys())}")

        phase_info = self.meta["phases"][phase]
        self.total_len = int(phase_info["num_samples"])
        self.subjects = phase_info["subjects"]
        self.subject_slices = phase_info["subject_slices"]  # dict: subj -> {start,length}

        # slice 결정
        if subject == "ALL":
            self.start = 0
            self.length = self.total_len
        else:
            if subject not in self.subject_slices:
                raise ValueError(f"subject '{subject}' not found in meta subject_slices.")
            self.start = int(self.subject_slices[subject]["start"])
            self.length = int(self.subject_slices[subject]["length"])

        # 파일 경로들
        phase_dir = os.path.join(memmap_root, phase)
        self.paths = {
            "input_embeddings": os.path.join(phase_dir, "input_embeddings.npy"),
            "sent_level_EEG": os.path.join(phase_dir, "sent_level_EEG.npy"),
            "input_attn_mask": os.path.join(phase_dir, "input_attn_mask.npy"),
            "input_attn_mask_invert": os.path.join(phase_dir, "input_attn_mask_invert.npy"),
            "target_ids": os.path.join(phase_dir, "target_ids.npy"),
            "target_mask": os.path.join(phase_dir, "target_mask.npy"),
            "seq_len": os.path.join(phase_dir, "seq_len.npy"),
            "sentiment_label": os.path.join(phase_dir, "sentiment_label.npy"),
        }

        # lazy open (DataLoader worker spawn/fork 안전)
        self._opened = False
        self._arrays = {}

    def _open_if_needed(self):
        if self._opened:
            return
        # np.load(..., mmap_mode='r') => memmap
        self._arrays["input_embeddings"] = np.load(self.paths["input_embeddings"], mmap_mode=self.mmap_mode)
        self._arrays["sent_level_EEG"] = np.load(self.paths["sent_level_EEG"], mmap_mode=self.mmap_mode)
        self._arrays["input_attn_mask"] = np.load(self.paths["input_attn_mask"], mmap_mode=self.mmap_mode)
        self._arrays["input_attn_mask_invert"] = np.load(self.paths["input_attn_mask_invert"], mmap_mode=self.mmap_mode)
        self._arrays["target_ids"] = np.load(self.paths["target_ids"], mmap_mode=self.mmap_mode)
        self._arrays["target_mask"] = np.load(self.paths["target_mask"], mmap_mode=self.mmap_mode)
        self._arrays["seq_len"] = np.load(self.paths["seq_len"], mmap_mode=self.mmap_mode)
        self._arrays["sentiment_label"] = np.load(self.paths["sentiment_label"], mmap_mode=self.mmap_mode)
        self._opened = True

    def __len__(self):
        return self.length

    def get(self, local_idx: int):
        """
        local_idx: [0, length)
        반환 포맷은 ZuCo_dataset.__getitem__ 과 동일.
        """
        self._open_if_needed()
        gidx = self.start + int(local_idx)

        def _to_torch(arr):
            """Convert numpy/memmap row to a torch Tensor safely.

            If arrays are opened with np.load(..., mmap_mode='r'), slices are typically
            non-writable. torch.from_numpy on a non-writable array emits a warning because
            in-place writes would be undefined. We avoid the warning by copying only the
            requested row (NOT the whole dataset).
            """
            arr = np.asarray(arr)
            if not arr.flags.writeable:
                arr = arr.copy()
            return torch.from_numpy(arr)

        emb = _to_torch(self._arrays["input_embeddings"][gidx])
        if self.noise:
            emb = torch.randn_like(emb, dtype=emb.dtype)
            
        sent_eeg = _to_torch(self._arrays["sent_level_EEG"][gidx])
        attn = _to_torch(self._arrays["input_attn_mask"][gidx])
        attn_inv = _to_torch(self._arrays["input_attn_mask_invert"][gidx])
        tgt_ids = _to_torch(self._arrays["target_ids"][gidx])
        tgt_mask = _to_torch(self._arrays["target_mask"][gidx])

        seq_len = int(self._arrays["seq_len"][gidx])
        sent_lab = torch.tensor(int(self._arrays["sentiment_label"][gidx]), dtype=torch.long)

        return emb, seq_len, attn, attn_inv, tgt_ids, tgt_mask, sent_lab, sent_eeg


class ZuCoNpyDataset(Dataset):
    """
    메모리 절감형 ZuCo Dataset.

    - converter가 만든 memmap .npy들을 mmap_mode='r'로 열어,
      __getitem__ 시점에 필요한 row만 읽어옵니다.
    - 따라서 기존 ZuCo_dataset처럼 self.inputs에 전체를 적재하지 않습니다.

    사용:
      ds = ZuCoNpyDataset(
          ["/mnt/.../zuco_decoding_npy/task1-SR", "/mnt/.../task2-NR"],
          phase="train",
          tokenizer=tokenizer,
          subject="ALL",
          eeg_type="GD",
          bands=[...],
          setting="unique_sent",
          is_add_CLS_token=False,
          test_input="eeg",
          max_len=56,
          mmap_eeg=True,
          require_meta=True,
      )
    """
    def __init__(
        self,
        npy_dataset_paths,
        phase,
        tokenizer,
        subject='ALL',
        eeg_type='GD',
        bands=['_t1','_t2','_a1','_a2','_b1','_b2','_g1','_g2'],
        setting='unique_sent',
        is_add_CLS_token=False,
        test_input='eeg',
        max_len=56,
        mmap_eeg=True,
        require_meta=True,
        float_dtype=None,
        **kwargs,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.phase = phase

        if isinstance(npy_dataset_paths, (str, os.PathLike)):
            npy_dataset_paths = [str(npy_dataset_paths)]
        else:
            npy_dataset_paths = [str(p) for p in npy_dataset_paths]

        mmap_mode = "r" if mmap_eeg else None

        expected = {
            "eeg_type": eeg_type,
            "bands": bands,
            "max_len": int(max_len),
            "add_CLS_token": bool(is_add_CLS_token),
            "test_input": test_input,
            "setting": setting,
        }

        # tokenizer 일치 검사 (target_ids/target_mask가 tokenizer에 종속)
        tok_name = getattr(tokenizer, "name_or_path", None)
        if tok_name is not None:
            expected["tokenizer_name"] = tok_name
        # float_dtype는 meta에 input_embeddings dtype으로 들어가는데, 저장된 dtype과 다르면 mismatch.
        # float_dtype를 강제하고 싶으면 converter에서 동일하게 설정하세요.
        # 여기서는 float_dtype가 주어졌을 때만 검사하도록 처리
        if float_dtype is not None:
            expected["dtypes"] = expected.get("dtypes", {})
            # meta['dtypes']['input_embeddings'] 와 맞추는 목적
            # (meta 전체 dtypes dict 비교는 너무 빡세서 여기선 생략)

        self.parts = []
        for task_dir in npy_dataset_paths:
            memmap_root = _resolve_memmap_root(task_dir)
            part = _ZuCoMemmapPart(
                memmap_root=memmap_root,
                phase=phase,
                subject=subject,
                expected=expected,
                mmap_mode=mmap_mode,
                require_meta=require_meta,
                noise = True if test_input == "noise" else False
            )
            self.parts.append(part)

        # cumulative lengths (concat)
        self._cum = []
        s = 0
        for p in self.parts:
            s += len(p)
            self._cum.append(s)

        if len(self) == 0:
            raise ValueError("Dataset length is 0. Check phase/subject/config or conversion results.")


    def __len__(self):
        return self._cum[-1] if self._cum else 0

    def _locate(self, idx: int):
        if idx < 0:
            idx = len(self) + idx
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)

        # find first cum > idx
        for part_i, end in enumerate(self._cum):
            if idx < end:
                start = 0 if part_i == 0 else self._cum[part_i - 1]
                local = idx - start
                return self.parts[part_i], local
        raise IndexError(idx)

    def __getitem__(self, idx):
        part, local_idx = self._locate(idx)
        return part.get(local_idx)
    
'''sanity test'''
if __name__ == '__main__':

    check_dataset = 'stanford_sentiment'

    if check_dataset == 'ZuCo':
        whole_dataset_dicts = []
        
        dataset_path_task1 = '/shared/nas/data/m1/wangz3/SAO_project/SAO/dataset/ZuCo/task1-SR/pickle/task1-SR-dataset-with-tokens_6-25.pickle' 
        with open(dataset_path_task1, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))

        dataset_path_task2 = '/shared/nas/data/m1/wangz3/SAO_project/SAO/dataset/ZuCo/task2-NR/pickle/task2-NR-dataset-with-tokens_7-10.pickle' 
        with open(dataset_path_task2, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))

        # dataset_path_task3 = '/shared/nas/data/m1/wangz3/SAO_project/SAO/dataset/ZuCo/task3-TSR/pickle/task3-TSR-dataset-with-tokens_7-10.pickle' 
        # with open(dataset_path_task3, 'rb') as handle:
        #     whole_dataset_dicts.append(pickle.load(handle))

        dataset_path_task2_v2 = '/shared/nas/data/m1/wangz3/SAO_project/SAO/dataset/ZuCo/task2-NR-2.0/pickle/task2-NR-2.0-dataset-with-tokens_7-15.pickle' 
        with open(dataset_path_task2_v2, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))

        print()
        for key in whole_dataset_dicts[0]:
            print(f'task2_v2, sentence num in {key}:',len(whole_dataset_dicts[0][key]))
        print()

        tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
        dataset_setting = 'unique_sent'
        subject_choice = 'ALL'
        print(f'![Debug]using {subject_choice}')
        eeg_type_choice = 'GD'
        print(f'[INFO]eeg type {eeg_type_choice}') 
        bands_choice = ['_t1','_t2','_a1','_a2','_b1','_b2','_g1','_g2'] 
        print(f'[INFO]using bands {bands_choice}')
        train_set = ZuCo_dataset(whole_dataset_dicts, 'train', tokenizer, subject = subject_choice, eeg_type = eeg_type_choice, bands = bands_choice, setting = dataset_setting)
        dev_set = ZuCo_dataset(whole_dataset_dicts, 'dev', tokenizer, subject = subject_choice, eeg_type = eeg_type_choice, bands = bands_choice, setting = dataset_setting)
        test_set = ZuCo_dataset(whole_dataset_dicts, 'test', tokenizer, subject = subject_choice, eeg_type = eeg_type_choice, bands = bands_choice, setting = dataset_setting)

        print('trainset size:',len(train_set))
        print('devset size:',len(dev_set))
        print('testset size:',len(test_set))

    elif check_dataset == 'stanford_sentiment':
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        SST_dataset = SST_tenary_dataset(SST_SENTIMENT_LABELS, tokenizer)
        print('SST dataset size:',len(SST_dataset))
        print(SST_dataset[0])
        print(SST_dataset[1])