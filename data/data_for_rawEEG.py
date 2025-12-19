import numpy as np
import torch
import pickle
from torch.utils.data import Dataset
from transformers import BartTokenizer

# macro
#ZUCO_SENTIMENT_LABELS = json.load(open('./dataset/ZuCo/task1-SR/sentiment_labels/sentiment_labels.json'))
#SST_SENTIMENT_LABELS = json.load(open('./dataset/stanfordsentiment/ternary_dataset.json'))

def normalize_1d(input_tensor):
    # normalize a 1d tensor
    mean = torch.mean(input_tensor)
    std = torch.std(input_tensor)
    input_tensor = (input_tensor - mean)/std
    return input_tensor 

def get_input_sample(sent_obj, tokenizer, bands = ['_t1','_t2','_a1','_a2','_b1','_b2','_g1','_g2'], max_len = 56):
   
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
        # print(f'  - skip bad sentence')   
        return None
    input_sample = {}
    # get target label
    target_string = sent_obj['content']
    target_tokenized = tokenizer(target_string, padding='max_length', max_length=max_len, truncation=True, return_tensors='pt', return_attention_mask = True)
    input_sample['target_ids'] = target_tokenized['input_ids'][0]
    input_sample['target_mask'] = target_tokenized['attention_mask'][0]
    # get sentence level EEG features
    sent_level_eeg_tensor = get_sent_eeg(sent_obj, bands)
    # try:
    #     sent_level_eeg_tensor = torch.from_numpy(sent_obj['sentence_level_EEG']) # This gives a dictionary
    # except:
    #     return None

    if torch.isnan(sent_level_eeg_tensor).any():
        # print('[NaN sent level eeg]: ', target_string)
        return None
    # if sent_level_eeg_tensor.shape[1] < 30:
    #     return None
    
    input_sample['sent_level_EEG'] = sent_level_eeg_tensor
    #input_sample['sent_level_EEG'] = torch.randn(sent_level_eeg_tensor.size()) # random input code
    #print("NOISE:", input_sample['sent_level_EEG'])

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
    # input_sample['sentiment_label'] = torch.tensor(-100) # dummy value

    # window = 100   # 200ms
    # stride = 50    # 100ms overlap
    # segments = []
    # num_segments = 56
    
    if sent_obj['rawData'] is None:
        return None
    else:
        # for i in range(num_segments):
        #     start = i * stride
        #     end   = start + window
        #     # 각 chunk는 shape이 [C, window]가 됨
        #     chunk = sent_obj['rawData'][:, start:end]  # x가 [C, total_time] 이므로
        #     segments.append(chunk)

        # segments 리스트에 [C, window] 짜리 조각들이 56개 쌓임
        # 이를 한 번에 합치면 [num_segments, C, window] => [56, 105, 100] (예시)
        # x = np.stack(segments, axis=0)
        # x = np.mean(x, axis = 1)

        # input_sample['rawEEG'] = np.transpose(np.array(segments), (1, 0, 2))
        input_sample['rawEEG'] = sent_obj['rawData']

    input_sample['input_attn_mask'] = torch.zeros(max_len) # 0 is masked out

    input_sample['input_attn_mask'][:len(sent_obj['word'])] = torch.ones(len(sent_obj['word'])) # 1 is not masked
    

    # mask out padding tokens reverted: handle different use case: this is for pytorch transformers
    input_sample['input_attn_mask_invert'] = torch.ones(max_len) # 1 is masked out

    input_sample['input_attn_mask_invert'][:len(sent_obj['word'])] = torch.zeros(len(sent_obj['word'])) # 0 is not masked



    return input_sample

class ZuCo_dataset(Dataset):
    def __init__(self, input_dataset_dicts, phase, tokenizer, subject = 'ALL',bands = ['_t1','_t2','_a1','_a2','_b1','_b2','_g1','_g2'], setting = 'unique_sent'):
        self.inputs = []
        self.tokenizer = tokenizer

        if not isinstance(input_dataset_dicts,list):
            input_dataset_dicts = [input_dataset_dicts]
        print(f'[INFO]loading {len(input_dataset_dicts)} task datasets')

        count = 0
        for input_dataset_dict in input_dataset_dicts:
            subjects = list(input_dataset_dict.keys())
            
            total_num_sentence = len(input_dataset_dict[subjects[0]])
            
            train_divider = int(0.8*total_num_sentence)
            dev_divider = train_divider + int(0.1*total_num_sentence)
            
            if setting == 'unique_sent':
                # take first 80% as trainset, 10% as dev and 10% as test
                if phase == 'train':
                    print('[INFO]initializing a train set...')
                    for key in subjects:
                        for i in range(train_divider):
                            input_sample = get_input_sample(input_dataset_dict[key][i],self.tokenizer,bands = bands)
                            if input_sample is not None:
                                self.inputs.append(input_sample)
                elif phase == 'dev':
                    print('[INFO]initializing a dev set...')
                    for key in subjects:
                        for i in range(train_divider,dev_divider):
                            input_sample = get_input_sample(input_dataset_dict[key][i],self.tokenizer,bands = bands)
                            if input_sample is not None:
                                self.inputs.append(input_sample)
                elif phase == 'test':
                    print('[INFO]initializing a test set...')
                    for key in subjects:
                        for i in range(dev_divider,total_num_sentence):
                            input_sample = get_input_sample(input_dataset_dict[key][i],self.tokenizer,bands = bands)
                            if input_sample is not None:
                                self.inputs.append(input_sample)
            elif setting == 'unique_subject':
                if subject == 'ALL':
                    raise ValueError('for unique_subject setting, please specify a subject name')
                test_subject = subject
                print(f'[INFO]using {test_subject} as test subject')
                
                if test_subject not in subjects:
                    count += 1
                    if phase == 'train':
                        for key in subjects:
                            for i in range(dev_divider):
                                input_sample = get_input_sample(input_dataset_dict[key][i],self.tokenizer,bands = bands)
                                if input_sample is not None:
                                    self.inputs.append(input_sample)
                    if phase == 'dev':
                        for key in subjects:
                            for i in range(dev_divider,total_num_sentence):
                                input_sample = get_input_sample(input_dataset_dict[key][i],self.tokenizer,bands = bands)
                                if input_sample is not None:
                                    self.inputs.append(input_sample)
                else:
                    subjects.remove(test_subject)
                    if phase == 'train':
                        for key in subjects:
                            for i in range(dev_divider):
                                input_sample = get_input_sample(input_dataset_dict[key][i],self.tokenizer,bands = bands)
                                if input_sample is not None:
                                    self.inputs.append(input_sample)
                    if phase == 'dev':
                        for key in subjects:
                            for i in range(dev_divider,total_num_sentence):
                                input_sample = get_input_sample(input_dataset_dict[key][i],self.tokenizer,bands = bands)
                                if input_sample is not None:
                                    self.inputs.append(input_sample)
                    if phase == 'test':
                        for i in range(total_num_sentence):
                                input_sample = get_input_sample(input_dataset_dict[test_subject][i],self.tokenizer,bands = bands)
                                if input_sample is not None:
                                    self.inputs.append(input_sample)
                                else:
                                    ValueError(f'no valid test sample for {test_subject}, please check your setting!')
            print('++ adding task to dataset, now we have:', len(self.inputs))
        if count > 3:
            raise ValueError(f'{test_subject} not found in any dataset, please check your setting!')

        # print('[INFO]input tensor size:', self.inputs[0]['rawEEG'].size())
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
        )
        # keys: input_embeddings, input_attn_mask, input_attn_mask_invert, target_ids, target_mask, 

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
        eeg_type_choice = 'GD'
        bands_choice = ['_t1','_t2','_a1','_a2','_b1','_b2','_g1','_g2'] 
        train_set = ZuCo_dataset(whole_dataset_dicts, 'train', tokenizer, subject = subject_choice, eeg_type = eeg_type_choice, bands = bands_choice, setting = dataset_setting)
        dev_set = ZuCo_dataset(whole_dataset_dicts, 'dev', tokenizer, subject = subject_choice, eeg_type = eeg_type_choice, bands = bands_choice, setting = dataset_setting)
        test_set = ZuCo_dataset(whole_dataset_dicts, 'test', tokenizer, subject = subject_choice, eeg_type = eeg_type_choice, bands = bands_choice, setting = dataset_setting)

        print('trainset size:',len(train_set))
        print('devset size:',len(dev_set))
        print('testset size:',len(test_set))