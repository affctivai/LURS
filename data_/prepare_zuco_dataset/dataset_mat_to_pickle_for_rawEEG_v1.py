import scipy.io as io
import h5py
import os
import json
from glob import glob
from tqdm import tqdm
import numpy as np
import pickle
import argparse

def clip_or_pad_eeg(eeg_data, target_length=5500):
    """
    eeg_data: NumPy 배열, shape = [num_channels, time]
    target_length: 원하는 최대/최소 시간 길이 (기본 5500)
    
    반환: shape = [num_channels, target_length]
    """
    # EEG 데이터의 현재 길이(시간 축)
    current_length = eeg_data.shape[1]

    if current_length > target_length:
        # 지정된 길이를 초과할 경우 앞부분부터 잘라냄
        eeg_data = eeg_data[:, :target_length]
    elif current_length < target_length:
        # 지정된 길이에 미치지 못하면 뒤쪽에 0으로 패딩
        pad_length = target_length - current_length
        eeg_data = np.pad(
            eeg_data,
            pad_width=((0, 0), (0, pad_length)),
            mode='constant',
            constant_values=0
        )

    return eeg_data
parser = argparse.ArgumentParser(description='Specify task name for converting ZuCo v1.0 Mat file to Pickle')
parser.add_argument('-t', '--task_name', help='name of the task in /dataset/ZuCo, choose from {task1-SR,task2-NR,task3-TSR}', required=True)
args = vars(parser.parse_args())


"""config"""
version = 'v1' # 'old'
# version = 'v2' # 'new'

task_name = args['task_name']
# task_name = 'task1-SR'
# task_name = 'task2-NR'
# task_name = 'task3-TSR'


print('##############################')
print(f'start processing ZuCo {task_name}...')


if version == 'v1':
    # old version 
    input_mat_files_dir = f'/home/saul_park/workspace/data/ZuCo/ZuCov1/{task_name}/Matlabfiles/' 
elif version == 'v2':
    # new version, mat73 
    input_mat_files_dir = f'/home/saul_park/workspace/data/ZuCo/ZuCov1/{task_name}/Matlabfiles/' 

output_dir = f'./dataset/ZuCo/raw_EEG/{task_name}/pickle'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

"""load files"""
mat_files = glob(os.path.join(input_mat_files_dir,'*.mat'))
mat_files = sorted(mat_files)

if len(mat_files) == 0:
    print(f'No mat files found for {task_name}')
    quit()

dataset_dict = {}
for mat_file in tqdm(mat_files):
    subject_name = os.path.basename(mat_file).split('_')[0].replace('results','').strip()
    dataset_dict[subject_name] = []
    
    if version == 'v1':
        matdata = io.loadmat(mat_file, squeeze_me=True, struct_as_record=False)['sentenceData']
    elif version == 'v2':
        matdata = h5py.File(mat_file,'r')
        print(matdata)

    for sent in matdata:
        word_data = sent.word
        if not isinstance(word_data, float):
            # sentence level:
            sent_obj = {'content':sent.content}
            sent_obj['sentence_level_EEG'] = {'mean_t1':sent.mean_t1, 'mean_t2':sent.mean_t2, 'mean_a1':sent.mean_a1, 'mean_a2':sent.mean_a2, 'mean_b1':sent.mean_b1, 'mean_b2':sent.mean_b2, 'mean_g1':sent.mean_g1, 'mean_g2':sent.mean_g2}

            if  not isinstance(sent.rawData, float):
                rawData = clip_or_pad_eeg(sent.rawData)
                sent_obj['rawData'] = rawData
            else:
                sent_obj['rawData'] = None

            sent_obj['word'] = []
            word_tokens_all = []
            for word in word_data:
                word_obj = {'content':word.content}
                word_tokens_all.append(word.content)
                # TODO: add more version of word level eeg: GD, SFD, GPT
                word_obj['nFixations'] = word.nFixations
                if word.nFixations > 0:    
                    word_obj['word_level_EEG'] = {'FFD':{'FFD_t1':word.FFD_t1, 'FFD_t2':word.FFD_t2, 'FFD_a1':word.FFD_a1, 'FFD_a2':word.FFD_a2, 'FFD_b1':word.FFD_b1, 'FFD_b2':word.FFD_b2, 'FFD_g1':word.FFD_g1, 'FFD_g2':word.FFD_g2}}
                    word_obj['word_level_EEG']['TRT'] = {'TRT_t1':word.TRT_t1, 'TRT_t2':word.TRT_t2, 'TRT_a1':word.TRT_a1, 'TRT_a2':word.TRT_a2, 'TRT_b1':word.TRT_b1, 'TRT_b2':word.TRT_b2, 'TRT_g1':word.TRT_g1, 'TRT_g2':word.TRT_g2}
                    word_obj['word_level_EEG']['GD'] = {'GD_t1':word.GD_t1, 'GD_t2':word.GD_t2, 'GD_a1':word.GD_a1, 'GD_a2':word.GD_a2, 'GD_b1':word.GD_b1, 'GD_b2':word.GD_b2, 'GD_g1':word.GD_g1, 'GD_g2':word.GD_g2}
                    sent_obj['word'].append(word_obj)
                else:
                    # if a word has no fixation, use sentence level feature
                    # word_obj['word_level_EEG'] = {'FFD':{'FFD_t1':sent.mean_t1, 'FFD_t2':sent.mean_t2, 'FFD_a1':sent.mean_a1, 'FFD_a2':sent.mean_a2, 'FFD_b1':sent.mean_b1, 'FFD_b2':sent.mean_b2, 'FFD_g1':sent.mean_g1, 'FFD_g2':sent.mean_g2}}
                    # word_obj['word_level_EEG']['TRT'] = {'TRT_t1':sent.mean_t1, 'TRT_t2':sent.mean_t2, 'TRT_a1':sent.mean_a1, 'TRT_a2':sent.mean_a2, 'TRT_b1':sent.mean_b1, 'TRT_b2':sent.mean_b2, 'TRT_g1':sent.mean_g1, 'TRT_g2':sent.mean_g2}
                    
                    # NOTE:if a word has no fixation, simply skip it
                    continue
            
            dataset_dict[subject_name].append(sent_obj)

        else:
            print(f'missing sent: subj:{subject_name} content:{sent.content}, return None')
            dataset_dict[subject_name].append(None)

            # continue
    # print(dataset_dict.keys())
    # print(dataset_dict[subject_name][0].keys())
    # print(dataset_dict[subject_name][0]['content'])
    # print(dataset_dict[subject_name][0]['word'][0].keys())
    # print(dataset_dict[subject_name][0]['word'][0]['word_level_EEG']['FFD'])

"""output"""
output_name = f'{task_name}-dataset.pickle'
# with open(os.path.join(output_dir,'task1-SR-dataset.json'), 'w') as out:
#     json.dump(dataset_dict,out,indent = 4)

with open(os.path.join(output_dir,output_name), 'wb') as handle:
    pickle.dump(dataset_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('write to:', os.path.join(output_dir,output_name))


"""sanity check"""
# check dataset
with open(os.path.join(output_dir,output_name), 'rb') as handle:
    whole_dataset = pickle.load(handle)
print('subjects:', whole_dataset.keys())

if version == 'v1':
    print('num of sent:', len(whole_dataset['ZAB']))
    print()

