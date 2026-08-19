import os
import numpy as np
import h5py
import data_loading_helpers_modified as dh
from glob import glob
from tqdm import tqdm
import pickle

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

task = "NR"

# rootdir = "/home/saul_park/workspace/data/ZuCov2/task1 - NR/Matlab files/"
rootdir = "/home/saul_park/workspace/data/ZuCo/ZuCov2/task1-NR/Matlabfiles/"

print('##############################')
print(f'start processing ZuCo task2-NR-2.0...')

dataset_dict = {}

for file in tqdm(os.listdir(rootdir)):
    if file.endswith(task+".mat"):

        file_name = rootdir + file

        # print('file name:', file_name)
        subject = file_name.split("ts")[1].split("_")[0]
        # print('subject: ', subject)

        # exclude YMH due to incomplete data because of dyslexia
        if subject != 'YMH':
            assert subject not in dataset_dict
            dataset_dict[subject] = []

            f = h5py.File(file_name,'r')
            # print('keys in f:', list(f.keys()))
            sentence_data = f['sentenceData']
            # print('keys in sentence_data:', list(sentence_data.keys()))
            
            # sent level eeg 
            # mean_t1 = np.squeeze(f[sentence_data['mean_t1'][0][0]][()])
            mean_t1_objs = sentence_data['mean_t1']
            mean_t2_objs = sentence_data['mean_t2']
            mean_a1_objs = sentence_data['mean_a1']
            mean_a2_objs = sentence_data['mean_a2']
            mean_b1_objs = sentence_data['mean_b1']
            mean_b2_objs = sentence_data['mean_b2']
            mean_g1_objs = sentence_data['mean_g1']
            mean_g2_objs = sentence_data['mean_g2']
            
            rawData = sentence_data['rawData']
            contentData = sentence_data['content']
            wordData = sentence_data['word']

            for idx in range(len(rawData)):
                # get sentence string
                obj_reference_content = contentData[idx][0]
                sent_string = dh.load_matlab_string(f[obj_reference_content])
                # print('sentence string:', sent_string)
                
                sent_obj = {'content':sent_string}

                # sent_obj['rawData'] = np.squeeze(f[rawData[idx][0]][()]).T
                obj_reference_rawData = sentence_data['rawData'][idx][0]
                sent_rawEEG = np.array([r for r in f[obj_reference_rawData]])

                sent_obj['rawData'] = clip_or_pad_eeg(sent_rawEEG.T)
                
                # get sentence level EEG
                sent_obj['sentence_level_EEG'] = {
                    'mean_t1':np.squeeze(f[mean_t1_objs[idx][0]][()]), 
                    'mean_t2':np.squeeze(f[mean_t2_objs[idx][0]][()]), 
                    'mean_a1':np.squeeze(f[mean_a1_objs[idx][0]][()]), 
                    'mean_a2':np.squeeze(f[mean_a2_objs[idx][0]][()]), 
                    'mean_b1':np.squeeze(f[mean_b1_objs[idx][0]][()]), 
                    'mean_b2':np.squeeze(f[mean_b2_objs[idx][0]][()]), 
                    'mean_g1':np.squeeze(f[mean_g1_objs[idx][0]][()]), 
                    'mean_g2':np.squeeze(f[mean_g2_objs[idx][0]][()])
                }

                sent_obj['word'] = []
                
                word_data, word_tokens_all, word_tokens_has_fixation, word_tokens_with_mask = dh.extract_word_level_data(f, f[wordData[idx][0]])
                if word_data == {}:
                    print(f'missing sent: subj:{subject} content:{sent_string}, append None')
                    dataset_dict[subject].append(None)
                    continue
                elif len(word_tokens_all) == 0:
                    print(f'no word level features: subj:{subject} content:{sent_string}, append None')
                    dataset_dict[subject].append(None)
                    continue

                else:                    
                    for widx in range(len(word_data)):
                        data_dict = word_data[widx]
                        word_obj = {'content':data_dict['content'], 'nFixations': data_dict['nFix']}
                        if 'GD_EEG' in data_dict:
                            # print('has fixation: ', data_dict['content'])
                            gd = data_dict["GD_EEG"]
                            ffd = data_dict["FFD_EEG"]
                            trt = data_dict["TRT_EEG"]
                            assert len(gd) == len(trt) == len(ffd) == 8
                            word_obj['word_level_EEG'] = {
                                'GD':{'GD_t1':gd[0], 'GD_t2':gd[1], 'GD_a1':gd[2], 'GD_a2':gd[3], 'GD_b1':gd[4], 'GD_b2':gd[5], 'GD_g1':gd[6], 'GD_g2':gd[7]},
                                'FFD':{'FFD_t1':ffd[0], 'FFD_t2':ffd[1], 'FFD_a1':ffd[2], 'FFD_a2':ffd[3], 'FFD_b1':ffd[4], 'FFD_b2':ffd[5], 'FFD_g1':ffd[6], 'FFD_g2':ffd[7]},
                                'TRT':{'TRT_t1':trt[0], 'TRT_t2':trt[1], 'TRT_a1':trt[2], 'TRT_a2':trt[3], 'TRT_b1':trt[4], 'TRT_b2':trt[5], 'TRT_g1':trt[6], 'TRT_g2':trt[7]}
                            }
                            sent_obj['word'].append(word_obj)
                dataset_dict[subject].append(sent_obj)

"""output"""
task_name = 'task2-NR-2.0'

if dataset_dict == {}:
    print(f'No mat file found for {task_name}')
    quit()

output_dir = f'./dataset/ZuCo/raw_EEG/{task_name}/pickle'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

output_name = f'{task_name}-dataset.pickle'
# with open(os.path.join(output_dir,'task1-SR-dataset.json'), 'w') as out:
#     json.dump(dataset_dict,out,indent = 4)

with open(os.path.join(output_dir,output_name), 'wb') as handle:
    pickle.dump(dataset_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('write to:', os.path.join(output_dir,output_name))

"""sanity check"""
print('subjects:', dataset_dict.keys())
print('num of sent:', len(dataset_dict['YAC']))