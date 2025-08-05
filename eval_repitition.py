from nltk.tokenize import word_tokenize
import os
import glob
import sys

def create_ngram(word_list, n):
  created_ngrams = [word_list[i:i+n] for i in range(len(word_list) - n + 1)]
  
  return [' '.join([vi for vi in created_ngram]) for created_ngram in created_ngrams]

if len(sys.argv) != 2:
    print("❌ 사용법: python evaluate.py \"input_dir1 input_dir2\" output_dir")
    sys.exit(1)

file_path = sys.argv[1]
 
with open(file_path, 'r') as f:
    content = f.read()
target = []
prediction = []
prediction_tf = []

unique_target = []
unique_list = []
unique_list_tf = []
predicted_string_total_num = 0
predicted_string_tf_total_num = 0

unique_unigram = set()
unique_unigram_tf = set()

unique_vigram = set()
unique_vigram_tf = set()
unique_3gram = set()
unique_3gram_tf = set()
unique_4gram = set()
unique_4gram_tf = set()

for t in content.splitlines():
    
    if t.split(':')[0] == 'target string':
        text = t.split(':')[1]
        target.append(text)
        if text not in unique_target:
            unique_target.append(text)
    elif t.split(':')[0] == 'predicted string with tf':
        text = t.split(':')[1]
        prediction_tf.append(text)
        unigram_tf = word_tokenize(text)
        vigram_tf = create_ngram(unigram_tf, 2)
        three_gram_tf = create_ngram(unigram_tf, 3)
        four_gram_tf = create_ngram(unigram_tf, 4)

        for uni in unigram_tf:    
            unique_unigram_tf.add(uni)
        for vi in vigram_tf:
            unique_vigram_tf.add(vi)
        for three in three_gram_tf:
            unique_3gram_tf.add(three)
        for four in four_gram_tf:
            unique_4gram_tf.add(four)
        
        predicted_string_tf_total_num += len(unigram_tf)


        if text not in unique_list_tf:
            unique_list_tf.append(text)
    elif t.split(':')[0] == 'predicted string':
        text = t.split(':')[1]
        prediction.append(text)
        unigram = word_tokenize(text)
        unigram = word_tokenize(text)
        vigram = create_ngram(unigram, 2)
        three_gram = create_ngram(unigram, 3)
        four_gram = create_ngram(unigram, 4)
        for uni in unigram:
            unique_unigram.add(uni)
        for vi in vigram:
            unique_vigram.add(vi)
        for three in three_gram:
            unique_3gram.add(three)
        for four in four_gram:
            unique_4gram.add(four)

        predicted_string_total_num += len(unigram)

        if text not in unique_list:
            unique_list.append(text)


target_counts = [0] * len(unique_target)
prediction_counts = [0] * len(unique_list)
prediction_tf_counts = [0] * len(unique_list_tf)
for i in range(len(target)):
    target_counts[unique_target.index(target[i])] += 1
    prediction_counts[unique_list.index(prediction[i])] += 1
    prediction_tf_counts[unique_list_tf.index(prediction_tf[i])] += 1

def line(out):
    out.write('\n'+'\n')
    out.write('-'* 30+'\n')
    out.write('\n'+'\n')

output_path = 'results/'  + file_path.split("/")[-1].split("_results.txt")[0] + '_repitition.txt'

with open(output_path, 'w', encoding='utf-8') as out:
    out.write(f'File: {file_path.split("/")[-1].split("_results.txt")[0]}\n')
    out.write('-'*50+'\n')
    out.write(f'Total targets: {len(target)}\n')
    out.write(f'unique target: {len(unique_target)}\n')
    out.write('-'*50+'\n')
    out.write(f'Total predictions: {len(prediction)}\n')
    out.write(f'unique prediction: {len(unique_list)}\n')
    out.write('-'*50+'\n')
    out.write(f'Total predictions with tf: {len(prediction_tf)}\n')
    out.write(f'unique prediction with tf: {len(unique_list_tf)}\n')
    out.write('='*50+'\n')

    out.write(f'Target counts:{round(sum(target_counts)/len(target_counts), 2)}\n')
    out.write(f'Prediction counts:{round(sum(prediction_counts)/len(prediction_counts), 2)}\n')
    out.write(f'Prediction with tf counts:{round(sum(prediction_tf_counts)/len(prediction_tf_counts), 2)}\n')
    line(out)

    out.write(f'unigram repetition rate: {round(len(unique_unigram)/predicted_string_total_num * 100, 2)}\n')
    out.write(f'vigram repetition rate: {round(len(unique_vigram)/predicted_string_total_num * 100, 2)}\n')
    out.write(f'trigram repetition rate: {round(len(unique_3gram)/predicted_string_total_num * 100, 2)}\n')
    out.write(f'fourgram repetition rate: {round(len(unique_4gram)/predicted_string_total_num * 100, 2)}\n')
    line(out)

    out.write(f'unigram_tf repetition rate: {round(len(unique_unigram_tf)/predicted_string_tf_total_num * 100, 2)}\n')
    out.write(f'vigram_tf repetition rate: {round(len(unique_vigram_tf)/predicted_string_tf_total_num * 100, 2)}\n')
    out.write(f'trigram_tf repetition rate: {round(len(unique_3gram_tf)/predicted_string_tf_total_num * 100, 2)}\n')
    out.write(f'fourgram_tf repetition rate: {round(len(unique_4gram_tf)/predicted_string_tf_total_num * 100, 2)}\n')