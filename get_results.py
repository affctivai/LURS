import ast
import sys
import os
import re
from nltk.tokenize import word_tokenize

def extract_metrics_from_file(file_path):
    
    filename = os.path.basename(file_path).replace(".txt", "")

    base_dir = os.path.dirname(file_path)
    target_dir = os.path.join(base_dir, "..", "results")

    additional_file_name = os.path.join(target_dir, f"{filename}.txt")

    with open(file_path, "r") as f:
        text = f.read()

    def extract_all_values(text, key):
        pattern = rf'{key}\s*=\s*(.*?)(?=\n\S|$)'
        matches = re.findall(pattern, text, re.DOTALL)
        if not matches:
            print(f"Warning: {key} not found.")
        return [m.strip() for m in matches]

    def safe_eval(val, default=None):
        try:
            return ast.literal_eval(val)
        except (ValueError, SyntaxError):
            print(f"Warning: Could not parse value: {val[:50]}... -> Using default: {default}")
            return default

    # WER, CER 관련 줄 삭제
    bleu_base = [safe_eval(val, [0,0,0,0]) for val in extract_all_values(text, "corpus_bleu_score")][0]
    rouge_base = [safe_eval(val, {'rouge-1': {'p':0, 'r':0, 'f':0}}) for val in extract_all_values(text, "rouge_scores")][0]
    bleu_tf = [safe_eval(val, [0,0,0,0]) for val in extract_all_values(text, "corpus_bleu_score_with_tf")][0]
    rouge_tf = [safe_eval(val, {'rouge-1': {'p':0, 'r':0, 'f':0}}) for val in extract_all_values(text, "rouge_scores_with_tf")][0]

    model_tag = f"{filename}"

    def pct(x): return round(x * 100, 2) if isinstance(x, (float, int)) else x
    
    bleu_base = [pct(x) for x in bleu_base]
    bleu_tf = [pct(x) for x in bleu_tf]
    rouge_base = [pct(rouge_base['rouge-1'].get(k,0)) for k in ['p', 'r', 'f']]
    rouge_tf = [pct(rouge_tf['rouge-1'].get(k,0)) for k in ['p', 'r', 'f']]



    def create_ngram(word_list, n):
        """
        주어진 리스트를 바이그램(bi-gram)으로 변환하는 함수입니다.

        Args:
            word_list: 단어(토큰)들이 담긴 리스트

        Returns:
            바이그램으로 구성된 리스트 (e.g., [['단어1', '단어2'], ['단어2', '단어3']])
        """
        # zip 함수를 사용하여 현재 요소와 다음 요소를 짝지어줍니다.
        # return [list(pair) for pair in zip(word_list, word_list[1:])]
        created_ngrams = [word_list[i:i+n] for i in range(len(word_list) - n + 1)]
        
        return [' '.join([vi for vi in created_ngram]) for created_ngram in created_ngrams]

    # 각 파일을 열어서 작업 수행
    with open(additional_file_name, 'r') as f:
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

    print(f'File: {model_tag}')
    print('-'*50)
    print(f'Total targets: {len(target)}')
    print(f'unique target: {len(unique_target)}')
    print('-'*50)
    print(f'Total predictions: {len(prediction)}')
    print(f'unique prediction: {len(unique_list)}')
    print('-'*50)
    print(f'Total predictions with tf: {len(prediction_tf)}')
    print(f'unique prediction with tf: {len(unique_list_tf)}')
    print('='*50)

    target_counts = [0] * len(unique_target)
    prediction_counts = [0] * len(unique_list)
    prediction_tf_counts = [0] * len(unique_list_tf)
    for i in range(len(target)):
        target_counts[unique_target.index(target[i])] += 1
        prediction_counts[unique_list.index(prediction[i])] += 1
        prediction_tf_counts[unique_list_tf.index(prediction_tf[i])] += 1

    def pct(x): return round(x * 100, 2) if isinstance(x, (float, int)) else x

    rep = round(sum(prediction_counts)/len(prediction_counts), 2)
    rep_tf = round(sum(prediction_tf_counts)/len(prediction_tf_counts), 2)

    diversity = []
    diversity_tf = []

    diversity.append(pct(len(unique_unigram)/predicted_string_total_num))
    diversity.append(pct(len(unique_vigram)/predicted_string_total_num))
    diversity.append(pct(len(unique_3gram)/predicted_string_total_num))
    diversity.append(pct(len(unique_4gram)/predicted_string_total_num))

    diversity_tf.append(pct(len(unique_unigram_tf)/predicted_string_tf_total_num))
    diversity_tf.append(pct(len(unique_vigram_tf)/predicted_string_tf_total_num))
    diversity_tf.append(pct(len(unique_3gram_tf)/predicted_string_tf_total_num))
    diversity_tf.append(pct(len(unique_4gram_tf)/predicted_string_tf_total_num))
   
 
    return model_tag, bleu_base, bleu_tf, rouge_base, rouge_tf, diversity, diversity_tf, rep, rep_tf

if len(sys.argv) != 2:
    print("Usage: python extract_metrics_table.py your_file.txt or folder/")
    sys.exit(1)


file_path = sys.argv[1]

import time

def write_results_header(file_path, f):
    model_tag, bleu_base, bleu_tf, rouge_base, rouge_tf, diversity, diversity_tf, rep, rep_tf= extract_metrics_from_file(file_path)
    try:
        f.write(f"{model_tag},{bleu_base[0]},{bleu_base[1]},{bleu_base[2]},{bleu_base[3]},{rouge_base[0]},{rouge_base[1]},{rouge_base[2]}, {diversity[0]}, {diversity[1]}, {diversity[2]}, {diversity[3]}, {rep}\n")
        f.write(f"{model_tag} w_tf,{bleu_tf[0]},{bleu_tf[1]},{bleu_tf[2]},{bleu_tf[3]},{rouge_tf[0]},{rouge_tf[1]},{rouge_tf[2]}, {diversity_tf[0]}, {diversity_tf[1]}, {diversity_tf[2]}, {diversity_tf[3]}, {rep_tf}\n")
    except Exception as e:
        print(f"Error writing results for {filename}: {e}")
        pass


with open(f"results_{time.time()}.csv", "w", encoding="utf-8") as f:
    f.write("Model,BLEU-1,BLEU-2,BLEU-3,BLEU-4,P,R,F,Diversity-1,Diversity-2,Diversity-3,Diversity-4,REP\n")

    if os.path.isfile(file_path):
        try:
            write_results_header(file_path, f)
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            sys.exit(1)
        
    elif os.path.isdir(file_path):
        for filename in os.listdir(file_path):
            if filename.endswith(".txt"):
                full_path = os.path.join(file_path, filename)
                try:
                    write_results_header(full_path, f)
                except Exception as e:
                    print(f"Error processing file {filename}: {e}")
                    continue
    else:
        print("Error: The provided path is neither a file nor a directory.")
        sys.exit(1)


