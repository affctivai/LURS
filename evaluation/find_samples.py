import glob
import os

# 대상 폴더 경로
folder_path = './results_0723/'  # 원하는 폴더 경로로 바꿔주세요

# 해당 폴더 내의 *_results.txt로 끝나는 모든 파일 검색
file_pattern = os.path.join(folder_path, '*_results.txt')
# file_pattern = os.path.join(folder_path, '*.txt')
matching_files = glob.glob(file_pattern)

import os
import glob
import evaluate  # Hugging Face evaluate 라이브러리
from transformers import BartTokenizer
import heapq

evaluate_bleu = evaluate.load("bleu")
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

def hf_tokenizer(text: str):
    # special-token 제거 옵션(add_special_tokens=False)은 중요
    return tokenizer.tokenize(text)

for file_path in matching_files:
    with open(file_path, 'r') as f:
        content = f.read()
    target = []
    prediction = []
    prediction_tf = []

    bleu_score = {}
    print(f'File: {file_path.split("/")[-1].split("_results.txt")[0]}')
    j=1
    evaluate_bleu_score_tf = 0
    evaluate_bleu_score = 0
    for t in content.splitlines():
        if t.split(':')[0] == 'target string':
            target.append(t.split(':')[1])
        elif t.split(':')[0] == 'predicted string with tf':
            prediction_tf.append(t.split(':')[1])
        elif t.split(':')[0] == 'predicted string':
            prediction.append(t.split(':')[1])
            for i in range(1, 5):
                score = evaluate_bleu.compute(
                    predictions=[prediction_tf[-1]],    # List[str]
                    references=[target[-1]],     # List[List[str]]
                    tokenizer=hf_tokenizer,    # 공백 기반 토크나이저 사용
                    max_order=i,
                    smooth=True
                )["bleu"]
                evaluate_bleu_score_tf += score

            for i in range(1, 5):
                score = evaluate_bleu.compute(
                    predictions=[prediction[-1]],      # List[str]
                    references=[target[-1]],     # List[List[str]]
                    tokenizer=hf_tokenizer,    # 공백 기반 토크나이저 사용
                    max_order=i,
                    smooth=True
                )["bleu"]
                evaluate_bleu_score +=score
            bleu_score[j] = evaluate_bleu_score/4 + evaluate_bleu_score_tf/4

            evaluate_bleu_score_tf = 0
            evaluate_bleu_score = 0
        j += 1

    # data.get을 키로 사용하여 값이 가장 큰 3개의 키를 찾습니다.
    best_keys = heapq.nlargest(30, bleu_score, key=bleu_score.get)

    print(f'Best BLEU least 30 keys: {best_keys}')



    print("=" * 30 + "\n")
