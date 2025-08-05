import ast
import sys
import os
import re

if len(sys.argv) != 2:
    print("Usage: python extract_metrics_table.py your_file.txt")
    sys.exit(1)

file_path = sys.argv[1]
filename = os.path.basename(file_path).replace(".txt", "")

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
corpus_bleu_scores = [safe_eval(val, [0,0,0,0]) for val in extract_all_values(text, "corpus_bleu_score")]
rouge_scores_list = [safe_eval(val, {'rouge-1': {'p':0, 'r':0, 'f':0}}) for val in extract_all_values(text, "rouge_scores")]
corpus_bleu_scores_tf = [safe_eval(val, [0,0,0,0]) for val in extract_all_values(text, "corpus_bleu_score_with_tf")]
rouge_scores_tf_list = [safe_eval(val, {'rouge-1': {'p':0, 'r':0, 'f':0}}) for val in extract_all_values(text, "rouge_scores_with_tf")]

def pct(x): return round(x * 100, 2) if isinstance(x, (float, int)) else x

# 테이블 헤더에서 WER, CER 컬럼 삭제
print("| Model                | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | P    | R    | F    |")
print("|--------------------|--------|--------|--------|--------|------|------|------|")

num_blocks = max(len(corpus_bleu_scores), len(corpus_bleu_scores_tf))

for i in range(num_blocks):
    bleu_base = [pct(x) for x in corpus_bleu_scores[i]] if i < len(corpus_bleu_scores) else ["N/A"]*4
    bleu_tf = [pct(x) for x in corpus_bleu_scores_tf[i]] if i < len(corpus_bleu_scores_tf) else ["N/A"]*4
    rouge_base = [pct(rouge_scores_list[i]['rouge-1'].get(k,0)) for k in ['p', 'r', 'f']] if i < len(rouge_scores_list) else ["N/A"]*3
    rouge_tf = [pct(rouge_scores_tf_list[i]['rouge-1'].get(k,0)) for k in ['p', 'r', 'f']] if i < len(rouge_scores_tf_list) else ["N/A"]*3
    
    # WER, CER 관련 변수 계산 삭제

    model_tag = f"{filename}_{i+1}"
    
    # 출력 f-string에서 WER, CER 변수 삭제
    print(f"| {model_tag:<19} | {bleu_base[0]:<6} | {bleu_base[1]:<6} | {bleu_base[2]:<6} | {bleu_base[3]:<6} | {rouge_base[0]:<4} | {rouge_base[1]:<4} | {rouge_base[2]:<4} |")
    print(f"| {model_tag + ' w/tf':<19} | {bleu_tf[0]:<6} | {bleu_tf[1]:<6} | {bleu_tf[2]:<6} | {bleu_tf[3]:<6} | {rouge_tf[0]:<4} | {rouge_tf[1]:<4} | {rouge_tf[2]:<4} |")