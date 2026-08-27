#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
preprocessed ZuCo pickle -> struct-npy 변환.

  preprocessed_data/<task>/pickle/<task>-dataset.pickle
    -> <out_root>/preprocessed/<task>/

옛 convert_zuco_pickle_to_memmap_npy.py 와의 차이:
  - BART 토크나이저를 쓰지 않는다. target_ids 를 굽지 않고 원문 content 를 그대로 둔다.
  - train/dev/test 로 나누지 않는다. split 은 예전처럼 ZuCo_dataset 이 런타임에 한다.
  - FFD / TRT / answer_EEG / nFixations / word_tokens_* 를 전부 보존한다.
  - 따라서 max_len·eeg_type·토크나이저를 바꿔도 재변환이 필요 없다.

사용:
  python convert_preprocessed_to_npy.py --task task1-SR
  python convert_preprocessed_to_npy.py --all
"""

from __future__ import annotations

import argparse
import importlib
import pickle
import sys
import time
from pathlib import Path

import numpy as np

# numpy>=2 로 저장된 pickle 을 numpy 1.x 에서도 읽기 위한 alias shim
if not hasattr(np, "_core"):  # pragma: no cover - 환경 의존
    import numpy.core as _npc

    sys.modules["numpy._core"] = _npc
    for _s in ["numeric", "multiarray", "umath", "_multiarray_umath", "numerictypes"]:
        try:
            sys.modules[f"numpy._core.{_s}"] = importlib.import_module(f"numpy.core.{_s}")
        except Exception:
            pass

from _struct_writer import scan, write  # noqa: E402

DEFAULT_PICKLE_ROOT = "/mnt/Members/speech/preprocessed_data"
DEFAULT_OUT_ROOT = "/mnt/Members/speech/EEGToText/_data/zuco_npy"
TASKS = ["task1-SR", "task2-NR", "task3-TSR", "task2-NR-2.0"]


def pickle_path_for(task: str, pickle_root: str) -> Path:
    return Path(pickle_root) / task / "pickle" / f"{task}-dataset.pickle"


def convert_one(task: str, pickle_root: str, out_root: str) -> dict:
    src = pickle_path_for(task, pickle_root)
    dst = Path(out_root) / "preprocessed" / task
    print(f"\n{'='*70}\n[TASK] {task}\n  src: {src}\n  dst: {dst}\n{'='*70}")

    t0 = time.time()
    print("[1/3] pickle 로드 중 ...", flush=True)
    with open(src, "rb") as f:
        dd = pickle.load(f)
    print(f"      완료 ({time.time()-t0:.0f}s)  subjects={len(dd)}")

    t1 = time.time()
    print("[2/3] 구조 스캔 + 검증 ...", flush=True)
    info = scan(dd, kind="preprocessed")
    print(
        f"      완료 ({time.time()-t1:.0f}s)  문장={info['n_sentences']} 단어={info['n_words']}\n"
        f"      sentence_keys={info['sentence_keys']}\n"
        f"      eeg_types={info['eeg_types']}  bands={info['n_bands']}\n"
        f"      band_dtypes={info['band_dtype']}  nFixations={info['nfixations_type']}\n"
        f"      answer_EEG={info['has_answer_eeg']}  empty_band={info['empty_band'] or '없음'}"
    )

    t2 = time.time()
    print("[3/3] npy 기록 중 ...", flush=True)
    meta = write(dd, info, "preprocessed", dst, str(src), free_as_you_go=True)
    print(f"      완료 ({time.time()-t2:.0f}s)")

    total = sum(p.stat().st_size for p in dst.iterdir() if p.is_file())
    print(f"[DONE] {task}: {total/1e9:.2f} GB  (원본 pickle {src.stat().st_size/1e9:.2f} GB)")
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=TASKS)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--pickle_root", default=DEFAULT_PICKLE_ROOT)
    ap.add_argument("--out_root", default=DEFAULT_OUT_ROOT)
    args = ap.parse_args()

    tasks = TASKS if args.all else ([args.task] if args.task else [])
    if not tasks:
        ap.error("--task 또는 --all 중 하나가 필요하다")

    for t in tasks:
        convert_one(t, args.pickle_root, args.out_root)
    print("\n[ALL DONE]")


if __name__ == "__main__":
    main()
