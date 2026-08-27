#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Raw EEG ZuCo pickle -> struct-npy 변환.

  /mnt/Members/sejin/ZuCo/<task>/<task>-dataset.pickle   (= EEGToText/DeWave/datasets/rawZuCo)
    -> <out_root>/raw/<task>/

raw pickle 의 문장 구조:
  {'content', 'current_length', 'rawData', 'sentence_level_EEG', 'word'}
  rawData 는 이미 (105, 5500) 고정 배열이다. 가변 길이가 아니므로 memmap 에 그대로 들어간다.
  current_length 는 패딩 전 실제 길이(5500 을 넘는 문장은 pickle 단계에서 이미 잘려 있다).

옛 convert_raw_pickle_to_memmap_npy.py 와의 차이:
  - BART 토크나이저를 쓰지 않고, train/dev/test 로 나누지도 않는다.
  - rawData 를 **원본 dtype 그대로** 저장한다. 옛 변환기는 float32 원본(task1-SR,
    task2-NR)을 float64 로 승격시켜 용량을 2배로 썼다. 정보 이득은 0이다.
  - word_level_EEG(FFD/GD/TRT), word content, nFixations, 문장 원문을 전부 보존한다.

사용:
  python convert_raw_to_npy.py --task task1-SR
  python convert_raw_to_npy.py --all
"""

from __future__ import annotations

import argparse
import importlib
import pickle
import sys
import time
from pathlib import Path

import numpy as np

if not hasattr(np, "_core"):  # pragma: no cover - 환경 의존
    import numpy.core as _npc

    sys.modules["numpy._core"] = _npc
    for _s in ["numeric", "multiarray", "umath", "_multiarray_umath", "numerictypes"]:
        try:
            sys.modules[f"numpy._core.{_s}"] = importlib.import_module(f"numpy.core.{_s}")
        except Exception:
            pass

from _struct_writer import scan, write  # noqa: E402

DEFAULT_PICKLE_ROOT = "/mnt/Members/sejin/ZuCo"
DEFAULT_OUT_ROOT = "/mnt/Members/speech/EEGToText/_data/zuco_npy"
TASKS = ["task1-SR", "task2-NR", "task2-NR-2.0"]


def pickle_path_for(task: str, pickle_root: str) -> Path:
    return Path(pickle_root) / task / f"{task}-dataset.pickle"


def convert_one(task: str, pickle_root: str, out_root: str) -> dict:
    src = pickle_path_for(task, pickle_root)
    dst = Path(out_root) / "raw" / task
    print(f"\n{'='*70}\n[TASK] {task} (raw)\n  src: {src} ({src.stat().st_size/1e9:.1f} GB)"
          f"\n  dst: {dst}\n{'='*70}")

    t0 = time.time()
    print("[1/3] pickle 로드 중 (수 분 걸린다) ...", flush=True)
    with open(src, "rb") as f:
        dd = pickle.load(f)
    print(f"      완료 ({time.time()-t0:.0f}s)  subjects={len(dd)}")

    t1 = time.time()
    print("[2/3] 구조 스캔 + 검증 ...", flush=True)
    info = scan(dd, kind="raw")
    print(
        f"      완료 ({time.time()-t1:.0f}s)  문장={info['n_sentences']} 단어={info['n_words']}\n"
        f"      sentence_keys={info['sentence_keys']}\n"
        f"      rawData: {info['raw_dtype']} {info['raw_shape']}\n"
        f"      band_dtypes={info['band_dtype']}  current_length={info['current_length_type']}"
    )

    t2 = time.time()
    print("[3/3] npy 기록 중 ...", flush=True)
    meta = write(dd, info, "raw", dst, str(src), free_as_you_go=True)
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
