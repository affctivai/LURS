#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
struct-npy 가 원본 pickle 과 **완전히 동일한지** 깊은 비교로 검증한다.

비교 대상:
  - dict 키 구성과 **순서**
  - list 길이
  - ndarray 의 dtype / shape / 전 원소 값 (NaN 위치까지)
  - 스칼라의 파이썬 타입과 값
  - None 여부

사용:
  python verify_struct_npy.py --kind preprocessed --task task1-SR            # 전수
  python verify_struct_npy.py --kind raw --task task2-NR --sample 300        # 표본
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from loader import load_zuco_npy  # noqa: E402

PICKLE_ROOTS = {
    "preprocessed": "/mnt/Members/speech/preprocessed_data",
    "raw": "/mnt/Members/sejin/ZuCo",
}
DEFAULT_NPY_ROOT = "/mnt/Members/speech/EEGToText/_data/zuco_npy"


def pickle_path_for(kind: str, task: str) -> Path:
    root = Path(PICKLE_ROOTS[kind])
    if kind == "preprocessed":
        return root / task / "pickle" / f"{task}-dataset.pickle"
    return root / task / f"{task}-dataset.pickle"


def deep_diff(a, b, path="") -> list:
    """a(원본) 와 b(npy) 의 차이를 문자열 리스트로 돌려준다. 빈 리스트면 완전 동일."""
    if a is None or b is None:
        if a is not b:
            return [f"{path}: None 불일치 (pickle={a!r}, npy={b!r})"]
        return []

    if isinstance(a, dict):
        if not isinstance(b, dict):
            return [f"{path}: 타입 불일치 dict vs {type(b).__name__}"]
        ka, kb = list(a.keys()), list(b.keys())
        if ka != kb:
            return [f"{path}: dict 키/순서 불일치\n    pickle={ka}\n    npy   ={kb}"]
        out = []
        for k in ka:
            out += deep_diff(a[k], b[k], f"{path}[{k!r}]")
            if len(out) > 20:
                return out
        return out

    if isinstance(a, (list, tuple)):
        if not isinstance(b, (list, tuple)):
            return [f"{path}: 타입 불일치 list vs {type(b).__name__}"]
        if len(a) != len(b):
            return [f"{path}: 길이 불일치 {len(a)} vs {len(b)}"]
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out += deep_diff(x, y, f"{path}[{i}]")
            if len(out) > 20:
                return out
        return out

    if isinstance(a, np.ndarray):
        if not isinstance(b, np.ndarray):
            return [f"{path}: 타입 불일치 ndarray vs {type(b).__name__}"]
        if a.dtype != b.dtype:
            return [f"{path}: dtype 불일치 {a.dtype} vs {b.dtype}"]
        if a.shape != b.shape:
            return [f"{path}: shape 불일치 {a.shape} vs {b.shape}"]
        if a.size == 0:
            return []
        if np.issubdtype(a.dtype, np.floating):
            same = np.array_equal(a, b, equal_nan=True)
        else:
            same = np.array_equal(a, b)
        if not same:
            d = np.abs(a.astype(np.float64) - b.astype(np.float64))
            return [f"{path}: 값 불일치 (최대 절대오차 {np.nanmax(d):.3e})"]
        return []

    if type(a) is not type(b):
        return [f"{path}: 스칼라 타입 불일치 {type(a).__name__} vs {type(b).__name__}"]
    if a != b:
        return [f"{path}: 값 불일치 {a!r} vs {b!r}"]
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=["preprocessed", "raw"])
    ap.add_argument("--task", required=True)
    ap.add_argument("--npy_root", default=DEFAULT_NPY_ROOT)
    ap.add_argument("--sample", type=int, default=0, help="0 이면 전수 비교")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    src = pickle_path_for(args.kind, args.task)
    npy = Path(args.npy_root) / args.kind / args.task

    print(f"[VERIFY] {args.kind}/{args.task}\n  pickle: {src}\n  npy   : {npy}")
    store = load_zuco_npy(npy)

    t0 = time.time()
    print("  pickle 로드 중 ...", flush=True)
    with open(src, "rb") as f:
        dd = pickle.load(f)
    print(f"  완료 ({time.time()-t0:.0f}s)")

    subs_p, subs_n = list(dd.keys()), list(store.keys())
    if subs_p != subs_n:
        print(f"  [FAIL] subject 순서/구성 불일치\n    pickle={subs_p}\n    npy   ={subs_n}")
        return 1
    print(f"  subjects {len(subs_p)} 일치")

    pairs = [(s, i) for s in subs_p for i in range(len(dd[s]))]
    if len(store[subs_p[0]]) != len(dd[subs_p[0]]):
        print("  [FAIL] 문장 개수 불일치")
        return 1

    if args.sample and args.sample < len(pairs):
        rng = np.random.default_rng(args.seed)
        pick = rng.choice(len(pairs), size=args.sample, replace=False)
        pairs = [pairs[i] for i in sorted(pick)]
        print(f"  표본 {len(pairs)} 문장 비교")
    else:
        print(f"  전수 {len(pairs)} 문장 비교")

    t1 = time.time()
    n_bad = 0
    n_none = 0
    for n, (s, i) in enumerate(pairs):
        a, b = dd[s][i], store[s][i]
        if a is None:
            n_none += 1
        diffs = deep_diff(a, b, f"{s}[{i}]")
        if diffs:
            n_bad += 1
            print(f"  [DIFF] {s}[{i}]")
            for d in diffs[:5]:
                print(f"     {d}")
            if n_bad >= 5:
                print("  ... 차이가 5건을 넘어 중단")
                break
        if (n + 1) % 2000 == 0:
            print(f"    {n+1}/{len(pairs)} ... ({time.time()-t1:.0f}s)", flush=True)

    print(f"\n  비교 완료 ({time.time()-t1:.0f}s)  None 문장 {n_none}개 포함")
    if n_bad == 0:
        print(f"==== RESULT: PASS — npy 가 pickle 과 완전히 동일 ({args.kind}/{args.task}) ====")
        return 0
    print(f"==== RESULT: FAIL — {n_bad} 문장 불일치 ====")
    return 1


if __name__ == "__main__":
    sys.exit(main())
