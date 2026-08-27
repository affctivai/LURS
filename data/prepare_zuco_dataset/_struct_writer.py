#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ZuCo pickle -> struct-npy 변환 공용 코어.

설계 원칙
---------
1. **무손실**: 원본 pickle 의 모든 필드를 보존한다. 전처리도, 토크나이저도 개입하지 않는다.
   따라서 max_len / eeg_type / 토크나이저를 바꿔도 재변환이 필요 없다.
2. **구조 보존**: 로드 이후 dataset_dict[subj][i] 는 원본과 완전히 동일한 dict 다.
   dict 키 순서, ndarray dtype/shape, 스칼라 파이썬 타입까지 그대로 복원한다.
3. **자기 검증**: 규칙성을 가정하지 않고 1-pass 로 스캔해 검증한다. 예상 밖 구조가
   나오면 조용히 뭉개지 않고 즉시 실패한다.

수치 페이로드는 몇 개의 큰 배열로 평탄화되어 memmap 가능해지고, 문자열만
작은 JSON 으로 남는다. 이것이 pickle 의 RAM 문제를 푸는 유일한 방법이다
(np.save(dict, allow_pickle=True) 는 object dtype 이라 mmap_mode 가 거부된다).
"""

from __future__ import annotations

import gc
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2]))  # EEGToText/ 를 import 경로에 추가
from loader.zuco_struct_spec import (  # noqa: E402
    FORMAT_VERSION,
    F_ANSWER_EEG,
    F_ANSWER_EEG_DTYPE,
    F_ANSWER_EEG_PRESENT,
    F_ANSWER_EEG_VALID,
    F_CURRENT_LENGTH,
    F_META,
    F_NFIXATIONS,
    F_RAW_DATA,
    F_RAW_VALID,
    F_SENT_EEG,
    F_SENT_EEG_DTYPE,
    F_SENT_EEG_VALID,
    F_SENT_IS_NONE,
    F_STRINGS,
    F_WORD_EEG,
    F_WORD_EEG_DTYPE,
    F_WORD_EEG_VALID,
    F_WORD_SLICE,
    type_name,
)

BAND_DIM = 105


class StructError(RuntimeError):
    """예상 밖의 pickle 구조. 조용히 뭉개지 말고 여기서 멈춘다."""


def _open_memmap(path: Path, dtype, shape):
    return np.lib.format.open_memmap(str(path), mode="w+", dtype=np.dtype(dtype), shape=shape)


def _is_empty(a) -> bool:
    return a.size == 0


# --------------------------------------------------------------------------- #
# Pass A : 구조 스캔 + 검증
# --------------------------------------------------------------------------- #
def scan(dataset_dict, kind: str) -> dict:
    subjects = list(dataset_dict.keys())
    if not subjects:
        raise StructError("빈 dataset_dict")

    info = {
        "subjects": subjects,
        "subject_slices": {},
        "sentence_keys": None,
        "sentence_level_EEG_keys": None,
        "answer_EEG_keys": None,
        "word_keys": None,
        "eeg_types": None,
        "word_band_keys": None,
        "nfixations_type": None,
        "current_length_type": None,
        # 밴드 배열 dtype 은 그룹(sent / answer / word)마다 다를 수 있다.
        # 예: task1-SR 은 sentence_level_EEG 가 float32, word_level_EEG 가 float64 다.
        "band_dtype": {},
        "empty_band": {},
        "raw_dtype": None,
        "raw_shape": None,
        "has_answer_eeg": False,
        "has_raw": kind == "raw",
        "word_token_keys": [],
    }

    n_sent = 0
    n_word = 0
    cursor = 0

    for subj in subjects:
        sents = dataset_dict[subj]
        info["subject_slices"][subj] = {"start": cursor, "count": len(sents)}
        cursor += len(sents)

        for si, sent in enumerate(sents):
            n_sent += 1
            if sent is None:
                continue

            skeys = list(sent.keys())
            if info["sentence_keys"] is None:
                info["sentence_keys"] = skeys
                info["word_token_keys"] = [k for k in skeys if k.startswith("word_tokens_")]
                info["has_answer_eeg"] = "answer_EEG" in skeys
                if kind == "raw":
                    for req in ("rawData", "current_length"):
                        if req not in skeys:
                            raise StructError(
                                f"raw 로 지정됐는데 '{req}' 키가 없다: {skeys}"
                            )
            elif skeys != info["sentence_keys"]:
                raise StructError(
                    f"{subj}[{si}] 문장 키 순서/구성이 다르다\n"
                    f"  expected={info['sentence_keys']}\n  got     ={skeys}"
                )

            # --- sentence_level_EEG ---
            se = sent["sentence_level_EEG"]
            if se is None:
                raise StructError(f"{subj}[{si}] sentence_level_EEG 가 None 이다")
            sek = list(se.keys())
            if info["sentence_level_EEG_keys"] is None:
                info["sentence_level_EEG_keys"] = sek
            elif sek != info["sentence_level_EEG_keys"]:
                raise StructError(f"{subj}[{si}] sentence_level_EEG 키 불일치: {sek}")
            for k in sek:
                info = _note_band(info, se[k], f"{subj}[{si}].sent[{k}]", "sent")

            # --- answer_EEG ---
            if info["has_answer_eeg"]:
                ae = sent["answer_EEG"]
                if ae is not None:
                    aek = list(ae.keys())
                    if info["answer_EEG_keys"] is None:
                        info["answer_EEG_keys"] = aek
                    elif aek != info["answer_EEG_keys"]:
                        raise StructError(f"{subj}[{si}] answer_EEG 키 불일치: {aek}")
                    for k in aek:
                        info = _note_band(info, ae[k], f"{subj}[{si}].ans[{k}]", "answer")

            # --- rawData / current_length ---
            if kind == "raw":
                rd = sent["rawData"]
                if rd is not None:
                    a = np.asarray(rd)
                    if info["raw_dtype"] is None:
                        info["raw_dtype"] = str(a.dtype)
                        info["raw_shape"] = list(a.shape)
                    elif str(a.dtype) != info["raw_dtype"] or list(a.shape) != info["raw_shape"]:
                        raise StructError(
                            f"{subj}[{si}] rawData 불일치: "
                            f"{a.dtype}{a.shape} vs {info['raw_dtype']}{info['raw_shape']}"
                        )
                cl = sent["current_length"]
                tn = type_name(cl)
                if info["current_length_type"] is None:
                    info["current_length_type"] = tn
                elif tn != info["current_length_type"]:
                    raise StructError(f"{subj}[{si}] current_length 타입 불일치: {tn}")

            # --- words ---
            words = sent["word"]
            n_word += len(words)
            for wi, w in enumerate(words):
                wk = list(w.keys())
                if info["word_keys"] is None:
                    info["word_keys"] = wk
                elif wk != info["word_keys"]:
                    raise StructError(f"{subj}[{si}].word[{wi}] 키 불일치: {wk}")

                tn = type_name(w["nFixations"])
                if info["nfixations_type"] is None:
                    info["nfixations_type"] = tn
                elif tn != info["nfixations_type"]:
                    raise StructError(f"{subj}[{si}].word[{wi}] nFixations 타입 불일치: {tn}")

                wl = w["word_level_EEG"]
                et = list(wl.keys())
                if info["eeg_types"] is None:
                    info["eeg_types"] = et
                    info["word_band_keys"] = {e: list(wl[e].keys()) for e in et}
                elif et != info["eeg_types"]:
                    raise StructError(f"{subj}[{si}].word[{wi}] eeg_type 불일치: {et}")
                for e in et:
                    bk = list(wl[e].keys())
                    if bk != info["word_band_keys"][e]:
                        raise StructError(f"{subj}[{si}].word[{wi}] {e} 밴드 키 불일치: {bk}")
                    for k in bk:
                        info = _note_band(
                            info, wl[e][k], f"{subj}[{si}].w{wi}.{e}[{k}]", "word"
                        )

    if info["sentence_keys"] is None:
        raise StructError("유효한 문장이 하나도 없다")

    nb = len(info["sentence_level_EEG_keys"])
    for e, bk in info["word_band_keys"].items():
        if len(bk) != nb:
            raise StructError(f"밴드 개수 불일치: sentence={nb}, {e}={len(bk)}")

    info["n_sentences"] = n_sent
    info["n_words"] = n_word
    info["n_bands"] = nb
    info["n_eeg_types"] = len(info["eeg_types"])
    return info


def _note_band(info, orig, where, group):
    """밴드 배열 하나를 보고 그룹별 dtype 목록/빈배열 형태를 기록·검증한다.

    ZuCo v1 preprocessed pickle 은 같은 그룹 안에서도 문장마다 dtype 이 섞여 있다
    (예: task1-SR 의 sentence_level_EEG 는 float32 문장과 float64 문장이 공존).
    그래서 dtype 을 하나로 강제하지 않고 관측된 목록을 모아 두었다가,
    저장은 가장 넓은 dtype 으로 하고 읽을 때 밴드별로 원래 dtype 으로 되돌린다.
    (float32 -> float64 -> float32 왕복은 bit-exact 다.)
    """
    a = np.asarray(orig)
    container = type_name(orig)
    dt = str(a.dtype)
    seen = info["band_dtype"].setdefault(group, [])
    if dt not in seen:
        seen.append(dt)

    if _is_empty(a):
        # ZuCo v2 의 결측 밴드는 ndarray 가 아니라 빈 파이썬 list 다. 컨테이너 종류까지
        # 기록해 두지 않으면 복원했을 때 list 가 ndarray 로 바뀌어 원본과 달라진다.
        cur = {"shape": list(a.shape), "container": container}
        prev = info["empty_band"].get(group)
        if prev is None:
            info["empty_band"][group] = cur
        elif cur != prev:
            raise StructError(f"{where} {group} 결측 밴드 표현 불일치: {cur} vs {prev}")
    else:
        if container != "numpy.ndarray":
            raise StructError(
                f"{where} {group} 비어 있지 않은 밴드가 ndarray 가 아니다: {container}"
            )
        if a.shape != (BAND_DIM,):
            raise StructError(
                f"{where} {group} 밴드 shape 이 (105,) 도 빈 배열도 아니다: {a.shape}"
            )
    return info


def storage_dtype(dtypes) -> str:
    """관측된 dtype 목록을 모두 무손실로 담을 수 있는 저장용 dtype."""
    if len(dtypes) == 1:
        return dtypes[0]
    widest = max(dtypes, key=lambda d: np.dtype(d).itemsize)
    for d in dtypes:
        if not np.can_cast(np.dtype(d), np.dtype(widest), casting="safe"):
            raise StructError(f"무손실로 합칠 수 없는 dtype 조합: {dtypes}")
    return widest


# --------------------------------------------------------------------------- #
# Pass B : 배열 채우기
# --------------------------------------------------------------------------- #
def write(dataset_dict, info, kind, out_dir: Path, source_pickle: str, free_as_you_go=True):
    out_dir.mkdir(parents=True, exist_ok=True)
    S, W = info["n_sentences"], info["n_words"]
    B, E = info["n_bands"], info["n_eeg_types"]
    # 그룹별 저장 dtype. 원본에 여러 dtype 이 섞여 있으면 가장 넓은 것으로 담고
    # 밴드별 원본 dtype 을 코드 배열에 따로 적어 둔다(읽을 때 그대로 되돌린다).
    dtl = {g: info["band_dtype"][g] for g in info["band_dtype"]}
    store_dt = {g: np.dtype(storage_dtype(v)) for g, v in dtl.items()}
    mixed = {g: len(v) > 1 for g, v in dtl.items()}
    code_of = {g: {d: i for i, d in enumerate(v)} for g, v in dtl.items()}

    sent_is_none = _open_memmap(out_dir / F_SENT_IS_NONE, bool, (S,))
    sent_eeg = _open_memmap(out_dir / F_SENT_EEG, store_dt["sent"], (S, B, BAND_DIM))
    sent_eeg_valid = _open_memmap(out_dir / F_SENT_EEG_VALID, bool, (S, B))
    sent_dt_mm = (
        _open_memmap(out_dir / F_SENT_EEG_DTYPE, np.uint8, (S, B))
        if mixed["sent"] else None
    )
    word_slice = _open_memmap(out_dir / F_WORD_SLICE, np.int64, (S, 2))
    word_eeg = _open_memmap(out_dir / F_WORD_EEG, store_dt["word"], (W, E, B, BAND_DIM))
    word_eeg_valid = _open_memmap(out_dir / F_WORD_EEG_VALID, bool, (W, E, B))
    word_dt_mm = (
        _open_memmap(out_dir / F_WORD_EEG_DTYPE, np.uint8, (W, E, B))
        if mixed["word"] else None
    )
    nfix = _open_memmap(out_dir / F_NFIXATIONS, np.float64, (W,))

    ans_eeg = ans_valid = ans_present = ans_dt_mm = None
    if info["has_answer_eeg"]:
        ans_eeg = _open_memmap(out_dir / F_ANSWER_EEG, store_dt["answer"], (S, B, BAND_DIM))
        ans_valid = _open_memmap(out_dir / F_ANSWER_EEG_VALID, bool, (S, B))
        ans_present = _open_memmap(out_dir / F_ANSWER_EEG_PRESENT, bool, (S,))
        if mixed["answer"]:
            ans_dt_mm = _open_memmap(out_dir / F_ANSWER_EEG_DTYPE, np.uint8, (S, B))

    raw_data = raw_valid = cur_len = None
    if kind == "raw":
        C, T = info["raw_shape"]
        raw_data = _open_memmap(out_dir / F_RAW_DATA, np.dtype(info["raw_dtype"]), (S, C, T))
        raw_valid = _open_memmap(out_dir / F_RAW_VALID, bool, (S,))
        cur_len = _open_memmap(out_dir / F_CURRENT_LENGTH, np.int64, (S,))

    sentence_content = [None] * S
    word_content = [None] * W
    word_tokens = {k: [None] * S for k in info["word_token_keys"]}

    gi = 0
    wcur = 0
    for subj in info["subjects"]:
        for sent in dataset_dict[subj]:
            if sent is None:
                sent_is_none[gi] = True
                word_slice[gi] = (wcur, 0)
                gi += 1
                continue

            sentence_content[gi] = sent["content"]

            se = sent["sentence_level_EEG"]
            for bi, k in enumerate(info["sentence_level_EEG_keys"]):
                a = np.asarray(se[k])
                if sent_dt_mm is not None:
                    sent_dt_mm[gi, bi] = code_of["sent"][str(a.dtype)]
                if _is_empty(a):
                    sent_eeg_valid[gi, bi] = False
                else:
                    sent_eeg[gi, bi] = a
                    sent_eeg_valid[gi, bi] = True

            if info["has_answer_eeg"]:
                ae = sent["answer_EEG"]
                if ae is None:
                    ans_present[gi] = False
                else:
                    ans_present[gi] = True
                    for bi, k in enumerate(info["answer_EEG_keys"]):
                        a = np.asarray(ae[k])
                        if ans_dt_mm is not None:
                            ans_dt_mm[gi, bi] = code_of["answer"][str(a.dtype)]
                        if _is_empty(a):
                            ans_valid[gi, bi] = False
                        else:
                            ans_eeg[gi, bi] = a
                            ans_valid[gi, bi] = True

            if kind == "raw":
                rd = sent["rawData"]
                if rd is None:
                    raw_valid[gi] = False
                else:
                    raw_valid[gi] = True
                    raw_data[gi] = np.asarray(rd)
                cur_len[gi] = int(sent["current_length"])

            for k in info["word_token_keys"]:
                word_tokens[k][gi] = sent[k]

            words = sent["word"]
            word_slice[gi] = (wcur, len(words))
            for w in words:
                word_content[wcur] = w["content"]
                nfix[wcur] = float(w["nFixations"])
                wl = w["word_level_EEG"]
                for ei, e in enumerate(info["eeg_types"]):
                    bd = wl[e]
                    for bi, k in enumerate(info["word_band_keys"][e]):
                        a = np.asarray(bd[k])
                        if word_dt_mm is not None:
                            word_dt_mm[wcur, ei, bi] = code_of["word"][str(a.dtype)]
                        if _is_empty(a):
                            word_eeg_valid[wcur, ei, bi] = False
                        else:
                            word_eeg[wcur, ei, bi] = a
                            word_eeg_valid[wcur, ei, bi] = True
                wcur += 1
            gi += 1

        if free_as_you_go:
            # raw pickle 은 수십 GB 라 subject 단위로 놓아준다
            dataset_dict[subj] = None
            gc.collect()

    if gi != S or wcur != W:
        raise StructError(f"채운 개수 불일치: sent {gi}/{S}, word {wcur}/{W}")

    for mm in (
        sent_is_none, sent_eeg, sent_eeg_valid, sent_dt_mm, word_slice, word_eeg,
        word_eeg_valid, word_dt_mm, nfix, ans_eeg, ans_valid, ans_present, ans_dt_mm,
        raw_data, raw_valid, cur_len,
    ):
        if mm is not None:
            mm.flush()

    with open(out_dir / F_STRINGS, "w", encoding="utf-8") as f:
        json.dump(
            {
                "sentence_content": sentence_content,
                "word_content": word_content,
                "word_tokens": word_tokens,
            },
            f,
            ensure_ascii=False,
        )

    meta = {
        "format": FORMAT_VERSION,
        "kind": kind,
        "source_pickle": os.path.abspath(source_pickle),
        "source_pickle_bytes": os.path.getsize(source_pickle),
        "created": datetime.now().isoformat(timespec="seconds"),
        "n_sentences": S,
        "n_words": W,
        "n_bands": B,
        "n_eeg_types": E,
        "band_dim": BAND_DIM,
        "subjects": info["subjects"],
        "subject_slices": info["subject_slices"],
        "sentence_keys": info["sentence_keys"],
        "sentence_level_EEG_keys": info["sentence_level_EEG_keys"],
        "answer_EEG_keys": info["answer_EEG_keys"],
        "word_keys": info["word_keys"],
        "eeg_types": info["eeg_types"],
        "word_band_keys": info["word_band_keys"],
        "word_token_keys": info["word_token_keys"],
        "nfixations_type": info["nfixations_type"],
        "current_length_type": info["current_length_type"],
        # 그룹별 밴드 dtype. 목록이 2개 이상이면 *_dtype.npy 코드 배열이 함께 있다.
        "band_dtypes": dtl,
        "band_storage_dtype": {g: str(v) for g, v in store_dt.items()},
        "band_dtype_mixed": mixed,
        # 결측 밴드의 표현(shape 와 컨테이너 종류). dtype 은 밴드별 코드로 결정된다.
        "empty_band": {
            g: info["empty_band"].get(g, {"shape": [0], "container": "numpy.ndarray"})
            for g in dtl
        },
        "has_answer_eeg": info["has_answer_eeg"],
        "has_raw": info["has_raw"],
        "raw_dtype": info["raw_dtype"],
        "raw_shape": info["raw_shape"],
        "dtypes": {
            "nfixations": "float64",
            "raw_data": info["raw_dtype"],
        },
    }
    with open(out_dir / F_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return meta
