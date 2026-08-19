#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
struct-npy 포맷 스펙 (writer/reader 공용 단일 정의).

디렉터리 하나 = task 하나. 학습 split(train/dev/test)은 저장하지 않는다.
원본 pickle 과 "로드 이후 완전히 동일"한 것이 목표이므로, split 은 예전처럼
ZuCo_dataset 이 런타임에 나눈다.

파일 목록
---------
meta.json              구조 정보(키 순서, dtype, subject slice ...). 아래 참조.
strings.json           문자열 전부(문장 content, 단어 content, word_tokens_*).
sent_is_none.npy       (S,)            bool   문장 객체가 None 이었는지
sent_eeg.npy           (S, B, 105)     f8     sentence_level_EEG
sent_eeg_valid.npy     (S, B)          bool
sent_eeg_dtype.npy     (S, B)          u1     밴드별 원본 dtype 인덱스(섞여 있을 때만)
word_slice.npy         (S, 2)          i8     문장 -> 단어 배열 [start, count]
word_eeg.npy           (W, E, B, 105)  f8     word_level_EEG[eeg_type][band]
word_eeg_valid.npy     (W, E, B)       bool   False 면 원본이 빈 배열(shape (0,))
nfixations.npy         (W,)            f8     nFixations (원본 타입은 meta 에 기록)

preprocessed 전용
answer_eeg.npy         (S, B, 105)     f8
answer_eeg_valid.npy   (S, B)          bool   밴드별 유효성
answer_eeg_present.npy (S,)            bool   answer_EEG 자체가 None 이 아니었는지

raw 전용
raw_data.npy           (S, C, T)       f4/f8  rawData (dtype 은 원본 그대로)
raw_valid.npy          (S,)            bool   rawData 가 None 이었는지
current_length.npy     (S,)            i8

meta.json 주요 필드
-------------------
format, kind('preprocessed'|'raw'), task, source_pickle, subjects,
subject_slices {subj: {start, count}}, n_sentences, n_words,
sentence_keys / word_keys / sentence_level_EEG_keys / answer_EEG_keys /
eeg_types / word_band_keys   <- 전부 **원본 dict 의 실제 키 순서**
nfixations_type, current_length_type, empty_band_shape, empty_band_dtype,
dtypes {...}
"""

from __future__ import annotations

import numpy as np

FORMAT_VERSION = "zuco_struct_npy_v1"

F_META = "meta.json"
F_STRINGS = "strings.json"
F_SENT_IS_NONE = "sent_is_none.npy"
F_SENT_EEG = "sent_eeg.npy"
F_SENT_EEG_VALID = "sent_eeg_valid.npy"
F_WORD_SLICE = "word_slice.npy"
F_WORD_EEG = "word_eeg.npy"
F_WORD_EEG_VALID = "word_eeg_valid.npy"
F_NFIXATIONS = "nfixations.npy"
F_SENT_EEG_DTYPE = "sent_eeg_dtype.npy"
F_WORD_EEG_DTYPE = "word_eeg_dtype.npy"
F_ANSWER_EEG_DTYPE = "answer_eeg_dtype.npy"
F_ANSWER_EEG = "answer_eeg.npy"
F_ANSWER_EEG_VALID = "answer_eeg_valid.npy"
F_ANSWER_EEG_PRESENT = "answer_eeg_present.npy"
F_RAW_DATA = "raw_data.npy"
F_RAW_VALID = "raw_valid.npy"
F_CURRENT_LENGTH = "current_length.npy"

# 채널 수 / 밴드 수는 데이터에서 읽어 meta 에 기록한다(하드코딩하지 않음).
DEFAULT_BAND_DIM = 105


def type_name(value) -> str:
    """스칼라 값의 복원용 타입 이름."""
    t = type(value)
    if t.__module__ in (None, "builtins"):
        return t.__name__
    return f"{t.__module__}.{t.__name__}"


def scalar_caster(name):
    """meta 에 기록된 타입 이름 -> 값 복원 함수.

    numpy 스칼라(np.int64 등)와 파이썬 int/float 를 구분해 원본 타입 그대로 되돌린다.
    """
    if name is None:
        return lambda v: v
    if name == "int":
        return lambda v: int(v)
    if name == "float":
        return lambda v: float(v)
    if name == "bool":
        return lambda v: bool(v)
    if name.startswith("numpy."):
        np_t = getattr(np, name.split(".", 1)[1], None)
        if np_t is not None:
            return lambda v: np_t(v)
    return lambda v: v
