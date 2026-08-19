#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ZuCo struct-npy store  --  pickle.load() 의 드롭인 대체.

원본 ZuCo pickle 은 dict[subject] -> list[sentence_obj] 형태의 중첩 구조라
np.save 로는 memmap 이 불가능하다(object dtype). 이 모듈이 읽는 struct-npy 포맷은
수치 페이로드를 몇 개의 큰 배열로 평탄화해 memmap 하고, 문자열/구조 정보만
작은 JSON 으로 들고 있다가, **문장 하나를 꺼내는 순간 원본과 동일한 dict 를 조립**한다.

    # before
    with open(pickle_path, 'rb') as f:
        dataset_dict = pickle.load(f)

    # after
    dataset_dict = load_zuco_npy(npy_root)

반환되는 객체는 진짜 dict (subject -> 문장 시퀀스) 이고,
dataset_dict[subj][i] 는 진짜 dict / None 이다. 따라서 그 이후의 코드
(get_input_sample, ZuCo_dataset 등)는 한 글자도 바뀌지 않는다.

포맷 스펙은 zuco_struct_spec.py 참조.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .zuco_struct_spec import (
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
    scalar_caster,
)

__all__ = ["load_zuco_npy", "ZuCoNpyStore", "LazySentenceList"]


class LazySentenceList(Sequence):
    """한 subject 의 문장 리스트. 인덱싱하는 순간 원본과 동일한 dict 를 조립해 돌려준다."""

    __slots__ = ("_store", "_start", "_count")

    def __init__(self, store: "ZuCoNpyStore", start: int, count: int):
        self._store = store
        self._start = int(start)
        self._count = int(count)

    def __len__(self) -> int:
        return self._count

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self[i] for i in range(*idx.indices(self._count))]
        if idx < 0:
            idx += self._count
        if not (0 <= idx < self._count):
            raise IndexError(f"sentence index out of range: {idx} (len={self._count})")
        return self._store._build_sentence(self._start + idx)

    def __repr__(self) -> str:  # pragma: no cover - 표시용
        return f"<LazySentenceList n={self._count}>"


class ZuCoNpyStore(dict):
    """subject -> LazySentenceList 인 진짜 dict. isinstance(x, dict) 도 True."""

    def __init__(self, root, mmap: bool = True, copy_raw: bool = True):
        super().__init__()
        self.root = Path(root)
        self.copy_raw = bool(copy_raw)

        with open(self.root / F_META, encoding="utf-8") as f:
            self.meta = json.load(f)
        if self.meta.get("format") != FORMAT_VERSION:
            raise ValueError(
                f"unexpected format {self.meta.get('format')!r}, expected {FORMAT_VERSION!r}"
            )

        mm = "r" if mmap else None
        self._sent_is_none = self._load(F_SENT_IS_NONE, mm)
        self._sent_eeg = self._load(F_SENT_EEG, mm)
        self._sent_eeg_valid = self._load(F_SENT_EEG_VALID, mm)
        self._word_slice = self._load(F_WORD_SLICE, mm)
        self._word_eeg = self._load(F_WORD_EEG, mm)
        self._word_eeg_valid = self._load(F_WORD_EEG_VALID, mm)
        self._nfixations = self._load(F_NFIXATIONS, mm)
        self._answer_eeg = self._load(F_ANSWER_EEG, mm)
        self._answer_eeg_valid = self._load(F_ANSWER_EEG_VALID, mm)
        self._answer_eeg_present = self._load(F_ANSWER_EEG_PRESENT, mm)
        self._raw_data = self._load(F_RAW_DATA, mm)
        self._raw_valid = self._load(F_RAW_VALID, mm)
        self._current_length = self._load(F_CURRENT_LENGTH, mm)

        with open(self.root / F_STRINGS, encoding="utf-8") as f:
            s = json.load(f)
        self._sentence_content = s["sentence_content"]
        self._word_content = s["word_content"]
        self._word_tokens = s.get("word_tokens", {})

        # 스칼라 필드를 원본 타입 그대로 복원하기 위한 캐스터
        self._cast_nfix = scalar_caster(self.meta.get("nfixations_type"))
        self._cast_curlen = scalar_caster(self.meta.get("current_length_type"))

        # 밴드 dtype: 그룹 안에서도 문장마다 섞여 있을 수 있다(ZuCo v1 preprocessed).
        # 섞인 그룹은 *_dtype.npy 코드 배열로 밴드마다 원본 dtype 을 되돌린다.
        self._band_dtypes = {
            g: [np.dtype(d) for d in v] for g, v in self.meta["band_dtypes"].items()
        }
        self._store_dt = {
            g: np.dtype(d) for g, d in self.meta["band_storage_dtype"].items()
        }
        # 결측 밴드의 원본 표현. v2 는 ndarray 가 아니라 빈 파이썬 list 다.
        self._empty_band = {
            g: (tuple(v["shape"]), v.get("container", "numpy.ndarray"))
            for g, v in self.meta["empty_band"].items()
        }
        self._dt_codes = {
            "sent": self._load(F_SENT_EEG_DTYPE, mm),
            "word": self._load(F_WORD_EEG_DTYPE, mm),
            "answer": self._load(F_ANSWER_EEG_DTYPE, mm),
        }

        for subj, sl in self.meta["subject_slices"].items():
            super().__setitem__(subj, LazySentenceList(self, sl["start"], sl["count"]))

    # ------------------------------------------------------------------ #

    def _load(self, name, mm):
        p = self.root / name
        if not p.exists():
            return None
        return np.load(p, mmap_mode=mm)

    def _band_dict(self, block, valid, keys, group, codes=None):
        """(B, 105) 블록 + 유효마스크 -> {'mean_t1': array(105,), ...}

        원본 키 순서와 밴드별 원본 dtype 을 그대로 복원한다.
        """
        out = {}
        dts = self._band_dtypes[group]
        shp, container = self._empty_band[group]
        uniform = None if codes is not None else dts[0]
        for bi, k in enumerate(keys):
            dt = uniform if uniform is not None else dts[int(codes[bi])]
            if valid[bi]:
                a = block[bi]
                out[k] = a if a.dtype == dt else a.astype(dt)
            elif container == "list":
                out[k] = []
            else:
                out[k] = np.empty(shp, dtype=dt)
        return out

    def _build_word(self, wi, blk, vblk, cblk):
        out = {}
        for k in self.meta["word_keys"]:
            if k == "content":
                out[k] = self._word_content[wi]
            elif k == "nFixations":
                out[k] = self._cast_nfix(self._nfixations[wi])
            elif k == "word_level_EEG":
                wl = {}
                for ei, et in enumerate(self.meta["eeg_types"]):
                    wl[et] = self._band_dict(
                        blk[ei], vblk[ei], self.meta["word_band_keys"][et], "word",
                        None if cblk is None else cblk[ei],
                    )
                out[k] = wl
            else:  # pragma: no cover - 변환기가 사전에 막는다
                raise KeyError(f"unhandled word key: {k}")
        return out

    def _build_sentence(self, gi):
        if bool(self._sent_is_none[gi]):
            return None

        ws, wc = (int(x) for x in self._word_slice[gi])
        # 문장 단위로 한 번에 읽어온다(memmap 접근 1회). 이후 슬라이스는 이 배열의 뷰.
        wblk = np.asarray(self._word_eeg[ws : ws + wc])
        wvld = np.asarray(self._word_eeg_valid[ws : ws + wc])
        wcod = self._dt_codes["word"]
        wcod = None if wcod is None else np.asarray(wcod[ws : ws + wc])
        sblk = np.asarray(self._sent_eeg[gi])
        svld = np.asarray(self._sent_eeg_valid[gi])

        out = {}
        for k in self.meta["sentence_keys"]:
            if k == "content":
                out[k] = self._sentence_content[gi]
            elif k == "sentence_level_EEG":
                sc = self._dt_codes["sent"]
                out[k] = self._band_dict(
                    sblk, svld, self.meta["sentence_level_EEG_keys"], "sent",
                    None if sc is None else sc[gi],
                )
            elif k == "answer_EEG":
                if self._answer_eeg is None or not bool(self._answer_eeg_present[gi]):
                    out[k] = None
                else:
                    out[k] = self._band_dict(
                        np.asarray(self._answer_eeg[gi]),
                        np.asarray(self._answer_eeg_valid[gi]),
                        self.meta["answer_EEG_keys"],
                        "answer",
                        None if self._dt_codes["answer"] is None else self._dt_codes["answer"][gi],
                    )
            elif k == "word":
                out[k] = [
                    self._build_word(
                        ws + j, wblk[j], wvld[j], None if wcod is None else wcod[j]
                    )
                    for j in range(wc)
                ]
            elif k == "rawData":
                if self._raw_data is None or not bool(self._raw_valid[gi]):
                    out[k] = None
                else:
                    out[k] = (
                        np.asarray(self._raw_data[gi]).copy()
                        if self.copy_raw
                        else self._raw_data[gi]
                    )
            elif k == "current_length":
                out[k] = self._cast_curlen(self._current_length[gi])
            elif k.startswith("word_tokens_"):
                out[k] = self._word_tokens[k][gi]
            else:  # pragma: no cover
                raise KeyError(f"unhandled sentence key: {k}")
        return out

    def __repr__(self) -> str:  # pragma: no cover - 표시용
        return (
            f"<ZuCoNpyStore task={self.meta.get('task')} kind={self.meta.get('kind')} "
            f"subjects={len(self)} sentences={self.meta.get('n_sentences')}>"
        )


def load_zuco_npy(root, mmap: bool = True, copy_raw: bool = True) -> ZuCoNpyStore:
    """struct-npy 디렉터리를 열어 pickle.load() 와 동일하게 쓸 수 있는 dict 를 돌려준다.

    Args:
        root: <...>/preprocessed/task1-SR 처럼 meta.json 이 있는 디렉터리.
        mmap: True 면 수치 배열을 memmap 으로 연다(기본). False 면 전부 RAM 에 올린다.
        copy_raw: raw EEG 를 문장마다 복사해서 돌려줄지 여부. False 면 memmap 뷰를
            그대로 돌려줘 복사가 없지만 읽기 전용이 된다.
    """
    return ZuCoNpyStore(root, mmap=mmap, copy_raw=copy_raw)
