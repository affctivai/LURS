#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ZuCo struct-npy 로더.

    from loader import load_zuco_npy, load_preprocessed, load_raw

    # pickle.load() 를 그대로 대체한다
    dataset_dict = load_preprocessed('task1-SR')
    sent = dataset_dict['ZAB'][0]        # 원본 pickle 과 완전히 동일한 dict

    # baseline_original 과 동일한 Dataset
    from loader.data import ZuCo_dataset, get_input_sample
    from loader.data_raw import ZuCo_dataset as RawZuCo_dataset
"""

from .zuco_npy_store import LazySentenceList, ZuCoNpyStore, load_zuco_npy
from .zuco_struct_spec import FORMAT_VERSION

__all__ = [
    "load_zuco_npy",
    "ZuCoNpyStore",
    "LazySentenceList",
    "FORMAT_VERSION",
    "load_preprocessed",
    "load_preprocessed_many",
    "load_raw",
    "load_raw_many",
]


def load_preprocessed(*a, **kw):
    """loader.data.load_preprocessed 지연 import (torch 없이도 store 만 쓸 수 있게)."""
    from .data import load_preprocessed as _f

    return _f(*a, **kw)


def load_preprocessed_many(*a, **kw):
    from .data import load_preprocessed_many as _f

    return _f(*a, **kw)


def load_raw(*a, **kw):
    from .data_raw import load_raw as _f

    return _f(*a, **kw)


def load_raw_many(*a, **kw):
    from .data_raw import load_raw_many as _f

    return _f(*a, **kw)
