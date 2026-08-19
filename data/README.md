# loader — ZuCo struct-npy 데이터 로더

`pickle.load()` 한 줄만 바꾸면 되는 드롭인 로더. **로드 이후에는 pickle 을 쓸 때와
완전히 동일하다** — `dataset_dict[subj][i]` 는 진짜 `dict` 이고, dict 키 순서,
ndarray dtype/shape, 스칼라 파이썬 타입, `None` 여부까지 원본과 같다.

```
zuco_struct_spec.py   포맷 스펙 (writer/reader 공용 단일 정의)
zuco_npy_store.py     load_zuco_npy — pickle.load 대체
data.py               preprocessed 로더 (baseline_original/data.py 기준)
data_raw.py           raw EEG 로더
```

## 쓰는 법

```python
import sys; sys.path.insert(0, '/mnt/Members/speech/EEGToText')

# ---- preprocessed ----
from loader.data import ZuCo_dataset, get_input_sample, load_preprocessed

whole_dataset_dicts = [load_preprocessed(t) for t in ['task1-SR', 'task2-NR']]
train_set = ZuCo_dataset(whole_dataset_dicts, 'train', tokenizer,
                         eeg_type='GD', bands=bands, setting='unique_sent')

# ---- raw EEG ----
from loader.data_raw import ZuCo_dataset as RawZuCo_dataset, load_raw

raw_dicts = [load_raw('task1-SR')]
train_set = RawZuCo_dataset(raw_dicts, 'train', tokenizer, bands=bands)
```

기존 코드에 끼워 넣을 때는 이 한 줄만 바꾸면 된다:

```python
# before
with open(pickle_path, 'rb') as f:
    whole_dataset_dicts.append(pickle.load(f))
# after
whole_dataset_dicts.append(load_preprocessed('task1-SR'))
```

## 옵션

```python
load_preprocessed('task1-SR', mmap=True)    # 기본. False 면 전부 RAM 에 올린다
load_raw('task1-SR', copy_raw=False)        # rawData 를 memmap 뷰로 (문장당 2.3MB 복사 회피, 읽기 전용)
```

raw 로더의 `segment` 인자는 rawEEG 표현을 고른다.

```python
get_input_sample(..., segment=False)  # 기본, baseline_original: rawEEG = rawData (105, 5500)
get_input_sample(..., segment=True)   # DeWave: window/stride 로 56조각 → (105, 56, 100)
```

## 동작 원리

수치 배열은 memmap 이라 열어도 RAM 을 거의 안 쓴다. lazy 한 것은 두 층
(`dataset_dict` 와 `dataset_dict[subj]`) 뿐이고, **문장 하나를 꺼내는 순간부터는
100% 순정 `dict`/`list`/`ndarray`/`str`** 이다. 따라서 `isinstance(x, dict)`,
`.keys()`, `copy.deepcopy`, 값 수정, `pickle.dumps` 전부 원본과 똑같이 동작한다.

pickle 대비 유일한 표면적 차이: `dataset_dict[subj]` 는 `list` 가 아니라
`Sequence` 를 구현한 `LazySentenceList` 다. `len()`, `[i]`, `[a:b]`, 순회는 모두
같지만 `isinstance(x, list)` 만 `False` 다.

## 검증

`../Make_npy/new/verify_struct_npy.py` 가 원본 pickle 과 깊은 비교(키 순서·dtype·
전 원소 값·스칼라 타입)를 한다. preprocessed 4개 task 전수(19,566 문장) PASS.
