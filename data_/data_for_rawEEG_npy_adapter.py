import os
import json
from bisect import bisect_right
from typing import List, Optional, Sequence, Union

import numpy as np
import torch
from torch.utils.data import Dataset


def _resolve_task_root(task_root: str) -> str:
    meta_here = os.path.join(task_root, "meta.json")
    memmap_meta = os.path.join(task_root, "memmap_v1", "meta.json")

    if os.path.isfile(meta_here):
        return task_root
    if os.path.isfile(memmap_meta):
        return os.path.join(task_root, "memmap_v1")

    raise FileNotFoundError(
        f"Could not find meta.json under:\n"
        f"  {meta_here}\n"
        f"  {memmap_meta}"
    )


class _SingleTaskNPYDataset(Dataset):

    def __init__(
        self,
        task_root: str,
        phase: str,
        tokenizer,
        subject: str = "ALL",
        eeg_type: str = "GD",
        bands: Optional[Sequence[str]] = None,
        setting: str = "unique_sent",
        seed: int = 42,
        noise: bool = False,
    ):
        super().__init__()
        self.task_root = _resolve_task_root(task_root)
        self.phase = phase
        self.tokenizer = tokenizer
        self.subject = subject
        self.eeg_type = eeg_type
        self.bands = list(bands) if bands is not None else None
        self.setting = setting
        self.noise = noise

        meta_path = os.path.join(self.task_root, "meta.json")
        with open(meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)

        subject_key = subject
        if subject_key == "ALL" and setting != "unique_sent":
            subject_key = setting

        if subject_key == "ALL" or subject_key == setting:
            self._loso = False
            phase_dir = os.path.join(self.task_root, phase)
            if not os.path.isdir(phase_dir):
                raise FileNotFoundError(f"Missing phase directory: {phase_dir}")

            raw_path = os.path.join(phase_dir, "rawEEG.npy")
            old_path = os.path.join(phase_dir, "input_embeddings.npy")
            if os.path.isfile(raw_path):
                self.raw_eeg = np.load(raw_path, mmap_mode="r")
            elif os.path.isfile(old_path):
                self.raw_eeg = np.load(old_path, mmap_mode="r")
            else:
                raise FileNotFoundError(
                    f"Neither rawEEG.npy nor input_embeddings.npy found in {phase_dir}"
                )

            self.sent_level_eeg = np.load(
                os.path.join(phase_dir, "sent_level_EEG.npy"), mmap_mode="r"
            )
            self.input_attn_mask = np.load(
                os.path.join(phase_dir, "input_attn_mask.npy"), mmap_mode="r"
            )
            self.input_attn_mask_invert = np.load(
                os.path.join(phase_dir, "input_attn_mask_invert.npy"), mmap_mode="r"
            )
            self.target_ids = np.load(
                os.path.join(phase_dir, "target_ids.npy"), mmap_mode="r"
            )
            self.target_mask = np.load(
                os.path.join(phase_dir, "target_mask.npy"), mmap_mode="r"
            )

            seq_len_path = os.path.join(phase_dir, "seq_len.npy")
            self.seq_len = np.load(seq_len_path, mmap_mode="r") if os.path.isfile(seq_len_path) else None

            phase_meta = self.meta.get("phases", {}).get(phase, {})
            subject_slices = phase_meta.get("subject_slices", {})

            if subject_key in subject_slices:
                start = int(subject_slices[subject_key]["start"])
                length = int(subject_slices[subject_key]["length"])
                self.start = start
                self.length = length
            else:
                self.start = 0
                self.length = int(self.raw_eeg.shape[0])

            sample_raw = self.raw_eeg[self.start]
            self.permute_lcw_to_clw = (
                sample_raw.ndim == 3 and sample_raw.shape[0] == 56 and sample_raw.shape[1] == 105
            )

        else:
            self._loso = True

            all_phases = sorted(self.meta.get("phases", {}).keys())

            self._raw_list = []
            self._sent_list = []
            self._attn_list = []
            self._attn_inv_list = []
            self._tgt_ids_list = []
            self._tgt_mask_list = []
            self._seq_len_list = []
            self._phase_offsets = []

            target_indices = []
            other_indices = []

            offset = 0
            for p in all_phases:
                p_dir = os.path.join(self.task_root, p)
                if not os.path.isdir(p_dir):
                    continue

                raw_path = os.path.join(p_dir, "rawEEG.npy")
                old_path = os.path.join(p_dir, "input_embeddings.npy")
                if os.path.isfile(raw_path):
                    raw_p = np.load(raw_path, mmap_mode="r")
                elif os.path.isfile(old_path):
                    raw_p = np.load(old_path, mmap_mode="r")
                else:
                    continue

                n_p = raw_p.shape[0]
                self._raw_list.append(raw_p)
                self._sent_list.append(np.load(os.path.join(p_dir, "sent_level_EEG.npy"), mmap_mode="r"))
                self._attn_list.append(np.load(os.path.join(p_dir, "input_attn_mask.npy"), mmap_mode="r"))
                self._attn_inv_list.append(np.load(os.path.join(p_dir, "input_attn_mask_invert.npy"), mmap_mode="r"))
                self._tgt_ids_list.append(np.load(os.path.join(p_dir, "target_ids.npy"), mmap_mode="r"))
                self._tgt_mask_list.append(np.load(os.path.join(p_dir, "target_mask.npy"), mmap_mode="r"))

                seq_len_path = os.path.join(p_dir, "seq_len.npy")
                self._seq_len_list.append(
                    np.load(seq_len_path, mmap_mode="r") if os.path.isfile(seq_len_path) else None
                )
                self._phase_offsets.append(offset)

                p_meta = self.meta.get("phases", {}).get(p, {})
                p_slices = p_meta.get("subject_slices", {})

                if subject in p_slices:
                    s = int(p_slices[subject]["start"])
                    l = int(p_slices[subject]["length"])
                    target_indices.extend(range(offset + s, offset + s + l))

                for subj_key, subj_info in p_slices.items():
                    if subj_key == subject:
                        continue
                    if not isinstance(subj_info, dict) or "start" not in subj_info:
                        continue
                    if subj_key in ("ALL",) or subj_key == setting:
                        continue
                    s = int(subj_info["start"])
                    l = int(subj_info["length"])
                    other_indices.extend(range(offset + s, offset + s + l))

                offset += n_p

            if not target_indices:
                print(f"[WARN] Subject '{subject}' not found in {self.task_root}; "
                      f"all samples from this task go to train/dev pool")

            rng = np.random.default_rng(seed)
            arr = np.array(other_indices)
            rng.shuffle(arr)
            split = int(len(arr) * 0.8)
            train_idx = sorted(arr[:split].tolist())
            dev_idx = sorted(arr[split:].tolist())
            test_idx = sorted(target_indices)

            if phase == "train":
                self.indices = train_idx
            elif phase == "dev":
                self.indices = dev_idx
            elif phase == "test":
                self.indices = test_idx
            else:
                raise ValueError(f"Unknown phase: {phase}")

            if self.indices:
                sample_raw = self._get_from(self._raw_list, self._phase_offsets, self.indices[0])
                self.permute_lcw_to_clw = (
                    sample_raw.ndim == 3 and sample_raw.shape[0] == 56 and sample_raw.shape[1] == 105
                )
            else:
                self.permute_lcw_to_clw = False

    @staticmethod
    def _get_from(array_list, offsets, global_idx):
        arr_idx = bisect_right(offsets, global_idx) - 1
        arr = array_list[arr_idx]
        if arr is None:
            return None
        local_idx = global_idx - offsets[arr_idx]
        return arr[local_idx]

    def __len__(self) -> int:
        if self._loso:
            return len(self.indices)
        return self.length

    def __getitem__(self, idx: int):
        if self._loso:
            real_idx = self.indices[idx]
            raw_arr      = self._get_from(self._raw_list,      self._phase_offsets, real_idx)
            sent_arr     = self._get_from(self._sent_list,     self._phase_offsets, real_idx)
            attn_arr     = self._get_from(self._attn_list,     self._phase_offsets, real_idx)
            attn_inv_arr = self._get_from(self._attn_inv_list, self._phase_offsets, real_idx)
            tgt_ids_arr  = self._get_from(self._tgt_ids_list,  self._phase_offsets, real_idx)
            tgt_mask_arr = self._get_from(self._tgt_mask_list, self._phase_offsets, real_idx)
            sl_arr       = self._get_from(self._seq_len_list,  self._phase_offsets, real_idx)

            sent_level = torch.tensor(sent_arr, dtype=torch.float32)
            raw        = torch.tensor(raw_arr,  dtype=torch.float32)
            if self.permute_lcw_to_clw:
                raw = raw.permute(1, 0, 2).contiguous()
            attn     = torch.tensor(attn_arr)
            attn_inv = torch.tensor(attn_inv_arr)
            tgt_ids  = torch.tensor(tgt_ids_arr,  dtype=torch.long)
            tgt_mask = torch.tensor(tgt_mask_arr, dtype=torch.long)
            real_length = int(sl_arr) if sl_arr is not None else int(attn.sum().item())

        else:
            real_idx = self.start + idx

            sent_level = torch.tensor(self.sent_level_eeg[real_idx], dtype=torch.float32)
            raw = torch.tensor(self.raw_eeg[real_idx], dtype=torch.float32)
            if self.permute_lcw_to_clw:
                raw = raw.permute(1, 0, 2).contiguous()

            attn     = torch.tensor(self.input_attn_mask[real_idx])
            attn_inv = torch.tensor(self.input_attn_mask_invert[real_idx])
            tgt_ids  = torch.tensor(self.target_ids[real_idx], dtype=torch.long)
            tgt_mask = torch.tensor(self.target_mask[real_idx], dtype=torch.long)

            if self.seq_len is not None:
                real_length = int(self.seq_len[real_idx])
            else:
                real_length = int(attn.sum().item())

        if self.noise:
            raw = torch.randn_like(raw)

        return (
            sent_level,
            raw,
            attn,
            attn_inv,
            tgt_ids,
            tgt_mask,
            real_length,
        )


class ZuCo_dataset(Dataset):

    def __init__(
        self,
        input_dataset_dicts: Union[str, List[str]],
        phase: str,
        tokenizer,
        subject: str = "ALL",
        eeg_type: str = "GD",
        bands: Optional[Sequence[str]] = None,
        setting: str = "unique_sent",
        seed: int = 42,
        test_input: str = "eeg",
        **kwargs,
    ):
        super().__init__()

        if isinstance(input_dataset_dicts, str):
            input_dataset_dicts = [input_dataset_dicts]

        noise = (test_input == "noise")

        self.datasets: List[_SingleTaskNPYDataset] = []
        self.cum_lengths: List[int] = []

        running = 0
        print(f'[INFO]loading {len(input_dataset_dicts)} task datasets')

        for task_root in input_dataset_dicts:
            ds = _SingleTaskNPYDataset(
                task_root=task_root,
                phase=phase,
                tokenizer=tokenizer,
                subject=subject,
                eeg_type=eeg_type,
                bands=bands,
                setting=setting,
                seed=seed,
                noise=noise,
            )
            self.datasets.append(ds)
            running += len(ds)
            self.cum_lengths.append(running)

            phase_name = "train" if phase == "train" else phase
            print(f'[INFO]initializing a {phase_name} set...')
            print(f'++ adding task to dataset, now we have: {running}')

        if len(self.datasets) == 0:
            raise ValueError("No task datasets were provided.")

    def __len__(self) -> int:
        return self.cum_lengths[-1]

    def __getitem__(self, idx: int):
        ds_idx = bisect_right(self.cum_lengths, idx)
        prev_cum = 0 if ds_idx == 0 else self.cum_lengths[ds_idx - 1]
        local_idx = idx - prev_cum
        return self.datasets[ds_idx][local_idx]
