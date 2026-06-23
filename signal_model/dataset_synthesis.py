from typing import List, Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import itertools
from pathlib import Path
import pickle


class CreateDataset(Dataset):
    def __init__(
        self,
        array,
        snr: torch.Tensor,
        num_datas: int,
        num_targets: torch.Tensor | int,
        return_cov: bool,
        extract_upper_tri_cov: bool,
        split_output: bool | int,
        seed: int = 42,
        precompute: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.array = array
        self.snr_list = snr
        self.num_datas = num_datas
        self.return_cov = return_cov
        self.num_targets = num_targets if isinstance(num_targets, torch.Tensor) else torch.tensor([num_targets])
        self.ext_utri = extract_upper_tri_cov
        self.split_output = split_output
        self.rng = torch.Generator(device="cpu")
        self.rng.manual_seed(seed)
        np.random.seed(seed)
        
        self.angles_list = torch.as_tensor(self.array.angles_list)
        self.num_angles = len(self.angles_list)

        if self.ext_utri:
            self.triu_indices = torch.triu_indices(
                self.array.num_antenna, 
                self.array.num_antenna, 
                offset=1, 
            )
        
        self.indices = self._create_indices()
        self.total_samples = len(self.indices)
        
        self.precomputed_data = None
        if precompute:
            self._precompute_all_data()

    def __len__(self) -> int:
        return self.total_samples
    
    def _create_indices(self) -> List[Tuple[int, int, int]]:
        indices = []
        for data_idx in range(self.num_datas):
            for snr_idx in range(len(self.snr_list)):
                for target_idx in range(len(self.num_targets)):
                    indices.append((data_idx, snr_idx, target_idx))
        return indices
    
    def __getitem__(self, idx):
        # Handle negative indexing and slicing
        if isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            return [self[i] for i in range(start, stop, step)]
        
        idx = idx % len(self)
        
        _, snr_idx, target_idx = self.indices[idx]
        
        if hasattr(self, 'precomputed_data') and self.precomputed_data is not None:
            return self.precomputed_data[idx]
        
        num_targets_val = self.num_targets[target_idx].item()
        angles = self.array.gen_rand_angles(num_targets_val)
        snr = self.snr_list[snr_idx]
        
        if self.return_cov:
            signal = self._generate_covariance_data(angles, snr)
        else:
            signal = self._generate_timeseries_data(angles, snr)
        
        labels = self._generate_labels(angles)
        
        return signal, labels
    
    def _generate_labels(self, angles: torch.Tensor) -> torch.Tensor:
        angles_tensor = torch.as_tensor(angles)
        diff = self.angles_list.unsqueeze(1) - angles_tensor.unsqueeze(0)
        closest_indices = torch.argmin(torch.abs(diff), dim=0)
        labels = torch.zeros(self.num_angles, dtype=torch.float32)
        labels[closest_indices] = 1.0
        if self.split_output and self.split_output != 0:
            split_size = self.num_angles // self.split_output
            labels = torch.split(labels, split_size)
        return labels

    def _extract_upper_triangular(self, cov_matrix: torch.Tensor) -> torch.Tensor:
        r_complex = cov_matrix[self.triu_indices[0], self.triu_indices[1]]
        r_real = torch.real(r_complex)
        r_imag = torch.imag(r_complex)
        r = torch.cat([r_real, r_imag, torch.angle(r_complex)])

        norm = torch.norm(r)
        return r / norm if norm > 0 else r

    def _generate_covariance_data(self, doas: torch.Tensor, snr: float) -> torch.Tensor:
        sig_numpy = self.array.receive_waveform(doas, int(snr))
        sig = torch.from_numpy(sig_numpy)
        cov = (sig.conj().T @ sig) / sig.shape[0]
        
        if self.ext_utri:
            result = self._extract_upper_triangular(cov)
        else:
            result = torch.stack([cov.real, cov.imag, torch.angle(cov)], dim=0)
        
        return result.float()
    
    def _generate_timeseries_data(self, doas: torch.Tensor, snr: float) -> torch.Tensor:
        sig_numpy = self.array.receive_waveform(doas, int(snr))
        sig = torch.from_numpy(sig_numpy)
        sig_combined = torch.cat([sig.real, sig.imag], dim=-1)
        return sig_combined.float()

    def _precompute_all_data(self):
        """Precompute all data for faster iteration."""
        print("Precomputing all data...")
        self.precomputed_data = []
        for idx in range(len(self)):
            self.precomputed_data.append(self[idx])
        print(f"Precomputed {len(self.precomputed_data)} samples")
    

class DatasetLoader:
    def __init__(self, batch_size: int = 32, num_workers: int = 2, seed: int = 42):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.rng = torch.Generator().manual_seed(seed)

    def __init__(
        self, 
        batch_size: int = 32, 
        num_workers: int = 2, 
        seed: int = 42,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        prefetch_factor: int = 2,
    ):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory and torch.cuda.is_available()
        self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor
        self.rng = torch.Generator().manual_seed(seed)
    
    def get_dataloaders(
        self,
        dataset: Dataset,
        train_ratio: float = 0.8,
        val_ratio: float = 0.2,
        test_ratio: float = 0.0,
        shuffle_train: bool = True,
        shuffle_val: bool = False,
        shuffle_test: bool = False,
        drop_last: bool = True,
    ) -> Tuple[DataLoader, ...]:
        if test_ratio > 0:
            splits = [train_ratio, val_ratio, test_ratio]
        else:
            splits = [train_ratio, val_ratio]
        
        total_len = len(dataset)
        
        split_lengths = []
        for i, ratio in enumerate(splits):
            if i == len(splits) - 1:
                split_length = total_len - sum(split_lengths)
            else:
                split_length = int(total_len * ratio)
            split_lengths.append(split_length)
        
        if any(l <= 0 for l in split_lengths):
            raise ValueError("Resulting splits have non-positive length. Adjust ratios or use larger dataset.")
        
        splits_datasets = random_split(
            dataset, 
            split_lengths, 
            generator=self.rng
        )
        
        dataloaders = []
        shuffle_flags = [shuffle_train, shuffle_val, shuffle_test]
        
        for i, split_dataset in enumerate(splits_datasets):
            shuffle = shuffle_flags[i] if i < len(shuffle_flags) else False
            
            current_batch_size = min(self.batch_size, len(split_dataset))
            if len(split_dataset) < self.batch_size:
                print(f"Warning: Split {i} has only {len(split_dataset)} samples, "
                      f"reducing batch size to {current_batch_size}")
            
            dataloader = DataLoader(
                split_dataset,
                batch_size=current_batch_size,
                shuffle=shuffle,
                num_workers=self.num_workers if len(split_dataset) > 0 else 0,
                pin_memory=self.pin_memory,
                drop_last=drop_last and len(split_dataset) >= current_batch_size,
                generator=self.rng if shuffle else None,
                persistent_workers=self.persistent_workers and self.num_workers > 0,
                prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            )
            dataloaders.append(dataloader)
        
        return tuple(dataloaders)
