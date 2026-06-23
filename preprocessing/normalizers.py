import torch
import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader
from collections.abc import Sequence
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    MaxAbsScaler,
    RobustScaler
)


class DatasetNormalizer(Dataset):
    """
    A Dataset wrapper that applies on-device normalization after fitting.

    This class fits a normalizer (e.g., from scikit-learn) on a subset of the
    data. It then detects the scaler type, caches its specific parameters 
    (e.g., mean, scale, min, center) as PyTorch tensors.

    This allows __getitem__ and, more importantly, inverse_transform_y
    to perform transformations directly on the data's device (e.g., 'cuda')
    without costly CPU synchronization.
    """

    def __init__(self, dataset: Dataset, x_normalizer=None, y_normalizers=None):
        self.dataset = dataset
        self.x_normalizer = x_normalizer  # The original sklearn object

        if y_normalizers and not isinstance(y_normalizers, Sequence):
            self.y_normalizers = [y_normalizers]
        else:
            self.y_normalizers = y_normalizers

        # --- Cached PyTorch Tensor Parameters ---
        # These will be populated during .fit()
        # We use dicts to store the specific params for each scaler type
        self.x_params = {}
        self.y_params_list = []

    def _cache_params_as_tensors(self, normalizer):
        """
        Detects scaler type and caches its parameters as PyTorch tensors.
        Returns a dictionary of parameters.
        """
        params = {'type': 'fallback'}  # Default, will use slow .numpy() path
        dev = torch.device('cpu')  # Cache params on CPU

        try:
            if isinstance(normalizer, StandardScaler):
                params['mean'] = torch.from_numpy(normalizer.mean_).float().to(dev)
                params['scale'] = torch.from_numpy(normalizer.scale_).float().to(dev)
                params['type'] = 'standard'

            elif isinstance(normalizer, MinMaxScaler):
                # formula: X_std = (X - X.min) * scale + range_min
                # inverse: X = (X_std - range_min) / scale + X.min
                params['min'] = torch.from_numpy(normalizer.min_).float().to(dev)
                params['scale'] = torch.from_numpy(normalizer.scale_).float().to(dev)
                params['range_min'] = normalizer.feature_range[0]  # This is just a float
                params['type'] = 'minmax'

            elif isinstance(normalizer, MaxAbsScaler):
                # formula: X_std = X / scale
                # inverse: X = X_std * scale
                params['scale'] = torch.from_numpy(normalizer.scale_).float().to(dev)
                params['type'] = 'maxabs'

            elif isinstance(normalizer, RobustScaler):
                # formula: X_std = (X - center) / scale
                # inverse: X = X_std * scale + center
                params['center'] = torch.from_numpy(normalizer.center_).float().to(dev)
                params['scale'] = torch.from_numpy(normalizer.scale_).float().to(dev)
                params['type'] = 'robust'

            if params['type'] != 'fallback':
                print(f" - Cached params for {type(normalizer).__name__} as tensors.")
            else:
                print(f" - Warning: {type(normalizer).__name__} not optimized. "
                      "Will use slower NumPy fallback.")

        except AttributeError:
            print(f" - Warning: Failed to cache params for {type(normalizer).__name__}. "
                  "Normalizer might not be fitted. Using fallback.")
            params['type'] = 'fallback'

        return params

    def fit(self, data_loader: DataLoader, num_batches: int):
        if num_batches <= 0:
            print("Warning: num_batches is zero or negative. Skipping fitting.")
            return

        print(f"Aggregating {num_batches} batches to fit normalizers...")

        x_fit_samples = []
        y_fit_samples_list = [[] for _ in self.y_normalizers] if self.y_normalizers else []

        # Reset cached params
        self.x_params = {}
        self.y_params_list = []

        # --- 1. Aggregate Data (on CPU) ---
        for i, batch in enumerate(data_loader):
            if i >= num_batches:
                break

            x_batch, y_batch = batch

            if self.x_normalizer:
                x_fit_samples.append(x_batch.cpu().numpy())

            if self.y_normalizers:
                y_tuple = y_batch if isinstance(y_batch, (list, tuple)) else (y_batch,)
                for j, y_part in enumerate(y_tuple):
                    y_fit_samples_list[j].append(y_part.cpu().numpy())

        if not x_fit_samples and not any(y_fit_samples_list):
            print("Warning: No data was collected. Skipping fitting.")
            return

        # --- 2. Fit Normalizers and Cache Tensor Params ---
        if self.x_normalizer and x_fit_samples:
            x_fit_final = np.concatenate(x_fit_samples, axis=0)
            self.x_normalizer.fit(x_fit_final.reshape(-1, 1))
            print(f"Input (X) normalizer fitted on {x_fit_final.shape[0]} samples.")
            self.x_params = self._cache_params_as_tensors(self.x_normalizer)

        if self.y_normalizers:
            y_fit_final_tuple = tuple(np.concatenate(y_part_list, axis=0) for y_part_list in y_fit_samples_list)

            for i, (normalizer, y_data) in enumerate(zip(self.y_normalizers, y_fit_final_tuple)):
                if normalizer:
                    normalizer.fit(y_data.reshape(-1, 1))
                    print(f"Output (y) normalizer #{i+1} fitted on {y_data.shape[0]} samples.")
                    self.y_params_list.append(self._cache_params_as_tensors(normalizer))
                else:
                    self.y_params_list.append({})  # Add empty dict to keep lists in sync

        print("Fitting complete.")

    def __len__(self):
        return len(self.dataset)

    def _apply_transform(self, data: torch.Tensor, params: dict, normalizer):
        """Helper to apply forward transform using cached params or fallback."""
        dev = data.device
        param_type = params.get('type', 'fallback')

        if param_type == 'standard':
            mean = params['mean'].to(dev)
            scale = params['scale'].to(dev)
            return (data - mean) / scale

        elif param_type == 'minmax':
            min_ = params['min'].to(dev)
            scale = params['scale'].to(dev)
            range_min = params['range_min']
            return (data - min_) * scale + range_min

        elif param_type == 'maxabs':
            scale = params['scale'].to(dev)
            return data / scale

        elif param_type == 'robust':
            center = params['center'].to(dev)
            scale = params['scale'].to(dev)
            return (data - center) / scale

        else:  # Fallback to original numpy method
            data_np = data.cpu().numpy()  # Move to CPU
            original_shape = data_np.shape
            transformed_np = normalizer.transform(data_np.reshape(-1, 1)).reshape(original_shape)
            # Move back to original device
            return torch.from_numpy(transformed_np).float().to(dev)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]  # Data is typically on CPU here

        if self.x_normalizer:
            x = self._apply_transform(x, self.x_params, self.x_normalizer)

        if self.y_normalizers:
            is_single_output = not isinstance(y, (tuple, list))
            y_tuple = (y,) if is_single_output else y

            normalized_y = []
            for i, y_part in enumerate(y_tuple):
                if i < len(self.y_params_list) and self.y_normalizers[i]:
                    params = self.y_params_list[i]
                    normalizer = self.y_normalizers[i]
                    normalized_y.append(self._apply_transform(y_part, params, normalizer))
                else:
                    normalized_y.append(y_part)  # No normalizer

            y = normalized_y[0] if is_single_output else tuple(normalized_y)

        return x, y

    def _apply_inverse_transform(self, data: torch.Tensor, params: dict, normalizer):
        """Helper to apply inverse transform using cached params or fallback."""
        dev = data.device
        param_type = params.get('type', 'fallback')

        # --- EFFICIENT, ON-DEVICE (GPU/CUDA) TRANSFORMS ---
        if param_type == 'standard':
            mean = params['mean'].to(dev)
            scale = params['scale'].to(dev)
            return data * scale + mean

        elif param_type == 'minmax':
            min_ = params['min'].to(dev)
            scale = params['scale'].to(dev)
            range_min = params['range_min']
            return (data - range_min) / scale + min_

        elif param_type == 'maxabs':
            scale = params['scale'].to(dev)
            return data * scale

        elif param_type == 'robust':
            center = params['center'].to(dev)
            scale = params['scale'].to(dev)
            return data * scale + center

        else:  # --- SLOW FALLBACK (GPU -> CPU -> GPU) ---
            y_np = data.cpu().numpy()  # Force to CPU
            original_shape = y_np.shape
            denorm_part_np = normalizer.inverse_transform(y_np.reshape(-1, 1)).reshape(original_shape)
            # Move result back to the original device
            return torch.from_numpy(denorm_part_np).float().to(dev)

    def inverse_transform_y(self, y_pred):
        """
        Denormalizes the model's output predictions on its current device.
        This is now highly efficient for GPU tensors for supported scalers.
        """
        if not self.y_normalizers:
            return y_pred

        is_single_output = not isinstance(y_pred, (tuple, list))
        y_pred_tuple = (y_pred,) if is_single_output else y_pred

        denormalized_y = []
        for i, y_part in enumerate(y_pred_tuple):
            if i < len(self.y_params_list) and self.y_normalizers[i]:
                params = self.y_params_list[i]
                normalizer = self.y_normalizers[i]
                denormalized_y.append(self._apply_inverse_transform(y_part, params, normalizer))
            else:
                denormalized_y.append(y_part)  # No normalizer

        return denormalized_y[0] if is_single_output else tuple(denormalized_y)

    def _normalize_sample(self, data):
        """Helper to normalize a single sample, now device-aware."""
        if not isinstance(data, torch.Tensor):
            data = torch.from_numpy(data).float()

        if self.x_normalizer:
            return self._apply_transform(data, self.x_params, self.x_normalizer)
        else:
            return data  # No normalizer

    def save_normalizers(self, path: str):
        with open(path, 'wb') as f:
            # Save both the sklearn objects and the cached tensor params
            state = {
                'x_normalizer': self.x_normalizer,
                'y_normalizers': self.y_normalizers,
                'x_params': self.x_params,
                'y_params_list': self.y_params_list
            }
            pickle.dump(state, f)
        print(f"Normalizers and cached params saved to {path}")

    def load_normalizers(self, path: str):
        with open(path, 'rb') as f:
            state = pickle.load(f)
            self.x_normalizer = state.get('x_normalizer')
            self.y_normalizers = state.get('y_normalizers')
            # Load cached params if they exist
            self.x_params = state.get('x_params', {})
            self.y_params_list = state.get('y_params_list', [])
        print(f"Normalizers and cached params loaded from {path}")
