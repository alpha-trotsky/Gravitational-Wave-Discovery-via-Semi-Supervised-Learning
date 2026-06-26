"""
data.py -- HDF5 loading, cropping, normalization, and Dataset/DataLoader construction.

Scientific framing:
  H-AE is a manifold-learning autoencoder trained purely on clean BBH inspiral
  signals. h1_signal stores the raw (unwhitened) GW strain at ~1e-21 scale.
  Training pairs are (signal, signal) -- the model learns the clean-signal manifold.
  The Hamiltonian inductive bias (approximate energy conservation in the adiabatic
  pre-merger regime) is physically appropriate for these inspiral waveforms.

  Noise robustness is evaluated separately (evaluate.py) by passing whitened noisy
  h1_strain through the trained model -- an honest OOD test since the model never
  saw noise during training.

Crop window (centre): [event_idx - 2048, event_idx - 256] = 1792 samples (~0.87 s).

Time-jitter augmentation (training only):
  The merger always sits at the same absolute sample index, so a model without
  augmentation can exploit fixed alignment rather than learning waveform structure.
  During training we randomly shift the extraction window by up to MAX_JITTER = 256
  samples (= 0.125 s at 2048 Hz) in either direction.  Val/test always use the
  original centre crop so that MSE numbers remain directly comparable.

Train/Val/Test split by progenitor mass:
  - Train : (mass1 <= 65 AND mass2 <= 65) excluding test1 overlap
  - Val   : 15% random holdout from the train range
  - Test1 : mass2 ∈ [5,15]  AND mass1 ∈ [50,65]  (asymmetric intermediate mass-ratio OOD)
  - Test2 : mass2 ∈ [0,20]  AND mass1 > 65        (extreme mass-ratio, heavy primary OOD)
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
SAMPLING_RATE  = 2048
SECONDS_BEFORE = 5.5
EVENT_IDX      = int(SECONDS_BEFORE * SAMPLING_RATE)  # 11264

# Centre crop (used for normalisation std and for val/test extraction)
CROP_START = EVENT_IDX - 2048   # 9216
CROP_END   = EVENT_IDX - 256    # 11008
CROP_LEN   = CROP_END - CROP_START   # 1792 samples

# Jitter margin stored around the centre crop
MAX_JITTER = 256   # samples; ±0.125 s at 2048 Hz
WIDE_START = CROP_START - MAX_JITTER   # 8960
WIDE_END   = CROP_END   + MAX_JITTER   # 11264  (= EVENT_IDX, right at merger)
WIDE_LEN   = WIDE_END - WIDE_START     # 2304 samples stored per signal

VAL_FRACTION = 0.15
RANDOM_SEED  = 42


# ──────────────────────────────────────────────────────────────────────────────
# Raw loading
# ──────────────────────────────────────────────────────────────────────────────

def load_raw(hdf_path: str):
    """Load h1_signal (clean waveforms) and mass parameters."""
    with h5py.File(hdf_path, "r") as f:
        h1_signal = f["injection_parameters/h1_signal"][:]   # (N, 16384) float64
        mass1     = f["injection_parameters/mass1"][:]
        mass2     = f["injection_parameters/mass2"][:]

    N = h1_signal.shape[0]
    if N < 100:
        print(f"\nWARNING: Only {N} injection samples found in dataset.hdf.\n"
              "   Splits will be very small. Consider regenerating with more samples.\n")
    return h1_signal, mass1, mass2


def crop_and_normalize_signal(signal: np.ndarray) -> np.ndarray:
    """
    Extract the WIDE window [WIDE_START, WIDE_END] = 2304 samples and
    normalise each sample by the std of the CENTRE crop [CROP_START, CROP_END].

    Normalising by the centre std means:
      - val/test (always centre crop) have std = 1 exactly as before.
      - jittered train crops have std ~ 1 (slightly different slice, same scale).

    Returns: (N, WIDE_LEN) float32
    """
    wide   = signal[:, WIDE_START:WIDE_END]           # (N, 2304) float64
    centre = signal[:, CROP_START:CROP_END]           # (N, 1792) -- for std only
    std    = centre.std(axis=1, keepdims=True)
    std    = np.where(std < 1e-30, 1.0, std)
    return (wide / std).astype(np.float32)            # (N, 2304) float32


def load_noisy_test_data(hdf_path: str, test_idx: np.ndarray) -> torch.Tensor:
    """
    Load whitened noisy h1_strain for the test indices (for noise-robustness eval).
    Always uses the fixed centre crop -- no jitter -- so it matches the test targets.

    Returns: (len(test_idx), 1, 1792) float32 tensor
    """
    with h5py.File(hdf_path, "r") as f:
        h1_strain = f["injection_samples/h1_strain"][test_idx]

    s   = h1_strain[:, CROP_START:CROP_END].astype(np.float64)
    std = s.std(axis=1, keepdims=True)
    std = np.where(std < 1e-30, 1.0, std)
    return torch.from_numpy((s / std).astype(np.float32)[:, None, :])  # (N,1,1792)


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class GWDataset(Dataset):
    """
    Clean-signal autoencoder dataset.

    Stores the wide window (N, 2304) normalised by centre-crop std.

    augment=True  (training)  : randomly shift extraction by ∈ [0, 2*MAX_JITTER]
                                samples each time a sample is accessed.
    augment=False (val/test)  : always extract the centre CROP_LEN samples
                                (offset = MAX_JITTER), reproducing the original crop.

    Both x and y are the same extracted slice (autoencoder target = input).
    """
    def __init__(self, wide_signals: np.ndarray, augment: bool = False):
        # wide_signals: (N, 2304) float32
        self.data    = torch.from_numpy(wide_signals)   # (N, 2304)
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.augment:
            # Uniform random offset in [0, 2*MAX_JITTER] -- each call independent
            offset = torch.randint(0, 2 * MAX_JITTER + 1, ()).item()
        else:
            offset = MAX_JITTER   # centre crop = original [CROP_START, CROP_END]

        s = self.data[idx, offset : offset + CROP_LEN].unsqueeze(0)  # (1, 1792)
        return s, s


# ──────────────────────────────────────────────────────────────────────────────
# Split & DataLoader builder
# ──────────────────────────────────────────────────────────────────────────────

def make_splits(hdf_path: str, batch_size: int = 16, num_workers: int = 0):
    """
    Load clean signals, build wide-window arrays, and return DataLoaders.

    Loaders returned:
      "train"      -- augmented (random jitter), shuffled
      "train_eval" -- no jitter, no shuffle  (for eval-mode train MSE)
      "val"        -- no jitter, no shuffle
      "test1"      -- no jitter, no shuffle  (mass1∈[5,15], mass2∈[50,65])
      "test2"      -- no jitter, no shuffle  (mass1∈[0,20], mass2>65)

    Returns: loaders (dict), info (dict)
    """
    h1_signal, mass1, mass2 = load_raw(hdf_path)
    wide_signals = crop_and_normalize_signal(h1_signal)   # (N, 2304)

    # Test regions (mass1 >= mass2 always; mass1 = heavier, mass2 = lighter component)
    test1_mask = (mass2 >= 5)  & (mass2 <= 15) & (mass1 >= 50) & (mass1 <= 65)
    test2_mask = (mass2 <= 20) & (mass1 > 65)

    # Training: broad low-mass region minus the test1 overlap
    in_train_range = (mass1 <= 65) & (mass2 <= 65)
    train_all_mask = in_train_range & ~test1_mask

    train_all_idx = np.where(train_all_mask)[0]
    test1_idx     = np.where(test1_mask)[0]
    test2_idx     = np.where(test2_mask)[0]

    rng      = np.random.default_rng(RANDOM_SEED)
    n_val    = max(1, int(len(train_all_idx) * VAL_FRACTION))
    val_pick = rng.choice(len(train_all_idx), size=n_val, replace=False)
    val_idx   = train_all_idx[val_pick]
    train_idx = np.delete(train_all_idx, val_pick)

    print(f"\nDataset split:")
    print(f"  train      : {len(train_idx):>6}  (mass1<=65, mass2<=65, excl. test1 region)")
    print(f"  val        : {len(val_idx):>6}  (15% holdout from train range)")
    print(f"  test1      : {len(test1_idx):>6}  mass2 in [5,15]  & mass1 in [50,65]  "
          f"(asymmetric intermediate-q OOD)")
    print(f"  test2      : {len(test2_idx):>6}  mass2 <= 20      & mass1 > 65         "
          f"(extreme mass-ratio, heavy primary OOD)")
    print(f"  Time-jitter: ON for train (+-{MAX_JITTER} samples = "
          f"+-{MAX_JITTER/SAMPLING_RATE*1000:.0f} ms), OFF for val/test\n")

    for name, idx in [("test1", test1_idx), ("test2", test2_idx)]:
        if len(idx) < 50:
            print(f"  WARNING: {name} only {len(idx)} samples "
                  "(< 50). OOD evaluation may be noisy.")
    if len(train_idx) < batch_size:
        print(f"  WARNING: Train set ({len(train_idx)}) smaller than "
              f"batch_size ({batch_size}). Reducing batch_size to match.")
        batch_size = max(1, len(train_idx))

    # Separate dataset objects so augment flag differs between train and eval
    train_ds = GWDataset(wide_signals, augment=True)
    eval_ds  = GWDataset(wide_signals, augment=False)

    def make_loader(dataset, indices, shuffle):
        return DataLoader(
            Subset(dataset, indices),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=False,
            drop_last=False,
        )

    loaders = {
        "train":      make_loader(train_ds, train_idx,  shuffle=True),
        "train_eval": make_loader(eval_ds,  train_idx,  shuffle=False),
        "val":        make_loader(eval_ds,  val_idx,    shuffle=False),
        "test1":      make_loader(eval_ds,  test1_idx,  shuffle=False),
        "test2":      make_loader(eval_ds,  test2_idx,  shuffle=False),
    }

    info = {
        "n_train":    len(train_idx),
        "n_val":      len(val_idx),
        "n_test1":    len(test1_idx),
        "n_test2":    len(test2_idx),
        "train_idx":  train_idx,
        "val_idx":    val_idx,
        "test1_idx":  test1_idx,
        "test2_idx":  test2_idx,
        "batch_size": batch_size,
        "hdf_path":   hdf_path,
    }

    return loaders, info
