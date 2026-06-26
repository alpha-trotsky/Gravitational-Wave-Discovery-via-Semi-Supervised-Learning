"""
H-DAE Data Pipeline
====================
Generates clean BBH waveforms, loads real LIGO noise from local HDF5 files,
injects signals, and preprocesses for training.

Pipeline (Aframe-aligned):
  1. Load 10.5s of real LIGO noise from local HDF5 file
  2. Use first 8s for PSD estimation (Welch method)
  3. Generate synthetic BBH waveform (PyCBC SEOBNRv4)
  4. Inject waveform into remaining 2.5s noise window at target SNR
  5. Whiten both noisy and clean using estimated PSD
  6. Crop 0.5s from each edge to remove whitening artifacts
  7. Final 1.5s window at 2048 Hz = 3072 samples

Output per sample:
  - noisy_strain:  [3072] float32 — model input
  - clean_strain:  [3072] float32 — model target
  - snr:           float  — injection SNR
  - mass1, mass2:  float  — black hole masses in solar masses

Usage:
------
    from data_pipeline import GravitationalWaveDataset, get_snr_range
    from torch.utils.data import DataLoader

    # Build dataset from a directory of GWOSC HDF5 files
    dataset = GravitationalWaveDataset(
        data_dir  = "./data/ligo_noise/",
        n_samples = 500,
        snr_range = (12.0, 100.0),
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Iterate
    for batch in loader:
        noisy = batch['noisy_strain']  # [B, 1, 3072]
        clean = batch['clean_strain']  # [B, 1, 3072]
        snr   = batch['snr']           # [B]

    # SNR curriculum — call each epoch to get the current SNR range
    for epoch in range(100):
        snr_range = get_snr_range(epoch, total_epochs=100)
        # rebuild or resample dataset with new snr_range

    # Run matched filter benchmark on a sample
    from data_pipeline import matched_filter_benchmark
    peak_snr = matched_filter_benchmark(injected, waveform, psd)
"""

import os
import random

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pycbc.psd
import torch
from pycbc.filter import matched_filter, sigma, whiten, highpass, lowpass
from pycbc.psd import interpolate, inverse_spectrum_truncation
from pycbc.types import FrequencySeries, TimeSeries
from pycbc.waveform import get_td_waveform
from scipy.signal import resample_poly
from torch.utils.data import DataLoader, Dataset

# ── Constants ──────────────────────────────────────────────────────────────────
SAMPLE_RATE     = 2048   # Hz — target sample rate
TOTAL_DURATION  = 10.5   # seconds — full noise window
PSD_DURATION    = 8.0    # seconds — used for PSD estimation
INJECT_DURATION = 2.5    # seconds — injection window (last 2.5s of total)
CROP            = 0.5    # seconds cropped from each edge after whitening
TRAIN_DURATION  = 1.5    # seconds — final training window
N_SAMPLES       = int(TRAIN_DURATION * SAMPLE_RATE)  # 3072
F_LOWER         = 20.0   # Hz — low frequency cutoff


# ── Waveform Generation ────────────────────────────────────────────────────────
def generate_bbh_waveform(
    mass1       : float,
    mass2       : float,
    distance    : float = 500.0,
    inclination : float = 0.0,
    approximant : str   = "SEOBNRv4",
) -> TimeSeries:
    """
    Generate a synthetic BBH inspiral waveform using PyCBC.
    Returns the plus polarization h+ as a TimeSeries at SAMPLE_RATE.

    Args:
        mass1:       Primary mass in solar masses
        mass2:       Secondary mass in solar masses
        distance:    Luminosity distance in Mpc
        inclination: Orbital inclination in radians
        approximant: Waveform model (SEOBNRv4 is standard)

    Returns:
        hp: Plus polarization TimeSeries
    """
    hp, _ = get_td_waveform(
        approximant = approximant,
        mass1       = mass1,
        mass2       = mass2,
        distance    = distance,
        inclination = inclination,
        delta_t     = 1.0 / SAMPLE_RATE,
        f_lower     = F_LOWER,
    )
    return hp


# ── Noise Loading from HDF5 ────────────────────────────────────────────────────
def load_hdf5_chunk(
    filepath        : str,
    chunk_start_idx : int   = None,
    target_rate     : int   = SAMPLE_RATE,
    chunk_duration  : float = TOTAL_DURATION,
) -> TimeSeries:
    """
    Load a chunk of strain data from a local GWOSC HDF5 file,
    downsampled to target_rate (2048 Hz).

    GWOSC files are 4096s long at 4096 Hz. This function:
      - Opens the file
      - Picks a random start position (with 10% edge buffers)
      - Reads chunk_duration seconds of raw strain
      - Downsamples from 4096 Hz to 2048 Hz
      - Returns a PyCBC TimeSeries

    Args:
        filepath:        Path to GWOSC HDF5 file
        chunk_start_idx: Sample index to start from (randomized if None)
        target_rate:     Target sample rate in Hz
        chunk_duration:  Duration of chunk in seconds

    Returns:
        strain: PyCBC TimeSeries at target_rate Hz
    """
    with h5py.File(filepath, 'r') as f:
        original_rate = int(round(1.0 / f['strain']['Strain'].attrs['Xspacing']))
        gps_start     = f['meta']['GPSstart'][()]
        total_samples = f['strain']['Strain'].shape[0]
        chunk_samples = int(chunk_duration * original_rate)

        # 10% buffer at each end to avoid file-edge artifacts
        buffer    = int(0.1 * total_samples)
        max_start = total_samples - chunk_samples - buffer

        if chunk_start_idx is None:
            chunk_start_idx = random.randint(buffer, max_start)

        raw = f['strain']['Strain'][chunk_start_idx : chunk_start_idx + chunk_samples]

    # Downsample if needed (4096 → 2048)
    if original_rate != target_rate:
        raw = resample_poly(raw, target_rate, original_rate).astype(np.float64)

    chunk_gps = gps_start + chunk_start_idx / original_rate
    return TimeSeries(raw, delta_t=1.0 / target_rate, epoch=chunk_gps)


def get_random_chunk(data_dir: str) -> TimeSeries:
    """
    Pick a random HDF5 file from data_dir and return a random 10.5s chunk.

    Args:
        data_dir: Directory containing GWOSC HDF5 files

    Returns:
        strain: PyCBC TimeSeries at 2048 Hz
    """
    files    = [f for f in os.listdir(data_dir) if f.endswith('.hdf5')]
    filepath = os.path.join(data_dir, random.choice(files))
    return load_hdf5_chunk(filepath)


# ── PSD Estimation ─────────────────────────────────────────────────────────────
def estimate_psd(noise_segment: TimeSeries) -> FrequencySeries:
    """
    Estimate PSD from the first 8s of a 10.5s noise segment using
    the Welch method (overlapping 4s windows averaged together).

    The PSD is used for:
      - Whitening (flattening the noise floor)
      - SNR scaling during injection
      - Matched filter benchmark

    Args:
        noise_segment: Full 10.5s PyCBC TimeSeries at 2048 Hz

    Returns:
        psd: FrequencySeries PSD estimate
    """
    n_psd    = int(PSD_DURATION * SAMPLE_RATE)
    psd_data = noise_segment[:n_psd]
    seg_len  = int(4 * SAMPLE_RATE)  # 4s Welch segments

    psd = pycbc.psd.welch(psd_data, seg_len=seg_len, seg_stride=seg_len // 2)
    psd = interpolate(psd, noise_segment.delta_f)
    psd = inverse_spectrum_truncation(
        psd, int(4 * SAMPLE_RATE), low_frequency_cutoff=F_LOWER
    )
    return psd


# ── Signal Injection ───────────────────────────────────────────────────────────
def inject_signal(
    noise_segment : TimeSeries,
    waveform      : TimeSeries,
    psd           : FrequencySeries,
    target_snr    : float,
    rng           = None,
) -> tuple[TimeSeries, TimeSeries, float]:
    """
    Inject a BBH waveform into the last 2.5s of the noise segment,
    scaled to a target SNR.

    Steps:
      1. Extract last 2.5s of noise (injection window)
      2. Resize waveform to fit (merger lands at end of window)
      3. Scale waveform amplitude to hit target_snr
      4. Add scaled waveform to noise

    Args:
        noise_segment: Full 10.5s TimeSeries
        waveform:      Clean BBH waveform TimeSeries
        psd:           PSD FrequencySeries (from first 8s of noise)
        target_snr:    Desired injection SNR

    Returns:
        injected:     2.5s noisy strain with signal injected
        clean:        2.5s clean signal only (zero-padded to window)
        achieved_snr: Actual injection SNR (= target_snr)
    """
    from pycbc.types import TimeSeries
    from pycbc.psd import interpolate as psd_interpolate
    import numpy as np

    if rng is None:
        rng = random

    n_inject = int(INJECT_DURATION * SAMPLE_RATE)
    injection_noise = noise_segment[len(noise_segment) - n_inject:]

    w = np.array(waveform.data, dtype=np.float64)

    if len(w) >= n_inject:
        aligned_np = w[-n_inject:]
    else:
        aligned_np = np.zeros(n_inject, dtype=np.float64)
        aligned_np[-len(w):] = w

    aligned = TimeSeries(
        aligned_np,
        delta_t=waveform.delta_t,
        epoch=injection_noise.start_time
    )

    waveform_fd = aligned.to_frequencyseries()

    # 2. Interpolate the PSD (real numbers) to match the waveform's delta_f.
    # Never interpolate the complex waveform!
    matched_psd = psd_interpolate(psd, waveform_fd.delta_f)

    # 3. Compute the natural SNR of the unscaled waveform using the correctly matched PSD
    natural_snr = sigma(aligned, psd=matched_psd, low_frequency_cutoff=F_LOWER)

    # 4. Scale the waveform to hit the exact target SNR
    scale = target_snr / natural_snr
    scaled_waveform = aligned * scale

    # 5. Place the scaled waveform randomly within the central 1.5s crop.
    #    Ensure the full waveform stays within the crop to avoid truncation.
    clean_arr = np.zeros(n_inject, dtype=np.float64)
    n_crop = int(CROP * SAMPLE_RATE)
    max_start = n_crop + N_SAMPLES - len(scaled_waveform)
    if max_start < n_crop:
        # If waveform is longer than the final 1.5s crop, truncate to the last N_SAMPLES
        scaled_waveform = scaled_waveform[-N_SAMPLES:]
        max_start = n_crop

    start_ix = rng.randint(n_crop, max_start)
    end_ix = start_ix + len(scaled_waveform)
    clean_arr[start_ix:end_ix] = np.array(scaled_waveform, dtype=np.float64)

    # Inject into noise (deterministic alignment)
    injected = TimeSeries(
        injected.data + clean_arr,
        delta_t=injection_noise.delta_t,
        epoch=injection_noise.start_time
    )

    clean = TimeSeries(clean_arr, delta_t=injection_noise.delta_t, epoch=injection_noise.start_time)

    return injected, clean, target_snr

# ── Whitening and Cropping ─────────────────────────────────────────────────────
def whiten_and_crop(
    strain : TimeSeries,
    psd    : FrequencySeries,
) -> np.ndarray:
    
    """
    Whiten strain using estimated PSD then crop 0.5s from each edge
    to remove whitening filter artifacts.

    Whitening divides by sqrt(PSD) in frequency domain, equalizing
    the noise floor across all frequencies so the model does not
    learn frequency-specific noise patterns.

    Both noisy and clean signals must be whitened with the same PSD
    so they live in the same amplitude space.

    Args:
        strain: 2.5s PyCBC TimeSeries to whiten
        psd:    PSD FrequencySeries

    Returns:
        cropped: np.ndarray of shape [3072] (1.5s at 2048 Hz)
    """
    from pycbc.psd import interpolate as psd_interpolate

    # Bandlimit before whitening to avoid high-frequency amplification
    bandpassed = highpass(strain, F_LOWER)
    bandpassed = lowpass(bandpassed, F_HIGH)

    # Match PSD frequency resolution to the strain being whitened
    psd_interp = psd_interpolate(psd, bandpassed.delta_f)

    # Whiten via PyCBC helper (handles windowing/tapering)
    whitened = whiten(bandpassed, psd_interp, low_frequency_cutoff=F_LOWER)

    # Crop 0.5s from each edge
    n_crop  = int(CROP * SAMPLE_RATE)
    cropped = whitened.data[n_crop : n_crop + N_SAMPLES]

    assert len(cropped) == N_SAMPLES, \
        f"Expected {N_SAMPLES} samples, got {len(cropped)}"
    return cropped.astype(np.float32)

# ── Full Single-Sample Pipeline ────────────────────────────────────────────────
def generate_sample(
    data_dir   : str,
    mass1      : float = None,
    mass2      : float = None,
    target_snr : float = None,
) -> dict:
    """
    Full pipeline for one training sample:
      noise → PSD → waveform → inject → whiten → crop

    Args:
        data_dir:   Directory of GWOSC HDF5 files
        mass1:      Primary mass in solar masses (randomized if None)
        mass2:      Secondary mass in solar masses (randomized if None)
        target_snr: Injection SNR (randomized 12-100 if None)

    Returns:
        dict with keys:
            'noisy_strain': np.ndarray [3072] — model input
            'clean_strain': np.ndarray [3072] — model target
            'snr':          float
            'mass1':        float
            'mass2':        float
    """
    if mass1 is None:
        mass1 = random.uniform(10, 80)
    if mass2 is None:
        mass2 = random.uniform(10, mass1)
    if target_snr is None:
        target_snr = random.uniform(12, 100)

    noise    = get_random_chunk(data_dir)
    psd      = estimate_psd(noise)
    waveform = generate_bbh_waveform(mass1, mass2)

    injected, clean_signal, achieved_snr = inject_signal(
        noise, waveform, psd, target_snr
    )

    noisy_strain = whiten_and_crop(injected, psd)
    clean_strain = whiten_and_crop(clean_signal, psd)

    return {
        "noisy_strain": noisy_strain,
        "clean_strain": clean_strain,
        "snr"         : achieved_snr,
        "mass1"       : mass1,
        "mass2"       : mass2,
    }


# ── PyTorch Dataset ────────────────────────────────────────────────────────────
class GravitationalWaveDataset(Dataset):
    """
    PyTorch Dataset for H-DAE training.
    Generates n_samples training pairs by running the full pipeline
    n_samples times with randomized noise chunks, masses, and SNR.

    Each sample:
        noisy_strain: [1, 3072] float32 tensor — model input
        clean_strain: [1, 3072] float32 tensor — model target
        snr:          scalar float32 tensor

    Args:
        data_dir:  Directory containing GWOSC HDF5 noise files
        n_samples: Number of training samples to generate
        snr_range: (min_snr, max_snr) for injection curriculum
    """
    def __init__(
        self,
        data_dir  : str,
        n_samples : int,
        snr_range : tuple[float, float] = (12.0, 100.0),
    ):
        self.samples = []
        print(f"Generating {n_samples} samples from {data_dir}...")

        for i in range(n_samples):
            target_snr = random.uniform(*snr_range)
            try:
                sample = generate_sample(
                    data_dir   = data_dir,
                    target_snr = target_snr,
                )
                self.samples.append(sample)
                if (i + 1) % 50 == 0:
                    print(f"  {i+1}/{n_samples}")
            except Exception as e:
                print(f"  Skipping sample {i}: {e}")

        print(f"Done. {len(self.samples)} valid samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "noisy_strain": torch.tensor(s["noisy_strain"]).unsqueeze(0),  # [1, 3072]
            "clean_strain": torch.tensor(s["clean_strain"]).unsqueeze(0),  # [1, 3072]
            "snr"         : torch.tensor(s["snr"], dtype=torch.float32),
        }


# ── SNR Curriculum ─────────────────────────────────────────────────────────────
def get_snr_range(epoch: int, total_epochs: int) -> tuple[float, float]:
    """
    Linearly decay minimum SNR from 12 down to 4 over training,
    as per the Aframe framework. Maximum SNR stays at 100.

    Args:
        epoch:        Current epoch (0-indexed)
        total_epochs: Total number of training epochs

    Returns:
        (min_snr, max_snr) for this epoch
    """
    progress = epoch / max(total_epochs - 1, 1)
    min_snr  = 12.0 - progress * (12.0 - 4.0)  # 12 → 4
    return (min_snr, 100.0)

def align_waveform_to_window(waveform: TimeSeries, window_len: int) -> TimeSeries:
    """
    Return a TimeSeries of exactly window_len samples, with merger aligned to end.
    If waveform is longer than the window, keep the last window_len samples.
    If shorter, left-pad with zeros.
    """
    import numpy as np
    from pycbc.types import TimeSeries

    w = np.asarray(waveform, dtype=np.float64)

    if len(w) >= window_len:
        aligned_np = w[-window_len:]
    else:
        aligned_np = np.zeros(window_len, dtype=np.float64)
        aligned_np[-len(w):] = w

    return TimeSeries(
        aligned_np,
        delta_t=waveform.delta_t
    )

def matched_filter_benchmark(
    noise_segment: TimeSeries,
    waveform: TimeSeries,
    psd,
    target_snr: float,
    f_lower: float = 20.0,
):
    """
    Benchmark one injection by matched filtering.

    Returns:
        injected              : noisy data with injected signal
        scaled_template       : clean injected template after scaling/alignment
        natural_snr           : optimal SNR of unscaled aligned waveform
        target_snr            : requested injection SNR
        recovered_peak_snr    : peak of matched-filter SNR time series
        snr_ts                : PyCBC matched-filter SNR time series
    """
    window_len = len(noise_segment)

    # 1. Align waveform to the injection window
    template = align_waveform_to_window(waveform, window_len)

    # 2. Compute optimal SNR of the unscaled template
    #    Ensure the PSD delta_f matches the template's delta_f.
    psd_for_template = interpolate(psd, template.delta_f)
    natural_snr = sigma(template, psd=psd_for_template, low_frequency_cutoff=f_lower)

    if natural_snr <= 0 or not np.isfinite(natural_snr):
        raise ValueError(f"Invalid natural_snr: {natural_snr}")

    # 3. Scale to desired target SNR
    scale = target_snr / natural_snr
    scaled_template = template * scale

    # 4. Inject into noise
    injected = noise_segment.copy()
    injected.inject(scaled_template)

    # 5. Recover with matched filter
    snr_ts = matched_filter(
        scaled_template,
        injected,
        psd=psd_for_template,
        low_frequency_cutoff=f_lower
    )

    recovered_peak_snr = float(np.abs(snr_ts).max())

    return {
        "injected": injected,
        "scaled_template": scaled_template,
        "natural_snr": float(natural_snr),
        "target_snr": float(target_snr),
        "recovered_peak_snr": recovered_peak_snr,
        "snr_ts": snr_ts,
    }

# ── Sanity Check ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    DATA_DIR = "./Model/LIGO-noise"
    SAVE_DIR = '/mnt/c/Users/Belia/OneDrive/Documents/GitHub/Gravitational-Wave-Discovery-via-Semi-Supervised-Learning/'

    # ── 1. Waveform generation ─────────────────────────────────────────────────
    print("=" * 50)
    print("1. Waveform generation")
    print("=" * 50)

    hp = generate_bbh_waveform(mass1=30, mass2=30)
    t  = np.arange(len(hp)) * hp.delta_t

    print(f"  Duration: {len(hp) * hp.delta_t:.2f}s  |  Samples: {len(hp)}")

    plt.figure(figsize=(10, 4))
    plt.plot(t, hp.numpy())
    plt.title("Synthetic BBH Waveform (30+30 solar masses)")
    plt.xlabel("Time (s)")
    plt.ylabel("Strain (h+)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(SAVE_DIR + 'BBH_chirp.png')
    plt.show()

    # ── 2. Load noise from HDF5 ────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("2. Loading LIGO noise from HDF5")
    print("=" * 50)

    if not os.path.isdir(DATA_DIR) or \
       not any(f.endswith('.hdf5') for f in os.listdir(DATA_DIR)):
        print(f"  No HDF5 files found in {DATA_DIR}")
        print("  Download GWOSC 4kHz HDF5 files and place them there.")
        sys.exit(1)

    noise   = get_random_chunk(DATA_DIR)
    t_noise = np.arange(len(noise)) * (1.0 / SAMPLE_RATE)

    print(f"  Loaded {len(noise)} samples ({len(noise)/SAMPLE_RATE:.1f}s) at {SAMPLE_RATE} Hz")

    plt.figure(figsize=(10, 4))
    plt.plot(t_noise, noise.data)
    plt.title("Raw LIGO Noise (H1, 10.5s)")
    plt.xlabel("Time (s)")
    plt.ylabel("Strain")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(SAVE_DIR + 'ligo_noise.png')
    plt.show()

    # ── 3. PSD estimation ──────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("3. PSD estimation")
    print("=" * 50)

    psd = estimate_psd(noise)
    print(f"  PSD estimated over first {PSD_DURATION}s — {len(psd)} frequency bins")

    # ── 4. Injection ───────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("4. Signal injection at SNR=20")
    print("=" * 50)

    injected, clean_signal, achieved_snr = inject_signal(
        noise, hp, psd, target_snr=20
    )
    print(f"  Injection SNR: {achieved_snr:.1f}")

    # ── 5. Whiten and crop ─────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("5. Whitening and cropping")
    print("=" * 50)

    noisy_strain = whiten_and_crop(injected, psd)
    print("len(clean_signal):", len(clean_signal))
    print("clean_signal duration:", len(clean_signal) * clean_signal.delta_t)
    clean_strain = whiten_and_crop(clean_signal, psd)

    print(f"  noisy_strain shape: {noisy_strain.shape}  (model input)")
    print(f"  clean_strain shape: {clean_strain.shape}  (model target)")

    t_train = np.arange(N_SAMPLES) / SAMPLE_RATE

    plt.figure(figsize=(10, 4))
    plt.plot(t_train, noisy_strain, alpha=0.7, label='Noisy (model input)')
    plt.plot(t_train, clean_strain, alpha=0.9, label='Clean (model target)')
    plt.plot(t_train, clean_signal[2048:5120], alpha=0.7, label='Clean pre-psd input')
    plt.title(f"Whitened Strain after Injection (SNR={achieved_snr:.0f})")
    plt.xlabel("Time (s)")
    plt.ylabel("Whitened Strain")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(SAVE_DIR + 'whitened_injection.png')
    plt.show()

    # ── 6. Matched filter benchmark ────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("6. Matched filter benchmark")
    print("=" * 50)

    mean_snr = matched_filter_benchmark(
        noise_segment   = noisy_strain,
        waveform  = hp,
        psd       = psd,
        target_snr = 20
    )

    print(f"  Matched filter mean SNR: {mean_snr:.1f}")
    # ── 7. Dataset generation test ─────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("7. Dataset generation (5 samples)")
    print("=" * 50)

    dataset = GravitationalWaveDataset(
        data_dir  = DATA_DIR,
        n_samples = 5,
        snr_range = (12.0, 100.0),
    )
    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    batch  = next(iter(loader))

    print(f"  noisy_strain batch shape: {batch['noisy_strain'].shape}")
    print(f"  clean_strain batch shape: {batch['clean_strain'].shape}")
    print(f"  SNRs in batch: {batch['snr'].tolist()}")

    print("\nAll checks passed.")