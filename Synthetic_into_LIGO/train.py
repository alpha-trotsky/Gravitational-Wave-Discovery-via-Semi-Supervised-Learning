"""
train.py — Training loop for the Hamiltonian Denoising Autoencoder

Total loss
----------
    L = L_recon + λ_h · L_hamiltonian

    L_recon      = MSE(x̂, x_noisy)        ← self-supervised; no clean targets
    L_hamiltonian = sym + λ_cons·(floor + cons)

The denoising is emergent: the Hamiltonian constraint forces the latent
space onto a physically-consistent manifold, so the decoder outputs
something structurally cleaner than the raw noisy input.

Usage
-----
    python train.py \
        --train_noisy data/train_noisy.npy \
        --val_noisy   data/val_noisy.npy   \
        --epochs 100  --batch_size 64

    # With clean arrays for post-hoc overlap logging (not used in loss):
    python train.py \
        --train_noisy data/train_noisy.npy  --train_clean data/train_clean.npy \
        --val_noisy   data/val_noisy.npy    --val_clean   data/val_clean.npy

Array format: float32 NumPy arrays of shape (N, 3072).
"""

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from model import HDAE, hamiltonian_loss


# ── Dataset ───────────────────────────────────────────────────────────────────

class LIGODataset(Dataset):
    """
    Wraps pre-processed (whitened, cropped) LIGO strain arrays.

    Parameters
    ----------
    noisy : (N, 3072) float32 — whitened strain with BBH injection
    clean : (N, 3072) float32 — whitened synthetic waveform only (optional)
                                Not used in training; kept for eval logging.
    """

    def __init__(self, noisy: np.ndarray, clean: np.ndarray = None):
        self.noisy = torch.from_numpy(noisy).float().unsqueeze(1)   # (N, 1, 3072)
        self.clean = (
            torch.from_numpy(clean).float().unsqueeze(1)
            if clean is not None else None
        )

    def __len__(self):
        return len(self.noisy)

    def __getitem__(self, idx):
        if self.clean is not None:
            return self.noisy[idx], self.clean[idx]
        return self.noisy[idx],     # tuple for consistent unpacking


# ── Training utilities ────────────────────────────────────────────────────────

def waveform_overlap(x_hat: torch.Tensor, x_clean: torch.Tensor) -> float:
    """
    Overlap O = <x̂, x_clean> / sqrt(<x̂,x̂><x_clean,x_clean>).
    Averaged over the batch. Range [−1, 1]; 1 = perfect match.
    """
    x_hat   = x_hat.squeeze(1)    # (B, 3072)
    x_clean = x_clean.squeeze(1)

    num   = (x_hat * x_clean).sum(dim=1)
    denom = x_hat.norm(dim=1) * x_clean.norm(dim=1) + 1e-8
    return (num / denom).mean().item()


def _train_step(model, x_noisy, optimizer, lambda_h, lambda_sym, lambda_cons, device):
    x_noisy = x_noisy.to(device)
    optimizer.zero_grad()

    x_hat, z, H = model(x_noisy)

    recon        = F.mse_loss(x_hat, x_noisy)
    h_loss, info = hamiltonian_loss(z, model, lambda_sym, lambda_cons)
    loss         = recon + lambda_h * h_loss

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return loss.item(), recon.item(), h_loss.item(), info


def _val_step(model, x_noisy, x_clean, lambda_h, lambda_sym, lambda_cons, device):
    x_noisy = x_noisy.to(device)

    # Reconstruction (no grad needed)
    with torch.no_grad():
        x_hat, z, H = model(x_noisy)
        recon = F.mse_loss(x_hat, x_noisy).item()

    # Hamiltonian loss (needs autograd internally; @enable_grad handles it)
    h_loss, info = hamiltonian_loss(z.detach(), model, lambda_sym, lambda_cons)
    total = recon + lambda_h * h_loss.item()

    # Optional overlap
    overlap = None
    if x_clean is not None:
        with torch.no_grad():
            overlap = waveform_overlap(x_hat, x_clean.to(device))

    return total, recon, h_loss.item(), info, overlap


# ── Main training loop ────────────────────────────────────────────────────────

def train(
    model       : HDAE,
    train_loader: DataLoader,
    val_loader  : DataLoader,
    num_epochs  : int   = 100,
    lr          : float = 1e-3,
    lambda_h    : float = 0.1,
    lambda_sym  : float = 0.01,
    lambda_cons : float = 0.1,
    device      : str   = 'cpu',
    save_dir    : str   = './checkpoints',
):
    os.makedirs(save_dir, exist_ok=True)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Cosine annealing: smoothly decays lr to 0 over all epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=lr * 0.01
    )

    best_val_loss = float('inf')
    history = []

    for epoch in range(1, num_epochs + 1):

        # ── Train ────────────────────────────────────────────────────────────
        model.train()
        t_total = t_recon = t_h = 0.0
        t_sym = t_floor = t_cons = t_Hmean = 0.0
        n_train = 0

        for batch in train_loader:
            x_noisy = batch[0]
            B = x_noisy.size(0)

            loss, recon, h, info = _train_step(
                model, x_noisy, optimizer,
                lambda_h, lambda_sym, lambda_cons, device
            )

            t_total += loss  * B;  t_recon += recon * B
            t_h     += h     * B
            t_sym   += info['sym']   * B;  t_floor += info['floor'] * B
            t_cons  += info['cons']  * B;  t_Hmean += info['H_mean']* B
            n_train += B

        scheduler.step()

        train_stats = {
            'total': t_total / n_train,   'recon': t_recon / n_train,
            'h':     t_h     / n_train,   'sym':   t_sym   / n_train,
            'floor': t_floor / n_train,   'cons':  t_cons  / n_train,
            'H_mean':t_Hmean / n_train,
        }

        # ── Validate ─────────────────────────────────────────────────────────
        model.eval()
        v_total = v_recon = v_h = 0.0
        v_sym = v_floor = v_cons = v_Hmean = 0.0
        overlaps = []
        n_val = 0

        for batch in val_loader:
            x_noisy = batch[0]
            x_clean = batch[1] if len(batch) > 1 else None
            B = x_noisy.size(0)

            total, recon, h, info, overlap = _val_step(
                model, x_noisy, x_clean,
                lambda_h, lambda_sym, lambda_cons, device
            )

            v_total += total * B;   v_recon += recon * B
            v_h     += h     * B
            v_sym   += info['sym']   * B;  v_floor += info['floor'] * B
            v_cons  += info['cons']  * B;  v_Hmean += info['H_mean']* B
            if overlap is not None:
                overlaps.append(overlap * B)
            n_val += B

        val_stats = {
            'total': v_total / n_val,   'recon': v_recon / n_val,
            'h':     v_h     / n_val,   'sym':   v_sym   / n_val,
            'floor': v_floor / n_val,   'cons':  v_cons  / n_val,
            'H_mean':v_Hmean / n_val,
            'overlap': sum(overlaps) / n_val if overlaps else None,
        }

        history.append({'epoch': epoch, 'train': train_stats, 'val': val_stats})

        # ── Print ────────────────────────────────────────────────────────────
        overlap_str = (
            f"  overlap={val_stats['overlap']:.4f}"
            if val_stats['overlap'] is not None else ''
        )
        print(
            f"Epoch {epoch:>3}/{num_epochs} │ "
            f"train  recon={train_stats['recon']:.4f}  h={train_stats['h']:.4f} │ "
            f"val  recon={val_stats['recon']:.4f}  h={val_stats['h']:.4f}  "
            f"H̄={val_stats['H_mean']:.3f}  sym={val_stats['sym']:.4f}"
            f"{overlap_str}"
        )

        # ── Checkpoint ───────────────────────────────────────────────────────
        if val_stats['total'] < best_val_loss:
            best_val_loss = val_stats['total']
            torch.save(
                {
                    'epoch':           epoch,
                    'model_state':     model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'val_loss':        best_val_loss,
                    'val_stats':       val_stats,
                },
                os.path.join(save_dir, 'best.pt'),
            )
            print(f"  ✓ saved best checkpoint (val_loss={best_val_loss:.6f})")

    # Save final weights separately
    torch.save(model.state_dict(), os.path.join(save_dir, 'final.pt'))
    return history


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Train H-DAE on LIGO strain data')

    # Data
    p.add_argument('--train_noisy', required=True,
                   help='(N, 3072) float32 .npy — whitened noisy strain (train)')
    p.add_argument('--val_noisy',   required=True,
                   help='(N, 3072) float32 .npy — whitened noisy strain (val)')
    p.add_argument('--train_clean', default=None,
                   help='(N, 3072) float32 .npy — clean waveform (optional, eval only)')
    p.add_argument('--val_clean',   default=None,
                   help='(N, 3072) float32 .npy — clean waveform (optional, eval only)')

    # Architecture
    p.add_argument('--latent_dim', type=int,   default=64,
                   help='Total latent dim; split evenly into q (32) and p (32)')

    # Training
    p.add_argument('--epochs',      type=int,   default=100)
    p.add_argument('--batch_size',  type=int,   default=64)
    p.add_argument('--lr',          type=float, default=1e-3)

    # Loss weights
    p.add_argument('--lambda_h',    type=float, default=0.1,
                   help='Weight on the full Hamiltonian loss term')
    p.add_argument('--lambda_sym',  type=float, default=0.01,
                   help='Weight on symplectic residual inside H-loss')
    p.add_argument('--lambda_cons', type=float, default=0.1,
                   help='Weight on energy floor + conservation inside H-loss')

    # Misc
    p.add_argument('--device',   default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--save_dir', default='./checkpoints')
    p.add_argument('--num_workers', type=int, default=4)

    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # ── Load arrays ──────────────────────────────────────────────────────────
    train_noisy = np.load(args.train_noisy)
    val_noisy   = np.load(args.val_noisy)
    train_clean = np.load(args.train_clean) if args.train_clean else None
    val_clean   = np.load(args.val_clean)   if args.val_clean   else None

    print(f"Train: {len(train_noisy)} samples   Val: {len(val_noisy)} samples")

    # ── Datasets & loaders ───────────────────────────────────────────────────
    train_ds = LIGODataset(train_noisy, train_clean)
    val_ds   = LIGODataset(val_noisy,   val_clean)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # ── Model ────────────────────────────────────────────────────────────────
    model = HDAE(latent_dim=args.latent_dim)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    print(f"Device    : {args.device}")
    print(
        f"Loss weights: λ_h={args.lambda_h}  "
        f"λ_sym={args.lambda_sym}  λ_cons={args.lambda_cons}"
    )
    print()

    # ── Train ────────────────────────────────────────────────────────────────
    history = train(
        model, train_loader, val_loader,
        num_epochs   = args.epochs,
        lr           = args.lr,
        lambda_h     = args.lambda_h,
        lambda_sym   = args.lambda_sym,
        lambda_cons  = args.lambda_cons,
        device       = args.device,
        save_dir     = args.save_dir,
    )