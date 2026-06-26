"""
model.py — Hamiltonian Denoising Autoencoder (H-DAE)

Architecture
------------
Encoder      : Conv1d [1→16→32→64], kernel=16, stride=2, LeakyReLU
               Linear → z ∈ R^64
Latent split : z[:32] = q (positions), z[32:] = p (momenta)
Decoder      : ConvTranspose1d mirrors encoder + additive skip connections
H-MLP        : (q, p) → scalar H(q,p), Tanh activations for smooth landscape

Input  : (B, 1, 3072)  — whitened noisy LIGO strain
Output : (B, 1, 3072)  — reconstructed clean strain
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


class HDAE(nn.Module):

    def __init__(self, latent_dim: int = 64):
        super().__init__()
        assert latent_dim % 2 == 0, "latent_dim must be even (split into q and p)"
        self.latent_dim = latent_dim

        # ── Encoder ─────────────────────────────────────────────────────────
        # Conv1d shape rule: L_out = floor((L_in + 2P - K) / S) + 1
        # With K=16, S=2, P=7: L_out = (L_in - 2)/2 + 1 = L_in/2  (exact halving)
        self.enc1 = nn.Sequential(
            nn.Conv1d(1,  16, kernel_size=16, stride=2, padding=7),
            nn.LeakyReLU(0.2, inplace=True),
        )  # (B,  1, 3072) → (B, 16, 1536)

        self.enc2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=16, stride=2, padding=7),
            nn.LeakyReLU(0.2, inplace=True),
        )  # (B, 16, 1536) → (B, 32, 768)

        self.enc3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=16, stride=2, padding=7),
            nn.LeakyReLU(0.2, inplace=True),
        )  # (B, 32,  768) → (B, 64, 384)

        self.to_latent = nn.Linear(64 * 384, latent_dim)   # (B, 24576) → (B, 64)

        # ── Decoder ─────────────────────────────────────────────────────────
        # ConvTranspose1d shape rule: L_out = (L_in-1)*S - 2P + K = 2*L_in
        self.from_latent = nn.Linear(latent_dim, 64 * 384)  # (B, 64) → (B, 24576)

        self.dec1 = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=16, stride=2, padding=7),
            nn.LeakyReLU(0.2, inplace=True),
        )  # (B, 64, 384) → (B, 32, 768)

        self.dec2 = nn.Sequential(
            nn.ConvTranspose1d(32, 16, kernel_size=16, stride=2, padding=7),
            nn.LeakyReLU(0.2, inplace=True),
        )  # (B, 32, 768) → (B, 16, 1536)

        self.dec3 = nn.ConvTranspose1d(16, 1, kernel_size=16, stride=2, padding=7)
        # (B, 16, 1536) → (B,  1, 3072)

        # ── Hamiltonian MLP ──────────────────────────────────────────────────
        # Tanh → smooth, everywhere-differentiable energy surface.
        # ReLU would make ∂H/∂z piecewise-constant → poor gradient signal
        # for the symplectic constraint.
        self.hamiltonian_net = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )

    # ── Sub-graph passes ────────────────────────────────────────────────────

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, tuple]:
        """x: (B, 1, 3072) → z: (B, 64),  skips: (s1, s2, s3)"""
        s1 = self.enc1(x)                    # (B, 16, 1536)
        s2 = self.enc2(s1)                   # (B, 32,  768)
        s3 = self.enc3(s2)                   # (B, 64,  384)
        z  = self.to_latent(s3.flatten(1))   # (B, 64)
        return z, (s1, s2, s3)

    def decode(self, z: torch.Tensor, skips: tuple) -> torch.Tensor:
        """z: (B, 64) → x_hat: (B, 1, 3072), with skip connections."""
        s1, s2, s3 = skips
        x = self.from_latent(z).view(-1, 64, 384)  # (B, 64, 384)
        x = self.dec1(x + s3)                       # skip from enc3 → (B, 32, 768)
        x = self.dec2(x + s2)                       # skip from enc2 → (B, 16, 1536)
        x = self.dec3(x + s1)                       # skip from enc1 → (B,  1, 3072)
        return x

    def hamiltonian(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, 64) → H: (B, 1)  scalar energy per sample."""
        return self.hamiltonian_net(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z, skips = self.encode(x)
        x_hat    = self.decode(z, skips)
        H        = self.hamiltonian(z)
        return x_hat, z, H


# ── Hamiltonian loss ─────────────────────────────────────────────────────────

@torch.enable_grad()  # safe to call inside torch.no_grad() contexts (e.g. validation)
def hamiltonian_loss(
    z: torch.Tensor,
    model: HDAE,
    lambda_sym: float = 0.01,
    lambda_cons: float = 0.1,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Physics-informed Hamiltonian loss with three components.

    1. Symplectic residual  ‖∂H/∂p − p‖²
       ─────────────────────────────────────────────────────────────────────
       For a canonical Hamiltonian with kinetic term T = ½‖p‖²:
           ∂H/∂p = p   (canonical momentum condition)
       Enforcing this trains the H-MLP to learn a physically structured
       energy surface AND trains the encoder to produce canonical (q, p)
       coordinates where p is a true momentum.

       Implementation trick: compute ∂H/∂p on a *detached leaf* copy of z
       (so the autograd call is isolated), then use that as a target for the
       *connected* p = z[:, 32:] — gradient flows back into the encoder.

    2. Energy floor  relu(−H).mean()
       ─────────────────────────────
       Physical energies are non-negative.

    3. Conservation  H.var()
       ─────────────────────
       All BBH samples represent the same class of physical system.
       The energy manifold should be compact (H shouldn't scatter wildly
       across the batch). Penalising variance encourages this.

    Returns
    -------
    total : scalar tensor (differentiable)
    info  : dict of float values for logging
    """
    # ── Step 1: compute ∂H/∂p on a detached leaf ─────────────────────────
    z_leaf = z.detach().requires_grad_(True)        # leaf: isolated from encoder
    H_leaf = model.hamiltonian(z_leaf)              # (B, 1)

    grad_H = torch.autograd.grad(
        outputs=H_leaf.sum(),
        inputs=z_leaf,
        create_graph=False,                         # values only, no higher-order
    )[0]                                            # (B, 64)

    dH_dp = grad_H[:, 32:].detach()                # (B, 32) — stop-gradient target

    # ── Step 2: symplectic residual (trains encoder via connected z) ──────
    p        = z[:, 32:]                            # (B, 32) — in encoder graph
    sym_loss = F.mse_loss(p, dH_dp)

    # ── Step 3: energy floor + conservation (trains hamiltonian_net) ──────
    H           = model.hamiltonian(z)              # (B, 1) — connected graph
    floor_loss  = F.relu(-H).mean()
    cons_loss   = H.var()

    total = sym_loss + lambda_cons * floor_loss + lambda_cons * cons_loss

    return total, {
        'sym':    sym_loss.item(),
        'floor':  floor_loss.item(),
        'cons':   cons_loss.item(),
        'H_mean': H.mean().item(),
        'H_std':  H.std().item(),
    }


# ── Quick sanity check ───────────────────────────────────────────────────────

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model  = HDAE(latent_dim=64).to(device)
    x      = torch.randn(4, 1, 3072).to(device)

    x_hat, z, H = model(x)
    print(f"Input  : {tuple(x.shape)}")
    print(f"Output : {tuple(x_hat.shape)}")
    print(f"Latent : {tuple(z.shape)}   q={tuple(z[:,:32].shape)}  p={tuple(z[:,32:].shape)}")
    print(f"H      : {tuple(H.shape)}   values: {H.detach().squeeze().tolist()}")

    h_loss, info = hamiltonian_loss(z, model)
    print(f"\nHamiltonian loss: {h_loss.item():.6f}")
    for k, v in info.items():
        print(f"  {k:<10}: {v:.6f}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {n_params:,}")