"""
losses.py — Physics (Hamiltonian) loss for H-DAE.

Physics loss: the Hamiltonian should be approximately conserved along the
inspiral trajectory. We enforce this by minimising the *variance* of H_t
over the time dimension — a small variance means H barely changes.

Crucially, no second-order autograd is needed here (no create_graph=True).
The gradient flows only through the MLP weights and the encoder outputs,
which is exactly what we want for the inductive bias.
"""

import torch
import torch.nn.functional as F


def hamiltonian_loss(H_t: torch.Tensor) -> torch.Tensor:
    """
    Compute the Hamiltonian conservation loss.

    Args:
        H_t : (B, T) — scalar Hamiltonian value at each latent timestep,
              as returned by HDAE.forward().

    Returns:
        Scalar loss = mean over batch of time-variance of H.
        A perfect conservative system would have loss == 0.
    """
    # var(dim=1) gives the variance along the time axis for each sample (B,)
    # mean() aggregates over the batch
    return H_t.var(dim=1).mean()


def reconstruction_loss(x_hat: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE between reconstruction and clean-signal target."""
    return F.mse_loss(x_hat, target)


def total_hdae_loss(x_hat, target, H_t, lambda_phys: float):
    """Combined loss for H-DAE: MSE + lambda_phys * Hamiltonian conservation loss."""
    recon = reconstruction_loss(x_hat, target)
    phys  = hamiltonian_loss(H_t)
    return recon + lambda_phys * phys, recon, phys
