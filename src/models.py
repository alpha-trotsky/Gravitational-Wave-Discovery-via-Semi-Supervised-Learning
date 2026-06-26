"""
models.py — CNNEncoder, CNNDecoder, HamiltonianMLP, HAE, BaselineCAE.

Architecture summary (default latent_dim=2):
  Encoder : (B, 1, 1792) → 4 strided Conv1d blocks → 1x1 proj → (B, 4, 112)
  Decoder : (B, 4, 112) → 1x1 proj → 4 ConvTranspose1d blocks → (B, 1, 1792)
  H-MLP   : (2*latent_dim,) → Linear(64) → Tanh → Linear(64) → Tanh → Linear(1)

Latent shape: (B, 2*latent_dim, 112) split into q=(B, latent_dim, 112), p=(B, latent_dim, 112).
T_latent = 1792 / 2^4 = 112, which falls in the requested [56, 112] range.

VAE  = Hamiltonian Autoencoder enforced via variance conservation(physics-constrained, trained on clean signals)
CAE  = Convolutional Autoencoder baseline (same architecture, no physics loss)
HAE = Hamiltonian Autoencoder (same architecture as CAE, but with hamiltonianphysics loss)
P-HAE = Port-Hamiltonian Autoencoder 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Helpers and Loss Functions
# ──────────────────────────────────────────────────────────────────────────────

def _gn(num_channels: int) -> nn.GroupNorm:
    """GroupNorm with up to 8 groups, guaranteed to divide num_channels."""
    num_groups = min(8, num_channels)
    while num_channels % num_groups != 0:
        num_groups -= 1
    return nn.GroupNorm(num_groups, num_channels)


def _conv_block(in_ch, out_ch, kernel=5, stride=2):
    """Strided Conv1d → GroupNorm → GELU."""
    padding = (kernel - 1) // 2
    return nn.Sequential(
        nn.Conv1d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=padding),
        _gn(out_ch),
        nn.GELU(),
    )


def _deconv_block(in_ch, out_ch, kernel=5, stride=2):
    """ConvTranspose1d → GroupNorm → GELU.

    output_padding=1 guarantees exact 2× upsampling:
      L_out = (L-1)*2 − 2*padding + kernel + output_padding = 2*L  ✓
    """
    padding = (kernel - 1) // 2
    return nn.Sequential(
        nn.ConvTranspose1d(in_ch, out_ch, kernel_size=kernel, stride=stride,
                           padding=padding, output_padding=1),
        _gn(out_ch),
        nn.GELU(),
    )


def compute_latent_velocity(z, dt=1.0):
    """
    Computes the actual velocity z_dot via central finite differences.
    z shape: (Batch, Channels, Time)
    """
    z_dot = torch.zeros_like(z)
    
    # Interior points: central difference
    z_dot[:, :, 1:-1] = (z[:, :, 2:] - z[:, :, :-2]) / (2 * dt)
    
    # Boundaries: forward and backward difference
    z_dot[:, :, 0] = (z[:, :, 1] - z[:, :, 0]) / dt
    z_dot[:, :, -1] = (z[:, :, -1] - z[:, :, -2]) / dt
    
    return z_dot

def proper_hnn_loss(hamiltonian_net, z, dt=1.0, create_graph: bool = True):
    """
    Computes the proper Hamiltonian constraint: z_dot = J * grad_z(H)
    Returns the MSE physics loss and the computed energy time-series H_t.
    """
    B, C, T = z.shape
    D = C // 2

    # 1. Actual velocity from the encoder's trajectory
    z_dot_actual = compute_latent_velocity(z, dt)

    # 2. Predicted velocity from the Hamiltonian MLP
    # Reshape z to pass through the MLP: (B, C, T) -> (B*T, C)
    z_flat = z.permute(0, 2, 1).reshape(B * T, C)
    
    H_flat = hamiltonian_net(z_flat) # (B*T, 1)
    
    # Compute spatial gradient (grad_z H). 
    # create_graph=True is REQUIRED to backprop through this gradient to update the encoder.
    grad_H_flat = torch.autograd.grad(
        outputs=H_flat.sum(),
        inputs=z_flat,
        create_graph=create_graph,
        retain_graph=create_graph,
    )[0]
    
    # Reshape back to (B, C, T)
    grad_H = grad_H_flat.reshape(B, T, C).permute(0, 2, 1)

    # 3. Construct Symplectic Matrix J
    J = torch.zeros(C, C, device=z.device)
    J[:D, D:] = torch.eye(D)
    J[D:, :D] = -torch.eye(D)

    # Predict velocity: z_dot_pred = J @ grad_H
    # Einstein summation handles the batch/time axes cleanly:
    z_dot_pred = torch.einsum('ij,bjt->bit', J, grad_H)

    # 4. Compute Loss
    phys_loss = F.mse_loss(z_dot_actual, z_dot_pred)
    H_t = H_flat.reshape(B, T)
    
    return phys_loss, H_t

def port_hamiltonian_loss(hamiltonian_net, R_matrix, z, dt=1.0, create_graph: bool = True):
    """
    Computes the Port-Hamiltonian constraint: z_dot = (J - R) * grad_z(H)
    Allows monotonic energy dissipation parameterized by the PSD matrix R.
    """
    B, C, T = z.shape
    D = C // 2

    z_dot_actual = compute_latent_velocity(z, dt)
    z_flat = z.permute(0, 2, 1).reshape(B * T, C)
    
    H_flat = hamiltonian_net(z_flat)
    
    grad_H_flat = torch.autograd.grad(
        outputs=H_flat.sum(),
        inputs=z_flat,
        create_graph=create_graph,
        retain_graph=create_graph,
    )[0]
    
    grad_H = grad_H_flat.reshape(B, T, C).permute(0, 2, 1)

    J = torch.zeros(C, C, device=z.device)
    J[:D, D:] = torch.eye(D)
    J[D:, :D] = -torch.eye(D)

    # Port-Hamiltonian Dynamics Matrix
    M = J - R_matrix

    # z_dot_pred = (J - R) @ grad_H
    z_dot_pred = torch.einsum('ij,bjt->bit', M, grad_H)

    phys_loss = F.mse_loss(z_dot_actual, z_dot_pred)
    H_t = H_flat.reshape(B, T)
    
    return phys_loss, H_t


# ──────────────────────────────────────────────────────────────────────────────
# Encoder
# ──────────────────────────────────────────────────────────────────────────────

class CNNEncoder(nn.Module):
    """
    (B, 1, 1792) → (B, 2*latent_dim, 112)

    Channels : 1 → 16 → 32 → 64 → 128 → 2*latent_dim
    Spatial  : 1792 → 896 → 448 → 224 → 112  (stride-2 halves each step)
    """
    def __init__(self, latent_dim: int = 2):
        super().__init__()
        self.latent_dim = latent_dim
        self.blocks = nn.Sequential(
            _conv_block(1,   16),
            _conv_block(16,  32),
            _conv_block(32,  64),
            _conv_block(64, 128),
        )
        self.proj = nn.Conv1d(128, 2 * latent_dim, kernel_size=1)

    def forward(self, x):
        return self.proj(self.blocks(x))   # (B, 2*latent_dim, 112)


# ──────────────────────────────────────────────────────────────────────────────
# Decoder
# ──────────────────────────────────────────────────────────────────────────────

class CNNDecoder(nn.Module):
    """
    (B, 2*latent_dim, 112) → (B, 1, 1792)  — mirrors encoder exactly.
    """
    def __init__(self, latent_dim: int = 2):
        super().__init__()
        self.latent_dim = latent_dim
        self.proj = nn.Conv1d(2 * latent_dim, 128, kernel_size=1)
        self.blocks = nn.Sequential(
            _deconv_block(128, 64),
            _deconv_block(64,  32),
            _deconv_block(32,  16),
        )
        self.final = nn.ConvTranspose1d(16, 1, kernel_size=5, stride=2,
                                        padding=2, output_padding=1)

    def forward(self, z):
        return self.final(self.blocks(self.proj(z)))   # (B, 1, 1792)

class HamiltonianMLP(nn.Module):
    """
    Learned scalar energy H: R^(2*latent_dim) → R.
    """
    def __init__(self, latent_dim: int = 2):
        super().__init__()
        in_dim = 2 * latent_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.Tanh(),
            nn.Linear(64,     64), nn.Tanh(),
            nn.Linear(64,      1),
        )

    def forward(self, qp):
        return self.net(qp)


# ──────────────────────────────────────────────────────────────────────────────
# Variance-based Hamiltonian Autoencoder (VHAE) - Renamed Baseline
# ──────────────────────────────────────────────────────────────────────────────

class VHAE(nn.Module):
    """
    Original Variance-based Hamiltonian Autoencoder.
    Penalizes only the variance of H over time (weak formulation).
    """
    def __init__(self, latent_dim: int = 2):
        super().__init__()
        self.physics_type = 'variance'
        self.latent_dim  = latent_dim
        self.encoder     = CNNEncoder(latent_dim)
        self.decoder     = CNNDecoder(latent_dim)
        self.hamiltonian = HamiltonianMLP(latent_dim)

    def forward(self, x):
        z = self.encoder(x)

        # Vectorised evaluation (no second-order gradients needed)
        B, C, T = z.shape
        z_flat = z.permute(0, 2, 1).reshape(B * T, C)
        H_flat = self.hamiltonian(z_flat)
        H_t = H_flat.reshape(B, T)

        D = C // 2
        q = z[:, :D, :]
        p = z[:, D:, :]

        x_hat = self.decoder(z)
        return x_hat, q, p, H_t


# ──────────────────────────────────────────────────────────────────────────────
# Proper Hamiltonian Autoencoder (ProperHAE)
# ──────────────────────────────────────────────────────────────────────────────

class ProperHAE(nn.Module):
    """
    Proper Hamiltonian Autoencoder.
    Enforces strictly conservative Hamiltonian mechanics (z_dot = J grad_H).
    H is computed exclusively inside proper_hnn_loss (with the required graph).
    """
    def __init__(self, latent_dim: int = 2):
        super().__init__()
        self.physics_type = 'proper'
        self.latent_dim  = latent_dim
        self.encoder     = CNNEncoder(latent_dim)
        self.decoder     = CNNDecoder(latent_dim)
        self.hamiltonian = HamiltonianMLP(latent_dim)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        D = z.shape[1] // 2
        q = z[:, :D, :]
        p = z[:, D:, :]
        return x_hat, q, p, None


# ──────────────────────────────────────────────────────────────────────────────
# Port-Hamiltonian Autoencoder (PortHAE)
# ──────────────────────────────────────────────────────────────────────────────

class PortHAE(nn.Module):
    """
    Port-Hamiltonian Autoencoder.
    Enforces dissipative mechanics suitable for BBH inspirals.
    Learns an internal dissipation matrix R.
    H is computed exclusively inside port_hamiltonian_loss (with the required graph).
    """
    def __init__(self, latent_dim: int = 2):
        super().__init__()
        self.physics_type = 'port'
        self.latent_dim  = latent_dim
        self.encoder     = CNNEncoder(latent_dim)
        self.decoder     = CNNDecoder(latent_dim)
        self.hamiltonian = HamiltonianMLP(latent_dim)

        # Cholesky factor L for the dissipation matrix R = L Lᵀ (guaranteed PSD).
        in_dim = 2 * latent_dim
        self.L_matrix = nn.Parameter(torch.eye(in_dim) * 0.1)

    def get_R(self):
        """Returns the PSD dissipation matrix R = L Lᵀ."""
        return self.L_matrix @ self.L_matrix.T

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        D = z.shape[1] // 2
        q = z[:, :D, :]
        p = z[:, D:, :]
        return x_hat, q, p, None
        
# ──────────────────────────────────────────────────────────────────────────────
# Baseline Convolutional Autoencoder (CAE)
# ──────────────────────────────────────────────────────────────────────────────

class BaselineCAE(nn.Module):
    """
    Vanilla CNN autoencoder — identical encoder/decoder to HAE, no physics loss.
    Forward returns (x_hat, z) for phase-space comparison plots.
    """
    def __init__(self, latent_dim: int = 2):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder    = CNNEncoder(latent_dim)
        self.decoder    = CNNDecoder(latent_dim)

    def forward(self, x):
        z     = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z


# Backwards-compatible alias: older code expects `HAE` class name.
class HAE(VHAE):
    pass
