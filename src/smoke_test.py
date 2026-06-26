"""
smoke_test.py — Forward/backward/val sanity check for all four model variants.

Runs without any HDF file. Uses synthetic (B=4, 1, 1792) data.
Exit 0 = all checks passed.  Exit 1 = at least one failure.

Usage:
  python src/smoke_test.py
"""

import sys, os
# Ensure src/ is on the path so `core.*` and `evaluation.*` imports resolve.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from torch.utils.data import DataLoader, TensorDataset

from core.models     import BaselineCAE, VHAE, HAE, ProperHAE, PortHAE
from core.losses     import reconstruction_loss, total_hdae_loss
from core.train      import _train_epoch, _val_epoch
from evaluation.utils import collect_outputs

DEVICE  = torch.device("cpu")
B, C, T = 4, 1, 1792
LATENT  = 2
PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"

failures = []


def check(label, cond, detail=""):
    if cond:
        print(f"  {PASS}  {label}")
    else:
        print(f"  {FAIL}  {label}" + (f"  ({detail})" if detail else ""))
        failures.append(label)


def make_loader(n=8):
    x = torch.randn(n, C, T)
    return DataLoader(TensorDataset(x, x), batch_size=B)


# ── 1. physics_type set in __init__ (not after first forward) ────────────────
print("\n[1] physics_type set in __init__")
for cls, expected in [(VHAE, 'variance'), (HAE, 'variance'),
                      (ProperHAE, 'proper'), (PortHAE, 'port')]:
    m = cls(latent_dim=LATENT)
    check(f"{cls.__name__}.physics_type == '{expected}' before forward",
          getattr(m, 'physics_type', None) == expected)
check("BaselineCAE has no physics_type",
      getattr(BaselineCAE(latent_dim=LATENT), 'physics_type', None) is None)


# ── 2. forward() output shapes and H_t contract ──────────────────────────────
print("\n[2] forward() output shapes")
x_dummy = torch.randn(B, C, T)

cae = BaselineCAE(latent_dim=LATENT)
x_hat, z = cae(x_dummy)
check("CAE: x_hat shape", x_hat.shape == (B, C, T),           x_hat.shape)
check("CAE: z shape",     z.shape == (B, 2 * LATENT, 112),    z.shape)

for cls, name, expect_H in [(VHAE, 'VHAE', True),
                              (ProperHAE, 'ProperHAE', False),
                              (PortHAE,   'PortHAE',   False)]:
    m   = cls(latent_dim=LATENT)
    out = m(x_dummy)
    check(f"{name}: returns 4-tuple",  isinstance(out, tuple) and len(out) == 4)
    x_hat, q, p, H_t = out
    check(f"{name}: x_hat shape",      x_hat.shape == (B, C, T),         x_hat.shape)
    check(f"{name}: q shape",          q.shape == (B, LATENT, 112),      q.shape)
    check(f"{name}: p shape",          p.shape == (B, LATENT, 112),      p.shape)
    if expect_H:
        check(f"{name}: H_t is (B,T) tensor",
              isinstance(H_t, torch.Tensor) and H_t.shape == (B, 112), str(H_t))
    else:
        check(f"{name}: H_t is None (no double-H-call)",
              H_t is None, str(H_t))


# ── 3. One training step per model ───────────────────────────────────────────
print("\n[3] Training step (loss.backward)")

def one_train_step(model, lam):
    loader = make_loader()
    opt    = torch.optim.Adam(model.parameters(), lr=1e-3)
    try:
        result = _train_epoch(model, loader, opt, DEVICE, lambda_phys=lam)
        check(f"{type(model).__name__}: train step completes", True)
        check(f"{type(model).__name__}: loss is finite",
              torch.isfinite(torch.tensor(result['loss'])), result['loss'])
    except Exception as e:
        check(f"{type(model).__name__}: train step completes", False, str(e))

one_train_step(BaselineCAE(latent_dim=LATENT), lam=0.0)
one_train_step(VHAE(latent_dim=LATENT),        lam=0.1)
one_train_step(ProperHAE(latent_dim=LATENT),   lam=0.1)
one_train_step(PortHAE(latent_dim=LATENT),     lam=0.1)


# ── 4. One validation step per model ─────────────────────────────────────────
print("\n[4] Validation step (no backward, correct no_grad gating)")

def one_val_step(model, lam):
    loader = make_loader()
    try:
        result = _val_epoch(model, loader, DEVICE, lambda_phys=lam)
        check(f"{type(model).__name__}: val step completes", True)
        check(f"{type(model).__name__}: val loss finite",
              torch.isfinite(torch.tensor(result['loss'])), result['loss'])
    except Exception as e:
        check(f"{type(model).__name__}: val step completes", False, str(e))

one_val_step(BaselineCAE(latent_dim=LATENT), lam=0.0)
one_val_step(VHAE(latent_dim=LATENT),        lam=0.1)
one_val_step(ProperHAE(latent_dim=LATENT),   lam=0.1)
one_val_step(PortHAE(latent_dim=LATENT),     lam=0.1)


# ── 5. collect_outputs works for all four model types ────────────────────────
print("\n[5] evaluation.utils.collect_outputs dispatch")

loader = make_loader()
for cls, name in [(BaselineCAE, 'BaselineCAE'), (VHAE, 'VHAE'),
                   (ProperHAE, 'ProperHAE'), (PortHAE, 'PortHAE')]:
    m = cls(latent_dim=LATENT)
    try:
        x_all, y_all, xhat_all, z_all = collect_outputs(m, loader, DEVICE)
        check(f"{name}: collect_outputs shapes ok",
              xhat_all.shape == x_all.shape and z_all.shape[0] == x_all.shape[0])
    except Exception as e:
        check(f"{name}: collect_outputs shapes ok", False, str(e))


# ── 6. physics_type stable across repeated forward calls ─────────────────────
print("\n[6] physics_type stability across repeated calls")
for cls, expected in [(VHAE, 'variance'), (ProperHAE, 'proper'), (PortHAE, 'port')]:
    m = cls(latent_dim=LATENT)
    for _ in range(3):
        m(x_dummy)
    check(f"{cls.__name__}: physics_type unchanged after 3 forwards",
          m.physics_type == expected)


# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
if failures:
    print(f"FAILED ({len(failures)} checks):")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
else:
    print("All checks passed.")
    sys.exit(0)
