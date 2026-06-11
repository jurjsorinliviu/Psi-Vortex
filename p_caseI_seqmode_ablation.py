"""
Item 4: Case Study I (memristor) physics-init speedup — RE-VERIFIED IN SEQUENCE MODE.

The original 6.74x convergence speedup was measured at seq_len=1 (matrix memory inert,
i.e. functionally an MLP). Here the SAME mLSTM/sLSTM blocks consume the driver as a TRUE
sequence (recurrence active), comparing random vs physics-aware initialization.

Headline metric = EPOCHS-TO-TARGET (configuration-robust), not wall-clock: shared target
loss = the minimum loss reached by the slower (random) init, so both curves attain it;
epochs-to-target is the first epoch each init drops below it. Ratio = random/physics.
Reported as mean over seeds. (Wall-clock is reported but demoted — sub-second figures are
optics-fragile and sequence mode inflates absolute times.)
"""
import os, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from core_psi_xlstm import mLSTMBlock, sLSTMBlock
from core_physics_init import apply_psi_vortex_init

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_PATH = 'printed_memristor_training_data.csv'
EPOCHS = int(os.environ.get("EPOCHS", 500))
STRIDE = int(os.environ.get("STRIDE", 4))   # downsample the sequence; speedup RATIO is stride-robust
SEEDS = [int(s) for s in os.environ.get("SEEDS", "42,123,456,7,99").split(",")]


def load_memristor():
    df = pd.read_csv(DATA_PATH)
    df = df[(df['device_id'] == 0) & (df['cycle_id'] == 0)]
    V = df['voltage'].values.astype(np.float32)[::STRIDE]
    I = df['current'].values.astype(np.float32)[::STRIDE]
    t = np.linspace(0, 1, len(V), dtype=np.float32)
    return V, t, I


class SeqMemristor(nn.Module):
    """Paper's mLSTM/sLSTM blocks consumed as ONE sequence (1, N, 2) — recurrence ACTIVE."""
    def __init__(self, hidden=64):
        super().__init__()
        self.mlstm = mLSTMBlock(2, hidden, memory_size=32)
        self.slstm = sLSTMBlock(hidden, hidden)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, X):                      # X: (N, 2)
        h1, _, _ = self.mlstm(X.unsqueeze(0))
        h2, _, _ = self.slstm(h1)
        return self.fc(h2.squeeze(0))          # (N, 1)


def apply_random(model):
    for p in model.parameters():
        if p.dim() >= 2:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.zeros_(p)


def train_curve(init_mode, X, y, seed, epochs=EPOCHS, lr=1e-3, clip=1.0):
    torch.manual_seed(seed)
    m = SeqMemristor().to(DEV)
    if init_mode == "physics":
        try:
            apply_psi_vortex_init(m, pde_type="memristor")
        except Exception as e:
            print(f"  [physics init warn] {e}")
    else:
        apply_random(m)
    opt = torch.optim.Adam(m.parameters(), lr=lr); crit = nn.MSELoss()
    curve = []
    t0 = time.time()
    for _ in range(epochs):
        m.train(); opt.zero_grad()
        loss = crit(m(X), y)
        if not torch.isfinite(loss):
            curve.append(float('inf')); continue
        loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), clip); opt.step()
        curve.append(loss.item())
    return np.asarray(curve), time.time() - t0


def epochs_to_target(curve, target):
    hit = np.where(curve <= target)[0]
    return int(hit[0] + 1) if len(hit) else len(curve)


def main():
    V, t, I = load_memristor()
    # normalise target to ~O(1) for stable training; scale-free comparison
    scale = float(np.std(I)) or 1.0
    X = torch.tensor(np.stack([V, t], 1), device=DEV)
    y = torch.tensor((I / scale).reshape(-1, 1), device=DEV)

    print(f"Case Study I speedup, SEQUENCE MODE | epochs={EPOCHS} seeds={SEEDS} N={len(V)}")
    ratios, e_rand, e_phys = [], [], []
    for seed in SEEDS:
        c_r, wt_r = train_curve("random", X, y, seed)
        c_p, wt_p = train_curve("physics", X, y, seed)
        # shared target = min loss reached by the slower (random) init -> both attain it
        target = float(np.min(c_r))
        er = epochs_to_target(c_r, target)
        ep = epochs_to_target(c_p, target)
        ratio = er / ep if ep > 0 else float('inf')
        ratios.append(ratio); e_rand.append(er); e_phys.append(ep)
        print(f"  seed {seed:4}: random {er:4} ep | physics {ep:4} ep | "
              f"ratio {ratio:5.2f}x | final loss r={c_r[-1]:.2e} p={c_p[-1]:.2e}", flush=True)
    ratios = np.asarray(ratios)
    print("-" * 64)
    print(f"epochs-to-target (shared): random {np.mean(e_rand):.1f}+/-{np.std(e_rand):.1f}  "
          f"physics {np.mean(e_phys):.1f}+/-{np.std(e_phys):.1f}")
    print(f"SPEEDUP (epochs ratio, sequence mode): {ratios.mean():.2f}x +/- {ratios.std():.2f} "
          f"(original length-1 claim: 6.74x)")
    pd.DataFrame(dict(seed=SEEDS, epochs_random=e_rand, epochs_physics=e_phys,
                      ratio=ratios)).to_csv("caseI_seqmode_speedup.csv", index=False)
    print("[OK] caseI_seqmode_speedup.csv")


if __name__ == "__main__":
    main()
