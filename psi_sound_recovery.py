"""
psi_sound_recovery.py — corrected, shared α-recovery pipeline.

Single source of truth for coupling-coefficient recovery, replacing the three
defects in the original analysis:
  (1) through-origin OLS  -> free-intercept slope (free_intercept_alpha)
  (2) length-1 forward    -> true-sequence processing (SeqRecoveryModel)
  (3) single trajectory   -> multi-realization training + held-out validation
                             (fit_recovery_model / recover_held_out)

Validated in diag_recovery_*.py: with the real Ψ-xLSTM blocks this recovers α on
held-out drivers (~5% mean error, held-out R²≈0.9, seed-stable) while shuffled /
absent-driver controls collapse (R²≤0.16).

Both 14_alpha_recovery_experiment.py and supplementary_experiments.py should import
from here rather than reimplementing recovery.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn

T_AMB_DEFAULT = 298.0
V_READ_DEFAULT = 0.2


# --------------------------------------------------------------------------- #
#  (1) ESTIMATOR — free-intercept slope of log-conductance vs ΔT
# --------------------------------------------------------------------------- #
def free_intercept_alpha(I_pred, dT, g_base, v_read=V_READ_DEFAULT,
                         dT_band=0.5, return_r2=True):
    """Recover α as the FREE-INTERCEPT slope of log(I/(v_read*g_base)) vs ΔT.

    A free intercept is essential: a through-origin fit conflates the mean
    conductance offset with the coupling slope and manufactures non-zero α from
    any prediction that merely matches the mean current (e.g. a constant).

    Returns (alpha_hat, r2) or alpha_hat if return_r2 is False.
    """
    I = np.asarray(I_pred, dtype=np.float64).flatten()
    dT = np.asarray(dT, dtype=np.float64).flatten()[:len(I)]
    base = v_read * g_base
    log_ratio = np.log(np.where(I > 1e-30, I, 1e-30) / base)
    mask = (np.abs(dT) > dT_band) & np.isfinite(log_ratio) & (np.abs(log_ratio) < 8.0)
    if mask.sum() < 10:
        mask = np.isfinite(log_ratio) & (np.abs(log_ratio) < 8.0)
    if mask.sum() < 5 or np.std(dT[mask]) < 1e-12 or np.std(log_ratio[mask]) < 1e-12:
        return (0.0, 0.0) if return_r2 else 0.0
    slope, _intercept = np.polyfit(dT[mask], log_ratio[mask], 1)
    r = np.corrcoef(dT[mask], log_ratio[mask])[0, 1]
    r2 = float(r * r) if np.isfinite(r) else 0.0
    return (float(slope), r2) if return_r2 else float(slope)


def latent_slope(latent, dT):
    """Amplitude-SENSITIVE replacement for the old scale-invariant |Pearson r|
    latent metric: regression slope of a 1-D latent summary against ΔT. Use this
    as coupling-strength evidence; report Pearson r separately, explicitly as a
    coupling-*timescale* (τ-shape) diagnostic, never as strength evidence."""
    latent = np.asarray(latent, dtype=np.float64).flatten()
    dT = np.asarray(dT, dtype=np.float64).flatten()[:len(latent)]
    if np.std(latent) < 1e-12 or np.std(dT) < 1e-12:
        return 0.0
    return float(np.polyfit(dT, latent, 1)[0])


# --------------------------------------------------------------------------- #
#  (2) MODEL WRAPPER — true-sequence processing (integrates driver history)
# --------------------------------------------------------------------------- #
class SeqRecoveryModel(nn.Module):
    """Wrap recurrent blocks so a driver sequence is consumed as ONE sequence
    (1, N, F), enabling the recurrence to integrate history — unlike the original
    length-1 (seq_len=1) usage. `blocks` is a callable mapping (1,N,F)->(1,N,H).

    Convenience constructor `from_psi_xlstm` builds the paper's mLSTM+sLSTM stack.
    """
    def __init__(self, blocks: nn.Module, hidden: int):
        super().__init__()
        self.blocks = blocks
        self.fc = nn.Linear(hidden, 1)

    def forward(self, X):                       # X: (N, F)
        h = self.blocks(X.unsqueeze(0))         # (1, N, H)
        return self.fc(h.squeeze(0))            # (N, 1)

    @classmethod
    def from_psi_xlstm(cls, in_dim=1, hidden=32):
        from core_psi_xlstm import mLSTMBlock, sLSTMBlock

        class _Stack(nn.Module):
            def __init__(s):
                super().__init__()
                s.m = mLSTMBlock(in_dim, hidden, memory_size=max(8, hidden // 2))
                s.s = sLSTMBlock(hidden, hidden)

            def forward(s, x):
                h1, _, _ = s.m(x)
                h2, _, _ = s.s(h1)
                return h2
        return cls(_Stack(), hidden)


# --------------------------------------------------------------------------- #
#  (3) TRAINING + HELD-OUT VALIDATION
# --------------------------------------------------------------------------- #
def log_target(I, g_base, v_read=V_READ_DEFAULT):
    I = np.asarray(I, dtype=np.float64).flatten()
    return np.log(np.maximum(I, 1e-30) / (v_read * g_base))


def fit_recovery_model(realizations, build_model, epochs=2000, lr=8e-4, clip=0.5,
                       device=None, keep_best=True):
    """Train a sequence recovery model in LOG space on MULTIPLE driver realizations.

    realizations: list of (V_tensor[N,F], y_tensor[N,1]) on `device`.
    build_model : zero-arg callable returning a fresh SeqRecoveryModel.
    Returns the trained model (best-loss weights if keep_best).
    """
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    m = build_model().to(device)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    crit = nn.MSELoss()
    best, best_loss = None, float('inf')
    for _ in range(epochs):
        m.train(); ep_losses = []
        for V, y in realizations:
            opt.zero_grad(); loss = crit(m(V), y)
            if not torch.isfinite(loss):                 # NaN guard
                opt.zero_grad(set_to_none=True); continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), clip); opt.step()
            ep_losses.append(loss.item())
        sched.step()
        # Checkpoint on the EPOCH-MEAN loss across all realizations, not the loss
        # of whichever realization happened to be last (which would be a noisy,
        # realization-order-dependent criterion when M realizations vary in difficulty).
        if keep_best and ep_losses:
            ep_mean = float(np.mean(ep_losses))
            if ep_mean < best_loss:
                best_loss = ep_mean
                best = {k: v.detach().clone() for k, v in m.state_dict().items()}
    if keep_best and best is not None:
        m.load_state_dict(best)
    return m


def recover_held_out(model, held_out, g_base, r2_accept=0.8, v_read=V_READ_DEFAULT):
    """Recover α on held-out driver realizations the model never trained on.

    held_out: list of (V_tensor[N,F], dT_array[N]).
    Returns dict: alpha_mean, alpha_std, r2_mean, identifiable (held-out R²>=r2_accept).
    """
    model.eval()
    alphas, r2s = [], []
    for V, dT in held_out:
        with torch.no_grad():
            yp = model(V).cpu().numpy().flatten()
        # The model predicts y = log(I/base); reconstruct I so the single
        # free-intercept estimator (slope + R²) is applied consistently.
        I_pred = np.exp(yp) * (v_read * g_base)
        a, r2 = free_intercept_alpha(I_pred, dT, g_base, v_read=v_read)
        alphas.append(a); r2s.append(r2)
    alphas, r2s = np.array(alphas), np.array(r2s)
    return dict(alpha_mean=float(alphas.mean()), alpha_std=float(alphas.std()),
                r2_mean=float(r2s.mean()),
                identifiable=bool(r2s.mean() >= r2_accept))
