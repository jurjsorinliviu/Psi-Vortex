"""Contemporary baselines for head-to-head comparison with Psi-Vortex.

Each baseline exposes the same interface so the comparison driver can treat them
uniformly:
    m = Baseline(...);  m.fit(V, t, I);  yhat = m.predict(V, t);  m.n_params

V, t, I are 1-D numpy arrays already max-abs normalized (V, I in ~[-1,1], t in
[0,1]) by the data loaders. All baselines model the pointwise map (V, t) -> I,
the same target the Psi-Vortex teacher/student fit, so comparisons are like-for-like.

  * MLP      -- strong plain neural baseline (the baseline the reviewers already cite).
  * ChebyKAN -- Kolmogorov-Arnold network (Chebyshev-polynomial basis); the modern
                architecture R6 asked to compare against. (pykan is not installed;
                this is a compact, faithful KAN implementation.)
  * SINDy    -- sparse identification (STLSQ over a polynomial library); the
                interpretable classical baseline. Its active-term count is its
                "parameter" count and its model is directly human-readable.
"""
import numpy as np
import torch
import torch.nn as nn


# ----------------------------------------------------------------------
# Shared torch training loop for the neural baselines
# ----------------------------------------------------------------------
def _train_torch(model, V, t, I, epochs=500, lr=5e-3, seed=0):
    torch.manual_seed(seed)
    X = torch.tensor(np.stack([V, t], axis=1), dtype=torch.float32)
    y = torch.tensor(I, dtype=torch.float32).view(-1, 1)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lf = nn.MSELoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss = lf(model(X), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return model


def _predict_torch(model, V, t):
    model.eval()
    with torch.no_grad():
        X = torch.tensor(np.stack([V, t], axis=1), dtype=torch.float32)
        return model(X).view(-1).numpy()


# ----------------------------------------------------------------------
# MLP
# ----------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, hidden=64, depth=3):
        super().__init__()
        layers, d = [], 2
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.Tanh()]
            d = hidden
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MLPBaseline:
    name = "MLP"
    def __init__(self, hidden=64, depth=3, epochs=500, seed=0):
        self.m = MLP(hidden, depth)
        self.epochs, self.seed = epochs, seed

    def fit(self, V, t, I):
        _train_torch(self.m, V, t, I, epochs=self.epochs, seed=self.seed)
        return self

    def predict(self, V, t):
        return _predict_torch(self.m, V, t)

    @property
    def n_params(self):
        return sum(p.numel() for p in self.m.parameters())


# ----------------------------------------------------------------------
# Chebyshev-KAN
# ----------------------------------------------------------------------
class ChebyKANLayer(nn.Module):
    def __init__(self, in_f, out_f, degree):
        super().__init__()
        self.degree = degree
        self.coeff = nn.Parameter(
            torch.randn(in_f, out_f, degree + 1) / (in_f * (degree + 1)) ** 0.5
        )

    def forward(self, x):
        x = torch.tanh(x)                       # keep inside Chebyshev domain
        T = [torch.ones_like(x), x]
        for k in range(2, self.degree + 1):
            T.append(2 * x * T[-1] - T[-2])
        T = torch.stack(T, dim=-1)              # [N, in_f, degree+1]
        return torch.einsum("nik,iok->no", T, self.coeff)


class ChebyKAN(nn.Module):
    def __init__(self, dims=(2, 8, 1), degree=4):
        super().__init__()
        self.layers = nn.ModuleList(
            [ChebyKANLayer(dims[i], dims[i + 1], degree) for i in range(len(dims) - 1)]
        )

    def forward(self, x):
        for lyr in self.layers:
            x = lyr(x)
        return x


class KANBaseline:
    name = "PIKAN"
    def __init__(self, dims=(2, 8, 1), degree=4, grid=5, k=3, epochs=500, seed=0):
        # Prefer the canonical pykan KAN; fall back to the compact Chebyshev-KAN.
        try:
            import warnings
            warnings.filterwarnings("ignore")
            from kan import KAN
            self.m = KAN(width=list(dims), grid=grid, k=k, seed=seed, auto_save=False)
            self.impl = "pykan"
        except Exception:
            self.m = ChebyKAN(dims, degree)
            self.impl = "cheby"
        self.epochs, self.seed = epochs, seed

    def fit(self, V, t, I):
        _train_torch(self.m, V, t, I, epochs=self.epochs, seed=self.seed)
        return self

    def predict(self, V, t):
        return _predict_torch(self.m, V, t)

    @property
    def n_params(self):
        return sum(p.numel() for p in self.m.parameters())


# ----------------------------------------------------------------------
# SINDy (STLSQ over a polynomial library)
# ----------------------------------------------------------------------
def _poly_library(V, t, degree):
    feats, names = [np.ones_like(V)], ["1"]
    for total in range(1, degree + 1):
        for a in range(total + 1):
            b = total - a
            feats.append((V ** a) * (t ** b))
            names.append(f"V^{a} t^{b}")
    return np.stack(feats, axis=1), names


class SINDyBaseline:
    name = "SINDy"
    def __init__(self, degree=2, threshold=0.05, n_iter=10):
        self.degree, self.threshold, self.n_iter = degree, threshold, n_iter

    def fit(self, V, t, I):
        Theta, self.names = _poly_library(V, t, self.degree)
        # Standardize library columns (except the constant) so STLSQ is well
        # conditioned -- standard SINDy practice; without it a collinear
        # polynomial basis yields huge cancelling coefficients that blow up
        # out of sample.
        self.mu = Theta.mean(axis=0)
        self.sigma = Theta.std(axis=0) + 1e-12
        self.mu[0], self.sigma[0] = 0.0, 1.0
        Z = (Theta - self.mu) / self.sigma
        coef = np.linalg.lstsq(Z, I, rcond=None)[0]
        for _ in range(self.n_iter):
            small = np.abs(coef) < self.threshold
            coef[small] = 0.0
            big = ~small
            if big.sum() == 0:
                break
            coef[big] = np.linalg.lstsq(Z[:, big], I, rcond=None)[0]
        self.coef = coef
        return self

    def predict(self, V, t):
        Theta, _ = _poly_library(V, t, self.degree)
        return ((Theta - self.mu) / self.sigma) @ self.coef

    @property
    def n_params(self):
        return int(np.sum(self.coef != 0.0))

    def equation(self):
        terms = [f"{c:+.3g}*{n}" for c, n in zip(self.coef, self.names) if c != 0.0]
        return "I = " + " ".join(terms)
