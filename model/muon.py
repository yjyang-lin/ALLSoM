# https://github.com/KellerJordan/Muon
# https://github.com/NoahAmsel/PolarExpress

from math import inf, sqrt

import numpy as np
import torch


def optimal_quintic(l, u):
    """Compute the minimax odd quintic approximation on [l, u]."""
    assert 0 <= l <= u

    if 1 - 5e-6 <= l / u:
        return (15 / 8) / u, (-10 / 8) / u**3, (3 / 8) / u**5

    q, r = (3 * l + u) / 4, (l + 3 * u) / 4
    E, old_E = inf, None

    while old_E is None or abs(old_E - E) > 1e-15:
        old_E = E
        lhs = np.array([
            [l, l**3, l**5, 1],
            [q, q**3, q**5, -1],
            [r, r**3, r**5, 1],
            [u, u**3, u**5, -1],
        ])
        a, b, c, E = np.linalg.solve(lhs, np.ones(4))
        q, r = np.sqrt(
            (-3 * b + np.array([-1, 1]) * sqrt(9 * b**2 - 20 * a * c))
            / (10 * c)
        )

    return float(a), float(b), float(c)


def optimal_composition(l, num_iters, safety_factor_eps=0.0, cushion=0.0):
    """Generate the Polar Express polynomial sequence."""
    u = 1.0
    assert 0 <= l <= u

    safety_factor = 1 + safety_factor_eps
    coeffs = []

    for i in range(num_iters):
        a, b, c = optimal_quintic(max(l, cushion * u), u)

        if cushion * u > l:
            pl = a * l + b * l**3 + c * l**5
            pu = a * u + b * u**3 + c * u**5
            scale = 2 / (pl + pu)
            a, b, c = a * scale, b * scale, c * scale

        if i < num_iters - 1:
            a /= safety_factor
            b /= safety_factor**3
            c /= safety_factor**5

        coeffs.append((a, b, c))
        l = a * l + b * l**3 + c * l**5
        u = 2 - l

    return coeffs


# Computed once when this module is imported.
POLAR_EXPRESS_COEFFS = optimal_composition(
    l=1e-3,
    num_iters=10,
    safety_factor_eps=1e-2,
    cushion=0.02,
)


def polar_express(G, steps=5):
    """Approximate the polar factor using Polar Express."""
    assert G.ndim >= 2

    X = G.float()
    transposed = G.size(-2) > G.size(-1)

    if transposed:
        X = X.mT

    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-7)

    for a, b, c in POLAR_EXPRESS_COEFFS[:steps]:
        A = X @ X.mT
        X = a * X + (b * A + c * (A @ A)) @ X

    return X.mT if transposed else X


def muon_update(grad, momentum, beta=0.95, steps=5, nesterov=True):
    """Muon update with Polar Express orthogonalization."""
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp(momentum, beta) if nesterov else momentum

    # Keep the original behavior: flatten Conv2d, but leave Conv1d batched.
    if update.ndim == 4:
        update = update.view(update.size(0), -1)

    update = polar_express(update, steps)

    # RMS scaling for using AdamW-scale learning rates.
    update *= 0.2 * sqrt(max(update.size(-2), update.size(-1)))
    return update


def adam_update(grad, exp_avg, exp_avg_sq, step, betas, eps):
    """Standard Adam update used for non-Muon parameters."""
    beta1, beta2 = betas
    exp_avg.lerp_(grad, 1 - beta1)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2)

    m = exp_avg / (1 - beta1**step)
    v = exp_avg_sq / (1 - beta2**step)
    return m / (v.sqrt() + eps)


class SingleDeviceMuonWithAuxAdam(torch.optim.Optimizer):
    """Use Muon for selected matrix parameters and AdamW for the rest."""

    def __init__(self, param_groups):
        for group in param_groups:
            if "use_muon" not in group:
                raise ValueError("Each parameter group must define 'use_muon'.")

            if group["use_muon"]:
                group.setdefault("lr", 0.02)
                group.setdefault("momentum", 0.95)
                group.setdefault("weight_decay", 0.0)
            else:
                group.setdefault("lr", 3e-4)
                group.setdefault("betas", (0.9, 0.95))
                group.setdefault("eps", 1e-10)
                group.setdefault("weight_decay", 0.0)

        super().__init__(param_groups, {})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr, wd = group["lr"], group["weight_decay"]

            if group["use_muon"]:
                for p in group["params"]:
                    if p.grad is None:
                        continue

                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(p)

                    update = muon_update(
                        p.grad,
                        state["momentum_buffer"],
                        beta=group["momentum"],
                    )

                    p.mul_(1 - lr * wd)
                    p.add_(update.reshape_as(p), alpha=-lr)

            else:
                for p in group["params"]:
                    if p.grad is None:
                        continue

                    state = self.state[p]
                    if "exp_avg" not in state:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0

                    state["step"] += 1
                    update = adam_update(
                        p.grad,
                        state["exp_avg"],
                        state["exp_avg_sq"],
                        state["step"],
                        group["betas"],
                        group["eps"],
                    )

                    p.mul_(1 - lr * wd)
                    p.add_(update, alpha=-lr)

        return loss