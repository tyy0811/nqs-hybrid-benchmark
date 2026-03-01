"""
04: Time-Conditional NQS (t-NQS) — Causal Interval Training (Hardened)
=====================================================================
1D TFIM quench dynamics via a from-scratch t-NQS implementation in JAX + Flax.

This hardened causal version adds five targeted fixes:
  1. hard parity projection into a chosen global spin-flip sector,
  2. an in-window energy conservation penalty relative to the left edge,
  3. shorter default causal windows,
  4. more carried-forward anchors per window,
  5. stronger weighting of early-window residuals after each handoff.

Sign convention:
  H = −J Σ Z_iZ_j − h Σ X_i  (ferromagnetic for J > 0)

Requirements:
    jax, flax, optax, numpy, matplotlib, scipy
    Python >= 3.11
"""

import argparse
import csv
import time as time_module
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

try:
    import jax
    import jax.numpy as jnp
    import flax.linen as nn
    import optax
    from scipy.linalg import expm
    JAX_AVAILABLE = True
except ImportError as e:
    JAX_AVAILABLE = False
    _IMPORT_ERROR = str(e)


# ═══════════════════════════════════════════════════════════════
# TFIM HAMILTONIAN
# ═══════════════════════════════════════════════════════════════

def build_tfim_matrix(N, h, J=1.0, pbc=True):
    """
    H = −J Σ Z_iZ_{i+1} − h Σ X_i  (ferromagnetic for J > 0).

    Returns dense 2^N × 2^N matrix (complex128).
    """
    dim = 2**N
    H = np.zeros((dim, dim), dtype=np.complex128)
    I2 = np.eye(2, dtype=np.complex128)
    SX = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    SZ = np.array([[1, 0], [0, -1]], dtype=np.complex128)

    def kron_chain(ops):
        result = ops[0]
        for op in ops[1:]:
            result = np.kron(result, op)
        return result

    n_bonds = N if pbc else N - 1
    for i in range(n_bonds):
        j = (i + 1) % N
        ops = [I2] * N
        ops[i] = SZ
        ops[j] = SZ
        H -= J * kron_chain(ops)
    for i in range(N):
        ops = [I2] * N
        ops[i] = SX
        H -= h * kron_chain(ops)
    return H


def all_spin_configs(N):
    """
    Enumerate all 2^N spin configurations in {+1, −1}^N.

    Returns array of shape (2^N, N), ordered to match the standard
    computational-basis ordering used by our Hamiltonian construction.
    """
    configs = np.zeros((2**N, N), dtype=np.float32)
    for idx in range(2**N):
        for bit in range(N):
            configs[idx, bit] = 1.0 - 2.0 * ((idx >> bit) & 1)
    return configs


def build_mz_diag(N):
    """Diagonal of the magnetization operator m_z = (1/N) Σ Z_i."""
    configs = all_spin_configs(N)
    return np.mean(configs, axis=1)


def build_nn_zz_diag(N, pbc=True):
    """Diagonal of the averaged nearest-neighbor ZZ correlator."""
    configs = all_spin_configs(N)
    n_bonds = N if pbc else N - 1
    corr = np.zeros(configs.shape[0], dtype=np.float64)
    for i in range(n_bonds):
        j = (i + 1) % N
        corr += configs[:, i] * configs[:, j]
    return corr / max(n_bonds, 1)


def build_nn_xx_matrix(N, pbc=True):
    """Dense matrix of the averaged nearest-neighbor XX correlator."""
    dim = 2**N
    op = np.zeros((dim, dim), dtype=np.complex128)
    n_bonds = N if pbc else N - 1
    for i in range(n_bonds):
        j = (i + 1) % N
        mask = (1 << i) ^ (1 << j)
        for basis in range(dim):
            op[basis ^ mask, basis] += 1.0
    return op / max(n_bonds, 1)


def build_flip_permutation(N):
    """Permutation implementing σ -> -σ in the computational basis."""
    dim = 2**N
    return np.arange(dim - 1, -1, -1, dtype=np.int32)


def project_state_to_parity_np(psi, flip_perm, parity_sign):
    """Project a dense state into the chosen parity sector and normalize it."""
    psi_proj = 0.5 * (psi + parity_sign * psi[flip_perm])
    norm = np.linalg.norm(psi_proj)
    if norm < 1e-14:
        return psi_proj.astype(np.complex64)
    return (psi_proj / norm).astype(np.complex64)


# ═══════════════════════════════════════════════════════════════
# EXACT DYNAMICS REFERENCE
# ═══════════════════════════════════════════════════════════════

def ed_reference(N, h_i, h_f, T, n_eval=100):
    """
    Exact quench dynamics: ground state of H(h_i) evolved under H(h_f).

    Returns dict with: times, psi_t (2^N × n_eval+1), mz_t, energy_t, psi0
    """
    print("  Computing ED reference...")
    H_i = build_tfim_matrix(N, h_i)
    evals, evecs = np.linalg.eigh(H_i)
    psi0 = evecs[:, 0]
    E0_i = evals[0]
    print(f"    E0(h_i={h_i}) = {E0_i:.8f}")

    H_f = build_tfim_matrix(N, h_f)
    mz_diag = build_mz_diag(N)

    dt = T / n_eval
    U_dt = expm(-1j * H_f * dt)

    times = np.linspace(0, T, n_eval + 1)
    psi_t = np.zeros((2**N, n_eval + 1), dtype=np.complex128)
    mz_t = np.zeros(n_eval + 1)
    energy_t = np.zeros(n_eval + 1)

    psi = psi0.copy()
    mz_op = np.diag(mz_diag)
    for k in range(n_eval + 1):
        psi_t[:, k] = psi
        mz_t[k] = np.real(psi.conj() @ mz_op @ psi)
        energy_t[k] = np.real(psi.conj() @ H_f @ psi)
        if k < n_eval:
            psi = U_dt @ psi

    print(f"    m_z(0) = {mz_t[0]:.6f},  m_z(T) = {mz_t[-1]:.6f}")
    print(f"    E(0) = {energy_t[0]:.6f},  drift = {abs(energy_t[-1] - energy_t[0]):.2e}")

    return {
        "times": times,
        "psi_t": psi_t,
        "mz_t": mz_t,
        "energy_t": energy_t,
        "psi0": psi0,
        "E0_initial": E0_i,
    }


# ═══════════════════════════════════════════════════════════════
# t-NQS MODEL
# ═══════════════════════════════════════════════════════════════

if JAX_AVAILABLE:
    class TimeConditionalNQS(nn.Module):
        """Time-conditional neural quantum state with deeper trunk and split heads."""
        hidden_dim: int = 64
        n_freq: int = 8
        n_layers: int = 2

        @nn.compact
        def __call__(self, sigma, t):
            freqs = jnp.arange(1, self.n_freq + 1, dtype=jnp.float32)
            t_features = jnp.concatenate([
                jnp.array([t], dtype=jnp.float32),
                jnp.sin(freqs * t),
                jnp.cos(freqs * t),
            ])

            x = jnp.concatenate([sigma, t_features])
            x = nn.Dense(self.hidden_dim)(x)
            x = nn.gelu(x)

            for _ in range(max(1, self.n_layers) - 1):
                x_in = x
                y = nn.Dense(self.hidden_dim)(x)
                y = nn.gelu(y)
                y = nn.Dense(self.hidden_dim)(y)
                y = nn.gelu(y)
                x = x_in + y

            amp = nn.Dense(max(self.hidden_dim // 2, 16))(x)
            amp = nn.gelu(amp)
            amp = nn.Dense(max(self.hidden_dim // 4, 8))(amp)
            amp = nn.gelu(amp)

            phase = nn.Dense(max(self.hidden_dim // 2, 16))(x)
            phase = nn.gelu(phase)
            phase = nn.Dense(max(self.hidden_dim // 4, 8))(phase)
            phase = nn.gelu(phase)

            log_amp = nn.Dense(1)(amp)[0]
            phase_out = nn.Dense(1)(phase)[0]
            return log_amp, phase_out
else:
    class TimeConditionalNQS:
        pass


# ═══════════════════════════════════════════════════════════════
# LOSS FUNCTION HELPERS
# ═══════════════════════════════════════════════════════════════

def make_loss_fn(
    model,
    H_jax,
    all_sigmas_jax,
    flip_perm_jax,
    parity_sign,
    mu,
    deriv_weight,
    energy_weight,
    anchor_times_jax,
    anchor_states_jax,
    deriv_time_jax,
    deriv_state_jax,
    window_start_jax,
    window_end_jax,
    early_weight_gamma,
    min_residual_weight,
):
    """
    Build a JIT-compiled loss function for one causal time window.

    Supports dynamic per-step scaling of residual / anchor / derivative / energy
    terms so that boundary matching can be emphasized early in each window while
    the TDSE residual ramps in after the handoff has stabilized.
    """
    parity_sign = jnp.float32(parity_sign)
    window_width = jnp.maximum(window_end_jax - window_start_jax, jnp.float32(1e-8))
    early_weight_gamma = jnp.float32(max(0.0, float(early_weight_gamma)))
    min_residual_weight = jnp.float32(min(max(float(min_residual_weight), 0.0), 1.0))
    left_energy = jnp.real(jnp.vdot(deriv_state_jax, H_jax @ deriv_state_jax))

    def eval_raw_state(params, t):
        def single(sigma):
            return model.apply(params, sigma, t)
        log_amps, phases = jax.vmap(single)(all_sigmas_jax)
        return jnp.exp(log_amps + 1j * phases.astype(jnp.complex64))

    def parity_project_state(psi_raw):
        return 0.5 * (psi_raw + parity_sign * psi_raw[flip_perm_jax])

    def unnormalized_state(params, t):
        return parity_project_state(eval_raw_state(params, t))

    def normalized_state(params, t):
        psi = unnormalized_state(params, t)
        norm = jnp.sqrt(jnp.maximum(jnp.sum(jnp.abs(psi) ** 2), 1e-12))
        return psi / norm

    def energy_expectation_from_unnorm(psi):
        H_psi = H_jax @ psi
        norm_sq = jnp.maximum(jnp.sum(jnp.abs(psi) ** 2), 1e-12)
        energy = jnp.real(jnp.vdot(psi, H_psi)) / norm_sq
        return energy, H_psi, norm_sq

    def residual_loss_and_energy_at_time(params, t):
        psi = unnormalized_state(params, t)
        energy, H_psi, norm_sq = energy_expectation_from_unnorm(psi)
        prob = jnp.abs(psi) ** 2 / norm_sq

        safe_psi = jnp.where(jnp.abs(psi) > 1e-12, psi, 1e-12 + 0j)
        E_loc = H_psi / safe_psi

        dpsi_r = jax.jacfwd(lambda t_: jnp.real(unnormalized_state(params, t_)))(t)
        dpsi_i = jax.jacfwd(lambda t_: jnp.imag(unnormalized_state(params, t_)))(t)
        dpsi = dpsi_r + 1j * dpsi_i.astype(jnp.complex64)
        dt_log_psi = dpsi / safe_psi

        r_loc = 1j * dt_log_psi - E_loc
        r_mean = jnp.sum(prob * r_loc)
        r_centered = r_loc - r_mean
        res_loss = jnp.real(jnp.sum(prob * jnp.abs(r_centered) ** 2))
        return res_loss, energy

    def anchor_fidelity_and_loss(params, t_anchor, psi_target):
        psi_pred = normalized_state(params, t_anchor)
        overlap = jnp.vdot(psi_target, psi_pred)
        fidelity = jnp.abs(overlap) ** 2
        return fidelity, -jnp.log(jnp.maximum(fidelity, 1e-10))

    def derivative_constraint_loss(params):
        psi_pred = normalized_state(params, deriv_time_jax)
        overlap = jnp.vdot(deriv_state_jax, psi_pred)
        phase = overlap / jnp.maximum(jnp.abs(overlap), 1e-8)
        psi_pred = psi_pred / phase

        dpsi_r = jax.jacfwd(lambda t_: jnp.real(normalized_state(params, t_)))(deriv_time_jax)
        dpsi_i = jax.jacfwd(lambda t_: jnp.imag(normalized_state(params, t_)))(deriv_time_jax)
        dpsi_pred = (dpsi_r + 1j * dpsi_i.astype(jnp.complex64)) / phase

        dpsi_pred_proj = dpsi_pred - jnp.vdot(psi_pred, dpsi_pred) * psi_pred

        H_psi = H_jax @ deriv_state_jax
        energy = jnp.vdot(deriv_state_jax, H_psi)
        dpsi_exact_proj = -1j * (H_psi - energy * deriv_state_jax)

        diff = dpsi_pred_proj - dpsi_exact_proj
        return jnp.real(jnp.mean(jnp.abs(diff) ** 2))

    def time_weights(t_batch):
        tau = (t_batch - window_start_jax) / window_width
        tau = jnp.clip(tau, 0.0, 1.0)
        weights = min_residual_weight + (1.0 - min_residual_weight) * jnp.exp(-early_weight_gamma * tau)
        return weights / jnp.maximum(jnp.mean(weights), 1e-8)

    @jax.jit
    def total_loss(params, t_batch, loss_scales):
        residual_scale, mu_scale, deriv_scale, energy_scale = loss_scales

        res_losses, energies = jax.vmap(lambda t: residual_loss_and_energy_at_time(params, t))(t_batch)
        weights = time_weights(t_batch)
        res_loss = jnp.sum(weights * res_losses) / jnp.maximum(jnp.sum(weights), 1e-8)
        energy_loss = jnp.sum(weights * (energies - left_energy) ** 2) / jnp.maximum(jnp.sum(weights), 1e-8)

        anchor_fids, anchor_losses = jax.vmap(lambda t_a, psi_a: anchor_fidelity_and_loss(params, t_a, psi_a))(
            anchor_times_jax, anchor_states_jax
        )
        anchor_loss = jnp.mean(anchor_losses)
        mean_anchor_fid = jnp.mean(anchor_fids)

        deriv_loss = derivative_constraint_loss(params)
        total = (
            residual_scale * res_loss
            + (mu * mu_scale) * anchor_loss
            + (deriv_weight * deriv_scale) * deriv_loss
            + (energy_weight * energy_scale) * energy_loss
        )
        return total, (res_loss, anchor_loss, deriv_loss, energy_loss, mean_anchor_fid)

    return total_loss


# ═══════════════════════════════════════════════════════════════
# TRAINING HELPERS
# ═══════════════════════════════════════════════════════════════

def _build_eval_state_fn(model, params, all_sigmas_np, flip_perm_np, parity_sign):
    all_sigmas_jax = jnp.array(all_sigmas_np, dtype=jnp.float32)
    flip_perm_jax = jnp.array(flip_perm_np, dtype=jnp.int32)
    parity_sign = jnp.float32(parity_sign)

    @jax.jit
    def eval_at_time(t):
        def single(sigma):
            return model.apply(params, sigma, t)
        log_amps, phases = jax.vmap(single)(all_sigmas_jax)
        psi_raw = jnp.exp(log_amps + 1j * phases.astype(jnp.complex64))
        psi = 0.5 * (psi_raw + parity_sign * psi_raw[flip_perm_jax])
        norm = jnp.sqrt(jnp.maximum(jnp.sum(jnp.abs(psi) ** 2), 1e-12))
        return psi / norm

    return lambda t: np.array(eval_at_time(jnp.float32(t)))


def _sample_time_batch(rng, n_times, t_left, t_right, t_eps, n_boundary_points):
    """Sample residual times in [t_left, t_right] and force a few near t_left."""
    t_left = float(t_left)
    t_right = float(t_right)
    n_boundary_points = int(max(0, min(n_boundary_points, n_times)))
    n_random = n_times - n_boundary_points

    if n_random > 0:
        if t_right <= t_left:
            t_rand = jnp.full((n_random,), jnp.float32(t_left))
        else:
            t_rand = jax.random.uniform(
                rng, shape=(n_random,),
                minval=jnp.float32(t_left),
                maxval=jnp.float32(t_right),
            )
    else:
        t_rand = jnp.empty((0,), dtype=jnp.float32)

    if n_boundary_points > 0:
        width = max(t_right - t_left, 0.0)
        tiny = max(t_eps, 1e-4 * max(width, 1.0))
        frac = jnp.arange(1, n_boundary_points + 1, dtype=jnp.float32)
        frac = frac / jnp.float32(n_boundary_points + 1)
        offsets = tiny * frac
        t_boundary = jnp.minimum(jnp.float32(t_left) + offsets, jnp.float32(t_right))
        return jnp.concatenate([t_rand, t_boundary], axis=0)

    return t_rand


def _select_anchor_subset(anchor_times, anchor_states, max_time, replay_cutoff, replay_points):
    """Keep all recent anchors plus a replay subset from older windows."""
    eligible = [(t, psi) for t, psi in zip(anchor_times, anchor_states) if t <= max_time + 1e-12]
    recent = [(t, psi) for t, psi in eligible if t >= replay_cutoff - 1e-12]
    older = [(t, psi) for t, psi in eligible if t < replay_cutoff - 1e-12]

    if replay_points > 0 and len(older) > replay_points:
        idx = np.linspace(0, len(older) - 1, replay_points, dtype=int)
        older = [older[i] for i in idx]

    selected = older + recent
    times = np.array([t for t, _ in selected], dtype=np.float32)
    states = np.stack([psi for _, psi in selected], axis=0).astype(np.complex64)
    return times, states, len(eligible)


def _allocate_integer_counts(total, weights, min_count=1):
    """Allocate an integer budget proportionally to positive weights."""
    weights = np.asarray(weights, dtype=np.float64)
    n = len(weights)
    total = int(total)
    if n == 0:
        return []
    if total <= n * min_count:
        return [min_count] * n
    base = np.full(n, min_count, dtype=int)
    remaining = total - n * min_count
    norm = weights / max(np.sum(weights), 1e-12)
    raw = remaining * norm
    extra = np.floor(raw).astype(int)
    base += extra
    leftover = remaining - int(np.sum(extra))
    if leftover > 0:
        frac = raw - extra
        order = np.argsort(-frac)
        for i in order[:leftover]:
            base[i] += 1
    return base.tolist()


def _build_adaptive_window_plan(
    T,
    n_windows,
    n_epochs,
    epochs_per_window=None,
    window_density="gaussian_middle",
    focus_center=None,
    focus_width=None,
    focus_strength=1.0,
    epoch_focus_boost=0.75,
):
    """Build non-uniform causal window edges and per-window epoch budgets."""
    T = float(T)
    n_windows = int(n_windows)
    if focus_center is None:
        focus_center = 0.5 * T
    if focus_width is None or focus_width <= 0:
        focus_width = max(T / 6.0, 0.25)
    focus_center = float(np.clip(focus_center, 0.0, T))
    focus_width = float(max(focus_width, 1e-6))

    if window_density == "uniform" or focus_strength <= 1e-12:
        edges = np.linspace(0.0, T, n_windows + 1)
    else:
        grid_n = max(4001, 250 * n_windows + 1)
        grid = np.linspace(0.0, T, grid_n)
        gauss = np.exp(-0.5 * ((grid - focus_center) / focus_width) ** 2)
        density = 1.0 + float(focus_strength) * gauss
        cdf = np.zeros_like(grid)
        cdf[1:] = np.cumsum(0.5 * (density[:-1] + density[1:]) * np.diff(grid))
        cdf /= max(cdf[-1], 1e-12)
        quantiles = np.linspace(0.0, 1.0, n_windows + 1)
        edges = np.interp(quantiles, cdf, grid)
        edges[0] = 0.0
        edges[-1] = T

    centers = 0.5 * (edges[:-1] + edges[1:])
    epoch_weights = 1.0 + float(epoch_focus_boost) * np.exp(-0.5 * ((centers - focus_center) / focus_width) ** 2)

    if epochs_per_window is None:
        counts = _allocate_integer_counts(int(n_epochs), epoch_weights, min_count=1)
    else:
        base = max(int(epochs_per_window), 1)
        counts = np.maximum(np.rint(base * epoch_weights).astype(int), 1).tolist()

    return edges.astype(np.float64), counts, {
        "focus_center": focus_center,
        "focus_width": focus_width,
        "window_density": window_density,
        "focus_strength": float(focus_strength),
        "epoch_focus_boost": float(epoch_focus_boost),
        "epoch_weights": np.array(epoch_weights, dtype=np.float64),
        "window_widths": np.diff(edges).astype(np.float64),
        "actual_n_epochs": int(np.sum(counts)),
    }


# ═══════════════════════════════════════════════════════════════
# CAUSAL TRAINING LOOP
# ═══════════════════════════════════════════════════════════════

def train_tnqs_causal(
    model,
    H_np,
    psi0_np,
    all_sigmas_np,
    T,
    n_epochs=3000,
    epochs_per_window=None,
    lr=1e-3,
    lr_final=1e-4,
    lr_schedule="cosine",
    warmup_fraction=0.05,
    grad_clip=1.0,
    mu=10.0,
    deriv_weight=5.0,
    energy_weight=5.0,
    n_times=16,
    n_windows=10,
    anchors_per_window=6,
    replay_points=6,
    window_overlap=0.15,
    residual_ramp_fraction=0.25,
    initial_residual_scale=0.25,
    anchor_boost=1.5,
    energy_ramp_start=0.6,
    energy_ramp_fraction=0.2,
    t_eps=1e-6,
    boundary_points=4,
    parity_sign=1.0,
    early_weight_gamma=3.0,
    min_residual_weight=0.25,
    window_density="gaussian_middle",
    focus_center=None,
    focus_width=None,
    focus_strength=1.0,
    epoch_focus_boost=0.75,
    seed=42,
    print_every=100,
    keep_best_per_window=True,
):
    """
    Train the t-NQS ansatz by marching causally over short time windows.

    Additions beyond the hardened baseline:
      - per-window epoch budgeting and LR schedules,
      - gradient clipping,
      - overlap between adjacent residual windows,
      - explicit replay of older anchors,
      - dynamic loss reweighting inside each window,
      - best-of-window parameter rollback.
    """
    H_jax = jnp.array(H_np, dtype=jnp.complex64)
    all_sigmas_jax = jnp.array(all_sigmas_np, dtype=jnp.float32)
    N = all_sigmas_np.shape[1]
    flip_perm_np = build_flip_permutation(N)
    flip_perm_jax = jnp.array(flip_perm_np, dtype=jnp.int32)

    rng = jax.random.PRNGKey(seed)
    dummy_sigma = jnp.ones(N, dtype=jnp.float32)
    params = model.init(rng, dummy_sigma, jnp.float32(0.0))

    n_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
    print(f"  Model parameters: {n_params}")

    optimizer = optax.chain(
        optax.clip_by_global_norm(grad_clip),
        optax.scale_by_adam(),
        optax.scale(-1.0),
    )
    opt_state = optimizer.init(params)

    nominal_edges, epochs_per_window_list, plan_meta = _build_adaptive_window_plan(
        T=T,
        n_windows=n_windows,
        n_epochs=n_epochs,
        epochs_per_window=epochs_per_window,
        window_density=window_density,
        focus_center=focus_center,
        focus_width=focus_width,
        focus_strength=focus_strength,
        epoch_focus_boost=epoch_focus_boost,
    )
    n_epochs = int(plan_meta["actual_n_epochs"])

    anchor_times = [0.0]
    psi0_norm = psi0_np / np.linalg.norm(psi0_np)
    psi0_proj = project_state_to_parity_np(psi0_norm, flip_perm_np, parity_sign)
    anchor_states = [psi0_proj]

    history = {
        "window": [],
        "global_epoch": [],
        "local_epoch": [],
        "window_start": [],
        "window_nominal_start": [],
        "window_end": [],
        "n_anchors": [],
        "lr": [],
        "residual_scale": [],
        "mu_scale": [],
        "deriv_scale": [],
        "energy_scale": [],
        "total_loss": [],
        "res_loss": [],
        "anchor_loss": [],
        "deriv_loss": [],
        "energy_loss": [],
        "anchor_fidelity": [],
        "elapsed_s": [],
    }

    rng_train = jax.random.PRNGKey(seed + 1)
    t0 = time_module.time()
    global_epoch = 0

    def window_lr(local_epoch, total_local_epochs):
        if total_local_epochs <= 1:
            progress = 1.0
        else:
            progress = local_epoch / max(total_local_epochs - 1, 1)
        if lr_schedule == "constant":
            lr_now = lr
        elif lr_schedule == "step":
            if progress < 0.5:
                lr_now = lr
            elif progress < 0.8:
                lr_now = np.sqrt(lr * lr_final)
            else:
                lr_now = lr_final
        else:
            lr_now = lr_final + 0.5 * (lr - lr_final) * (1.0 + np.cos(np.pi * progress))
        if warmup_fraction > 0:
            warmup_steps = max(int(np.ceil(total_local_epochs * warmup_fraction)), 1)
            if local_epoch + 1 < warmup_steps:
                lr_now *= (local_epoch + 1) / warmup_steps
        return float(lr_now)

    print(
        f"  Causal training for {n_epochs} epochs across {n_windows} windows "
        f"(n_times={n_times}, μ={mu}, λ_dt={deriv_weight}, λ_E={energy_weight})"
    )
    print(
        f"  LR schedule={lr_schedule} ({lr} → {lr_final}), grad_clip={grad_clip}, "
        f"overlap={window_overlap}, replay_points={replay_points}"
    )
    print(
        f"  Hard parity projection: {'even' if parity_sign > 0 else 'odd'} sector, "
        f"anchors/window={anchors_per_window}, early-weight γ={early_weight_gamma}"
    )
    print(
        f"  Window plan: {plan_meta['window_density']}  focus=({plan_meta['focus_center']:.3f} ± {plan_meta['focus_width']:.3f}), "
        f"strength={plan_meta['focus_strength']:.2f}, epoch_boost={plan_meta['epoch_focus_boost']:.2f}"
    )
    print()

    for w in range(n_windows):
        nominal_start = float(nominal_edges[w])
        nominal_end = float(nominal_edges[w + 1])
        width = nominal_end - nominal_start
        w_start = max(0.0, nominal_start - window_overlap * width)
        w_end = nominal_end
        n_local_epochs = epochs_per_window_list[w]

        current_times, current_states, eligible_anchor_count = _select_anchor_subset(
            anchor_times=anchor_times,
            anchor_states=anchor_states,
            max_time=nominal_end,
            replay_cutoff=w_start,
            replay_points=replay_points,
        )

        left_idx = max(i for i, t in enumerate(anchor_times) if t <= nominal_start + 1e-12)
        deriv_time = np.float32(anchor_times[left_idx])
        deriv_state = np.array(anchor_states[left_idx], dtype=np.complex64)

        loss_fn = make_loss_fn(
            model=model,
            H_jax=H_jax,
            all_sigmas_jax=all_sigmas_jax,
            flip_perm_jax=flip_perm_jax,
            parity_sign=parity_sign,
            mu=mu,
            deriv_weight=deriv_weight,
            energy_weight=energy_weight,
            anchor_times_jax=jnp.array(current_times, dtype=jnp.float32),
            anchor_states_jax=jnp.array(current_states, dtype=jnp.complex64),
            deriv_time_jax=jnp.array(deriv_time, dtype=jnp.float32),
            deriv_state_jax=jnp.array(deriv_state, dtype=jnp.complex64),
            window_start_jax=jnp.array(w_start, dtype=jnp.float32),
            window_end_jax=jnp.array(w_end, dtype=jnp.float32),
            early_weight_gamma=early_weight_gamma,
            min_residual_weight=min_residual_weight,
        )

        @jax.jit
        def train_step(params, opt_state, t_batch, lr_t, loss_scales):
            (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, t_batch, loss_scales)
            updates, opt_state_new = optimizer.update(grads, opt_state, params)
            updates = jax.tree_util.tree_map(lambda u: lr_t * u, updates)
            params_new = optax.apply_updates(params, updates)
            return params_new, opt_state_new, loss, aux

        print(
            f"  Window {w + 1}/{n_windows}: t ∈ [{w_start:.6f}, {w_end:.6f}] "
            f"(nominal [{nominal_start:.6f}, {nominal_end:.6f}], width={width:.4f})  "
            f"epochs={n_local_epochs}, anchors={len(current_times)} / {eligible_anchor_count}"
        )

        best_params = params
        best_opt_state = opt_state
        best_score = np.inf

        for local_epoch in range(n_local_epochs):
            rng_train, rng_t = jax.random.split(rng_train)
            t_batch = _sample_time_batch(
                rng=rng_t,
                n_times=n_times,
                t_left=w_start,
                t_right=w_end,
                t_eps=t_eps,
                n_boundary_points=boundary_points,
            )

            progress = 1.0 if n_local_epochs <= 1 else local_epoch / max(n_local_epochs - 1, 1)
            lr_t = window_lr(local_epoch, n_local_epochs)
            ramp = min(1.0, progress / max(residual_ramp_fraction, 1e-8))
            residual_scale = initial_residual_scale + (1.0 - initial_residual_scale) * ramp
            mu_scale = 1.0 + anchor_boost * (1.0 - progress)
            deriv_scale = 1.0 + 0.5 * anchor_boost * (1.0 - progress)
            energy_ramp = min(1.0, progress / max(energy_ramp_fraction, 1e-8))
            energy_scale = energy_ramp_start + (1.0 - energy_ramp_start) * energy_ramp
            loss_scales = jnp.array([residual_scale, mu_scale, deriv_scale, energy_scale], dtype=jnp.float32)

            params, opt_state, loss, aux = train_step(params, opt_state, t_batch, jnp.float32(lr_t), loss_scales)
            res_loss, anchor_loss, deriv_loss, energy_loss, mean_anchor_fid = aux
            global_epoch += 1
            elapsed = time_module.time() - t0

            score = float(loss)
            if score < best_score:
                best_score = score
                best_params = jax.tree_util.tree_map(lambda x: x, params)
                best_opt_state = jax.tree_util.tree_map(lambda x: x, opt_state)

            history["window"].append(w + 1)
            history["global_epoch"].append(global_epoch)
            history["local_epoch"].append(local_epoch + 1)
            history["window_start"].append(w_start)
            history["window_nominal_start"].append(nominal_start)
            history["window_end"].append(w_end)
            history["n_anchors"].append(len(current_times))
            history["lr"].append(lr_t)
            history["residual_scale"].append(residual_scale)
            history["mu_scale"].append(mu_scale)
            history["deriv_scale"].append(deriv_scale)
            history["energy_scale"].append(energy_scale)
            history["total_loss"].append(float(loss))
            history["res_loss"].append(float(res_loss))
            history["anchor_loss"].append(float(anchor_loss))
            history["deriv_loss"].append(float(deriv_loss))
            history["energy_loss"].append(float(energy_loss))
            history["anchor_fidelity"].append(float(mean_anchor_fid))
            history["elapsed_s"].append(elapsed)

            if (local_epoch + 1) % print_every == 0 or local_epoch == 0 or local_epoch + 1 == n_local_epochs:
                print(
                    f"    epoch {global_epoch:5d}/{n_epochs}  "
                    f"lr={lr_t:.2e}  "
                    f"loss={float(loss):.4e}  "
                    f"res={float(res_loss):.4e}  "
                    f"anchor={float(anchor_loss):.4e}  "
                    f"dt={float(deriv_loss):.4e}  "
                    f"Ewin={float(energy_loss):.4e}  "
                    f"F(anchor)={float(mean_anchor_fid):.6f}  "
                    f"[{elapsed:.1f}s]"
                )

        if keep_best_per_window:
            params = best_params
            opt_state = best_opt_state

        if w < n_windows - 1 and anchors_per_window > 0:
            eval_state_np = _build_eval_state_fn(model, params, all_sigmas_np, flip_perm_np, parity_sign)
            new_times = np.linspace(nominal_start, nominal_end, anchors_per_window + 1)[1:]
            for t_new in new_times:
                if any(abs(t_new - t_old) < 1e-10 for t_old in anchor_times):
                    continue
                anchor_times.append(float(t_new))
                anchor_states.append(eval_state_np(float(t_new)).astype(np.complex64))

    elapsed = time_module.time() - t0
    print(f"\n  Training complete in {elapsed:.1f}s")
    print(f"  Final: loss={history['total_loss'][-1]:.4e}, F(anchor)={history['anchor_fidelity'][-1]:.6f}")

    return params, history, {
        "anchor_times": np.array(anchor_times, dtype=np.float64),
        "n_windows": n_windows,
        "anchors_per_window": anchors_per_window,
        "parity_sector": "even" if parity_sign > 0 else "odd",
        "lr_schedule": lr_schedule,
        "lr_final": lr_final,
        "window_overlap": window_overlap,
        "replay_points": replay_points,
        "window_density": plan_meta["window_density"],
        "focus_center": plan_meta["focus_center"],
        "focus_width": plan_meta["focus_width"],
        "focus_strength": plan_meta["focus_strength"],
        "epoch_focus_boost": plan_meta["epoch_focus_boost"],
        "window_widths": plan_meta["window_widths"],
        "epochs_per_window_list": np.array(epochs_per_window_list, dtype=np.int32),
        "actual_n_epochs": n_epochs,
    }


# ═══════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════

def evaluate_tnqs(model, params, all_sigmas_np, H_f_np, ed_result, T, parity_sign=1.0, n_eval=100):
    """
    Evaluate the trained t-NQS at a fine grid of times and compare with ED.

    Computes:
      - m_z(t)
      - nearest-neighbor ZZ correlator
      - nearest-neighbor XX correlator
      - fidelity(t)
      - energy(t)
      - parity residual ||ψ - s P ψ||_2
      - the normalized state itself
    """
    all_sigmas_jax = jnp.array(all_sigmas_np, dtype=jnp.float32)
    N = all_sigmas_np.shape[1]
    flip_perm_np = build_flip_permutation(N)
    flip_perm_jax = jnp.array(flip_perm_np, dtype=jnp.int32)
    parity_sign_jax = jnp.float32(parity_sign)
    mz_diag = np.array(build_mz_diag(N), dtype=np.float64)
    zz_diag = np.array(build_nn_zz_diag(N), dtype=np.float64)
    xx_mat = np.array(build_nn_xx_matrix(N), dtype=np.complex128)

    times = np.linspace(0, T, n_eval + 1)

    @jax.jit
    def eval_at_time(t):
        def single(sigma):
            return model.apply(params, sigma, t)
        log_amps, phases = jax.vmap(single)(all_sigmas_jax)
        psi_raw = jnp.exp(log_amps + 1j * phases.astype(jnp.complex64))
        psi = 0.5 * (psi_raw + parity_sign_jax * psi_raw[flip_perm_jax])
        norm = jnp.sqrt(jnp.maximum(jnp.sum(jnp.abs(psi) ** 2), 1e-12))
        return psi / norm

    mz_t = np.zeros(n_eval + 1)
    zz_nn_t = np.zeros(n_eval + 1)
    xx_nn_t = np.zeros(n_eval + 1)
    energy_t = np.zeros(n_eval + 1)
    fidelity_t = np.zeros(n_eval + 1)
    parity_residual_t = np.zeros(n_eval + 1)
    psi_t = np.zeros((all_sigmas_np.shape[0], n_eval + 1), dtype=np.complex128)

    zz_exact_t = np.full(n_eval + 1, np.nan)
    xx_exact_t = np.full(n_eval + 1, np.nan)

    print("  Evaluating t-NQS on time grid...")
    for k, t in enumerate(times):
        psi_nqs = np.array(eval_at_time(jnp.float32(t)), dtype=np.complex128)
        psi_t[:, k] = psi_nqs

        mz_t[k] = np.real(np.sum(np.conj(psi_nqs) * mz_diag * psi_nqs))
        zz_nn_t[k] = np.real(np.sum(np.conj(psi_nqs) * zz_diag * psi_nqs))
        xx_nn_t[k] = np.real(np.vdot(psi_nqs, xx_mat @ psi_nqs))
        H_psi = H_f_np @ psi_nqs
        energy_t[k] = np.real(np.sum(np.conj(psi_nqs) * H_psi))
        parity_residual_t[k] = float(np.linalg.norm(psi_nqs - parity_sign * psi_nqs[flip_perm_np]))

        if ed_result is not None:
            psi_exact = ed_result["psi_t"][:, k]
            overlap = np.sum(np.conj(psi_exact) * psi_nqs)
            fidelity_t[k] = float(np.abs(overlap) ** 2)
            zz_exact_t[k] = np.real(np.sum(np.conj(psi_exact) * zz_diag * psi_exact))
            xx_exact_t[k] = np.real(np.vdot(psi_exact, xx_mat @ psi_exact))

    print(f"    m_z(0) = {mz_t[0]:.6f},  m_z(T) = {mz_t[-1]:.6f}")
    print(f"    Czz(0) = {zz_nn_t[0]:.6f},  Czz(T) = {zz_nn_t[-1]:.6f}")
    print(f"    F(0)   = {fidelity_t[0]:.6f},  F(T) = {fidelity_t[-1]:.6f}")
    print(f"    E(0)   = {energy_t[0]:.6f},  drift = {abs(energy_t[-1] - energy_t[0]):.2e}")
    print(f"    max ||ψ-sPψ||₂ = {np.max(parity_residual_t):.2e}")

    return {
        "times": times,
        "mz_t": mz_t,
        "zz_nn_t": zz_nn_t,
        "xx_nn_t": xx_nn_t,
        "zz_exact_t": zz_exact_t,
        "xx_exact_t": xx_exact_t,
        "energy_t": energy_t,
        "fidelity_t": fidelity_t,
        "parity_residual_t": parity_residual_t,
        "psi_t": psi_t,
    }


# ═══════════════════════════════════════════════════════════════
# OUTPUTS
# ═══════════════════════════════════════════════════════════════

def make_figure(ed_result, tnqs_eval, history, N, h_i, h_f, base_name):
    """Four-panel figure for training + dynamics diagnostics."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    ax = axes[0, 0]
    epochs = np.arange(1, len(history["total_loss"]) + 1)
    ax.semilogy(epochs, history["res_loss"], label="Residual loss", color="#2196F3", lw=1.2, alpha=0.8)
    ax.semilogy(epochs, history["anchor_loss"], label="Anchor loss", color="#E53935", lw=1.2, alpha=0.8)
    ax.semilogy(epochs, history["deriv_loss"], label="Derivative loss", color="#8E24AA", lw=1.1, alpha=0.8)
    ax.semilogy(epochs, history["energy_loss"], label="Energy loss", color="#FB8C00", lw=1.1, alpha=0.85)
    ax.semilogy(epochs, history["total_loss"], label="Total loss", color="k", lw=1.5, alpha=0.6)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Loss", fontsize=11)
    ax.set_title("(a) Causal training convergence", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    if ed_result is not None:
        ax.plot(tnqs_eval["times"], tnqs_eval["zz_exact_t"], "k-", lw=2, alpha=0.7, label="ED exact")
    ax.plot(tnqs_eval["times"], tnqs_eval["zz_nn_t"], "o-", color="#2196F3", ms=2, lw=1.2, label="t-NQS")
    ax.set_xlabel("Time t", fontsize=11)
    ax.set_ylabel("<sigma_i^z sigma_{i+1}^z>", fontsize=11)
    ax.set_title(f"(b) NN ZZ correlator: h={h_i}→{h_f}, N={N}", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(tnqs_eval["times"], tnqs_eval["fidelity_t"], "o-", color="#4CAF50", ms=2, lw=1.2)
    ax.axhline(1.0, color="k", ls="--", lw=0.8, alpha=0.3)
    ax.set_xlabel("Time t", fontsize=11)
    ax.set_ylabel("Fidelity |<psi_exact|psi_t-NQS>|^2", fontsize=11)
    ax.set_title("(c) State fidelity", fontsize=12)
    ax.set_ylim(bottom=max(0, min(tnqs_eval["fidelity_t"]) - 0.05), top=1.02)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    if ed_result is not None:
        ax.plot(ed_result["times"], ed_result["energy_t"], "k-", lw=2, alpha=0.7, label="ED exact")
    ax.plot(tnqs_eval["times"], tnqs_eval["energy_t"], "o-", color="#E53935", ms=2, lw=1.2, label="t-NQS")
    ax.set_xlabel("Time t", fontsize=11)
    ax.set_ylabel("<H_f>_t", fontsize=11)
    ax.set_title("(d) Energy conservation", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.suptitle(
        f"t-NQS causal interval training: 1D TFIM N={N}, quench h={h_i}→{h_f}",
        fontsize=14, y=1.01
    )
    plt.tight_layout()
    plt.savefig(base_name + ".png", dpi=150, bbox_inches="tight")
    plt.savefig(base_name + ".pdf", bbox_inches="tight")
    print(f"Saved: {base_name}.png / .pdf")


def save_csv_logs(base_name, args, history, ed_result, tnqs_eval, summary, training_meta):
    """Write training, evaluation, and summary CSV logs."""
    train_path = base_name + "_training.csv"
    with open(train_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "window", "global_epoch", "local_epoch",
            "window_start", "window_nominal_start", "window_end", "n_anchors",
            "lr", "residual_scale", "mu_scale", "deriv_scale", "energy_scale",
            "total_loss", "res_loss", "anchor_loss", "deriv_loss", "energy_loss",
            "anchor_fidelity", "elapsed_s"
        ])
        n_rows = len(history["global_epoch"])
        for i in range(n_rows):
            writer.writerow([
                history["window"][i],
                history["global_epoch"][i],
                history["local_epoch"][i],
                history["window_start"][i],
                history["window_nominal_start"][i],
                history["window_end"][i],
                history["n_anchors"][i],
                history["lr"][i],
                history["residual_scale"][i],
                history["mu_scale"][i],
                history["deriv_scale"][i],
                history["energy_scale"][i],
                history["total_loss"][i],
                history["res_loss"][i],
                history["anchor_loss"][i],
                history["deriv_loss"][i],
                history["energy_loss"][i],
                history["anchor_fidelity"][i],
                history["elapsed_s"][i],
            ])

    results_path = base_name + "_results.csv"
    with open(results_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "time",
            "mz_exact", "mz_tnqs", "abs_mz_error",
            "zz_nn_exact", "zz_nn_tnqs", "abs_zz_nn_error",
            "xx_nn_exact", "xx_nn_tnqs", "abs_xx_nn_error",
            "fidelity",
            "energy_exact", "energy_tnqs", "energy_error",
            "parity_l2_residual",
        ])
        for i, t in enumerate(tnqs_eval["times"]):
            mz_exact = float(ed_result["mz_t"][i]) if ed_result is not None else np.nan
            energy_exact = float(ed_result["energy_t"][i]) if ed_result is not None else np.nan
            zz_exact = float(tnqs_eval["zz_exact_t"][i])
            xx_exact = float(tnqs_eval["xx_exact_t"][i])
            mz_tnqs = float(tnqs_eval["mz_t"][i])
            zz_tnqs = float(tnqs_eval["zz_nn_t"][i])
            xx_tnqs = float(tnqs_eval["xx_nn_t"][i])
            energy_tnqs = float(tnqs_eval["energy_t"][i])
            writer.writerow([
                float(t),
                mz_exact,
                mz_tnqs,
                abs(mz_tnqs - mz_exact) if ed_result is not None else np.nan,
                zz_exact,
                zz_tnqs,
                abs(zz_tnqs - zz_exact) if not np.isnan(zz_exact) else np.nan,
                xx_exact,
                xx_tnqs,
                abs(xx_tnqs - xx_exact) if not np.isnan(xx_exact) else np.nan,
                float(tnqs_eval["fidelity_t"][i]),
                energy_exact,
                energy_tnqs,
                abs(energy_tnqs - energy_exact) if ed_result is not None else np.nan,
                float(tnqs_eval["parity_residual_t"][i]),
            ])

    summary_path = base_name + "_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "N", "hi", "hf", "T",
            "n_epochs", "epochs_per_window", "lr", "lr_final", "lr_schedule", "warmup_fraction", "grad_clip",
            "mu", "deriv_weight", "energy_weight",
            "hidden", "n_freq", "n_layers", "n_times", "n_windows",
            "anchors_per_window", "replay_points", "window_overlap", "boundary_points", "seed",
            "parity_sector", "early_weight_gamma", "min_residual_weight",
            "residual_ramp_fraction", "initial_residual_scale", "anchor_boost",
            "energy_ramp_start", "energy_ramp_fraction",
            "window_density", "focus_center", "focus_width", "focus_strength", "epoch_focus_boost",
            "actual_n_epochs", "n_anchor_total",
            "mean_fidelity", "min_fidelity",
            "max_abs_mz_error", "mean_abs_mz_error",
            "zz_nn_mae", "xx_nn_mae",
            "energy_drift", "energy_mae",
            "max_parity_l2_residual",
            "final_total_loss", "final_res_loss",
            "final_anchor_loss", "final_deriv_loss", "final_energy_loss",
            "final_anchor_fidelity",
        ])
        writer.writerow([
            args.N, args.hi, args.hf, args.T,
            args.n_epochs, args.epochs_per_window, args.lr, args.lr_final, args.lr_schedule, args.warmup_fraction, args.grad_clip,
            args.mu, args.deriv_weight, args.energy_weight,
            args.hidden, args.n_freq, args.n_layers, args.n_times, args.n_windows,
            args.anchors_per_window, args.replay_points, args.window_overlap, args.boundary_points, args.seed,
            training_meta["parity_sector"], args.early_weight_gamma, args.min_residual_weight,
            args.residual_ramp_fraction, args.initial_residual_scale, args.anchor_boost,
            args.energy_ramp_start, args.energy_ramp_fraction,
            training_meta.get("window_density", getattr(args, "window_density", "uniform")),
            training_meta.get("focus_center", getattr(args, "focus_center", np.nan)),
            training_meta.get("focus_width", getattr(args, "focus_width", np.nan)),
            training_meta.get("focus_strength", getattr(args, "focus_strength", 0.0)),
            training_meta.get("epoch_focus_boost", getattr(args, "epoch_focus_boost", 0.0)),
            training_meta.get("actual_n_epochs", args.n_epochs), len(training_meta["anchor_times"]),
            summary["mean_fidelity"],
            summary["min_fidelity"],
            summary["max_abs_mz_error"],
            summary["mean_abs_mz_error"],
            summary["zz_nn_mae"],
            summary["xx_nn_mae"],
            summary["energy_drift"],
            summary["energy_mae"],
            summary["max_parity_l2_residual"],
            history["total_loss"][-1],
            history["res_loss"][-1],
            history["anchor_loss"][-1],
            history["deriv_loss"][-1],
            history["energy_loss"][-1],
            history["anchor_fidelity"][-1],
        ])

    print(f"Saved: {train_path}")
    print(f"Saved: {results_path}")
    print(f"Saved: {summary_path}")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="t-NQS causal interval training for TFIM quench dynamics (adaptive middle-window focus)"
    )
    parser.add_argument("--N", type=int, default=8,
                        help="Chain length (default: 8; max ~12 for exact)")
    parser.add_argument("--hi", type=float, default=0.5,
                        help="Initial transverse field (ordered phase)")
    parser.add_argument("--hf", type=float, default=2.0,
                        help="Final transverse field (disordered phase)")
    parser.add_argument("--T", type=float, default=3.0,
                        help="Total evolution time")
    parser.add_argument("--n-epochs", type=int, default=3000,
                        help="Total training epochs across all windows (ignored if --epochs-per-window is set)")
    parser.add_argument("--epochs-per-window", type=int, default=None,
                        help="Explicit epochs per causal window")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Initial learning rate")
    parser.add_argument("--lr-final", type=float, default=1e-4,
                        help="Final per-window learning rate for scheduled decay")
    parser.add_argument("--lr-schedule", type=str, choices=["constant", "cosine", "step"], default="cosine",
                        help="Per-window learning-rate schedule")
    parser.add_argument("--warmup-fraction", type=float, default=0.05,
                        help="Fraction of each window used for LR warmup")
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="Global gradient clipping norm")
    parser.add_argument("--mu", type=float, default=10.0,
                        help="Anchor fidelity penalty weight")
    parser.add_argument("--deriv-weight", type=float, default=5.0,
                        help="Short-time derivative constraint weight")
    parser.add_argument("--energy-weight", type=float, default=5.0,
                        help="In-window energy conservation penalty weight")
    parser.add_argument("--hidden", type=int, default=128,
                        help="Hidden layer width")
    parser.add_argument("--n-freq", type=int, default=16,
                        help="Fourier embedding frequencies")
    parser.add_argument("--n-layers", type=int, default=2,
                        help="Number of shared residual trunk layers")
    parser.add_argument("--n-times", type=int, default=24,
                        help="Residual time points per training step")
    parser.add_argument("--n-windows", type=int, default=12,
                        help="Number of causal windows")
    parser.add_argument("--anchors-per-window", type=int, default=6,
                        help="New anchors to add after each window")
    parser.add_argument("--replay-points", type=int, default=6,
                        help="Older anchors to replay from earlier windows")
    parser.add_argument("--window-overlap", type=float, default=0.15,
                        help="Fractional overlap of each residual window into the previous one")
    parser.add_argument("--boundary-points", type=int, default=4,
                        help="Forced residual samples near each window start")
    parser.add_argument("--t-eps", type=float, default=1e-6,
                        help="Minimum near-boundary offset for forced samples")
    parser.add_argument("--parity-sector", type=str, choices=["even", "odd"], default="even",
                        help="Hard parity sector for global spin-flip symmetry")
    parser.add_argument("--early-weight-gamma", type=float, default=3.0,
                        help="Exponential emphasis on early-window residuals")
    parser.add_argument("--min-residual-weight", type=float, default=0.30,
                        help="Floor for late-window residual weight in [0,1]")
    parser.add_argument("--residual-ramp-fraction", type=float, default=0.25,
                        help="Fraction of each window over which residual weight ramps up")
    parser.add_argument("--initial-residual-scale", type=float, default=0.25,
                        help="Initial scale factor applied to residual loss at each new window")
    parser.add_argument("--anchor-boost", type=float, default=1.5,
                        help="Extra early-window emphasis on anchor and derivative losses")
    parser.add_argument("--energy-ramp-start", type=float, default=0.6,
                        help="Initial scale factor applied to the energy penalty at each new window")
    parser.add_argument("--energy-ramp-fraction", type=float, default=0.2,
                        help="Fraction of each window over which the energy weight ramps to full strength")
    parser.add_argument("--window-density", type=str, choices=["uniform", "gaussian_middle"], default="gaussian_middle",
                        help="How to place causal window edges; gaussian_middle makes shorter windows near the hard middle-time region")
    parser.add_argument("--focus-center", type=float, default=None,
                        help="Center time for adaptive windows/epoch concentration (default: T/2)")
    parser.add_argument("--focus-width", type=float, default=None,
                        help="Width of the adaptive middle-time focus region (default: max(T/6, 0.25))")
    parser.add_argument("--focus-strength", type=float, default=1.0,
                        help="How strongly to compress windows near the focus region")
    parser.add_argument("--epoch-focus-boost", type=float, default=0.75,
                        help="Extra epoch concentration in the focus region (0 disables adaptive per-window budgets)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not JAX_AVAILABLE:
        print(f"ERROR: JAX/Flax not available ({_IMPORT_ERROR})")
        print("Install: pip install jax flax optax --break-system-packages")
        return

    N = args.N
    if N > 12:
        print(f"WARNING: N={N} gives 2^{N} = {2**N:,} configurations.")
        print("  Exact enumeration will be very slow. For N > 12, an MCMC")
        print("  estimator should replace full enumeration.")
        print()

    if args.epochs_per_window is not None:
        args.n_epochs = args.epochs_per_window * args.n_windows

    parity_sign = 1.0 if args.parity_sector == "even" else -1.0

    print(f"t-NQS Causal Interval Training (Adaptive): N={N}, h = {args.hi} → {args.hf}")
    print(f"Time interval [0, {args.T}], {args.n_epochs} epochs, {args.n_windows} windows")
    print(f"Hilbert space: 2^{N} = {2**N} configurations (exact enumeration)")
    print(
        f"Parity sector: {args.parity_sector}, anchors/window={args.anchors_per_window}, "
        f"replay={args.replay_points}, overlap={args.window_overlap}"
    )
    print(
        f"Schedule: {args.lr_schedule} ({args.lr} → {args.lr_final}), hidden={args.hidden}, "
        f"layers={args.n_layers}, n_freq={args.n_freq}"
    )
    focus_center_disp = 0.5 * args.T if args.focus_center is None else args.focus_center
    focus_width_disp = max(args.T / 6.0, 0.25) if args.focus_width is None else args.focus_width
    print(
        f"Adaptive windows: {args.window_density}, focus=({focus_center_disp:.3f} ± {focus_width_disp:.3f}), "
        f"strength={args.focus_strength}, epoch_boost={args.epoch_focus_boost}"
    )
    print()

    all_sigmas = all_spin_configs(N)
    H_f = build_tfim_matrix(N, args.hf)

    print("=" * 55)
    print("PHASE 1: Exact diagonalization reference")
    print("=" * 55)
    ed = ed_reference(N, args.hi, args.hf, args.T, n_eval=100)
    print()

    print("=" * 55)
    print("PHASE 2: t-NQS causal training")
    print("=" * 55)
    model = TimeConditionalNQS(hidden_dim=args.hidden, n_freq=args.n_freq, n_layers=args.n_layers)

    params, history, training_meta = train_tnqs_causal(
        model=model,
        H_np=H_f,
        psi0_np=ed["psi0"],
        all_sigmas_np=all_sigmas,
        T=args.T,
        n_epochs=args.n_epochs,
        epochs_per_window=args.epochs_per_window,
        lr=args.lr,
        lr_final=args.lr_final,
        lr_schedule=args.lr_schedule,
        warmup_fraction=args.warmup_fraction,
        grad_clip=args.grad_clip,
        mu=args.mu,
        deriv_weight=args.deriv_weight,
        energy_weight=args.energy_weight,
        n_times=args.n_times,
        n_windows=args.n_windows,
        anchors_per_window=args.anchors_per_window,
        replay_points=args.replay_points,
        window_overlap=args.window_overlap,
        residual_ramp_fraction=args.residual_ramp_fraction,
        initial_residual_scale=args.initial_residual_scale,
        anchor_boost=args.anchor_boost,
        energy_ramp_start=args.energy_ramp_start,
        energy_ramp_fraction=args.energy_ramp_fraction,
        t_eps=args.t_eps,
        boundary_points=args.boundary_points,
        parity_sign=parity_sign,
        early_weight_gamma=args.early_weight_gamma,
        min_residual_weight=args.min_residual_weight,
        window_density=args.window_density,
        focus_center=args.focus_center,
        focus_width=args.focus_width,
        focus_strength=args.focus_strength,
        epoch_focus_boost=args.epoch_focus_boost,
        seed=args.seed,
    )
    print()

    print("=" * 55)
    print("PHASE 3: Evaluation")
    print("=" * 55)
    tnqs_eval = evaluate_tnqs(
        model, params, all_sigmas, H_f, ed, args.T,
        parity_sign=parity_sign, n_eval=100
    )
    print()

    print("=" * 55)
    print("SUMMARY")
    print("=" * 55)
    mean_fid = float(np.mean(tnqs_eval["fidelity_t"]))
    min_fid = float(np.min(tnqs_eval["fidelity_t"]))
    mz_diff = np.abs(ed["mz_t"] - tnqs_eval["mz_t"])
    zz_diff = np.abs(tnqs_eval["zz_exact_t"] - tnqs_eval["zz_nn_t"])
    xx_diff = np.abs(tnqs_eval["xx_exact_t"] - tnqs_eval["xx_nn_t"])
    energy_diff = np.abs(ed["energy_t"] - tnqs_eval["energy_t"])
    e_drift = float(abs(tnqs_eval["energy_t"][-1] - tnqs_eval["energy_t"][0]))
    max_parity_residual = float(np.max(tnqs_eval["parity_residual_t"]))

    print(f"  Mean fidelity:       {mean_fid:.6f}")
    print(f"  Min fidelity:        {min_fid:.6f}")
    print(f"  Max |Δm_z|:          {np.max(mz_diff):.4e}")
    print(f"  Mean |Δm_z|:         {np.mean(mz_diff):.4e}")
    print(f"  ZZ correlator MAE:   {np.mean(zz_diff):.4e}")
    print(f"  XX correlator MAE:   {np.mean(xx_diff):.4e}")
    print(f"  Energy drift:        {e_drift:.4e}")
    print(f"  Energy MAE:          {np.mean(energy_diff):.4e}")
    print(f"  Max ||ψ-sPψ||₂:      {max_parity_residual:.4e}")
    print()

    summary = {
        "mean_fidelity": mean_fid,
        "min_fidelity": min_fid,
        "max_abs_mz_error": float(np.max(mz_diff)),
        "mean_abs_mz_error": float(np.mean(mz_diff)),
        "zz_nn_mae": float(np.mean(zz_diff)),
        "xx_nn_mae": float(np.mean(xx_diff)),
        "energy_drift": e_drift,
        "energy_mae": float(np.mean(energy_diff)),
        "max_parity_l2_residual": max_parity_residual,
    }

    base_name = f"tnqs_causal_symplus_adapt_N{N}_h{args.hi}-{args.hf}"
    base_name = str(Path.cwd() / base_name)

    save_csv_logs(base_name, args, history, ed, tnqs_eval, summary, training_meta)
    make_figure(ed, tnqs_eval, history, N, args.hi, args.hf, base_name)


if __name__ == "__main__":
    main()
