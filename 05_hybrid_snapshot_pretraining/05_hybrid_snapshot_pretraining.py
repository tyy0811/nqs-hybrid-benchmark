"""
05: Hybrid Snapshot Pretraining — Synthetic Data Pipeline
=========================================================
Demonstrates the two-phase hybrid workflow of Lange, Bornet, Browaeys,
Bohrdt et al. (Quantum 9, 1675, 2025) on the 1D TFIM, following §6 of
NQS_Theoretical_Framework.md.

The core idea (§6.3):
  Phase 1 — Data-driven pretraining:
    Generate synthetic "measurement snapshots" from the ground state's
    Born distribution p*(σ) = |⟨σ|ψ*⟩|².  Train the NQS to maximizes
    the snapshot likelihood (minimize NLL).  This places parameters
    near the correct basin.

  Phase 2 — Hamiltonian-driven refinement:
    Starting from the pretrained parameters, run standard VMC + SR.
    The energy functional corrects any residual errors from finite
    snapshot statistics.

Key demonstration: pretrained NQS converges faster and to lower
energy than a cold-start NQS, especially at larger system sizes
where VMC alone can get trapped.

Additional features:
  - Noise robustness test (§6.4): inject per-site bit-flip noise
    into the snapshots and show that the hybrid workflow remains
    effective because Phase 2 corrects noise-induced biases.
  - Comparison with cold-start VMC (same architecture, random init).
  - Works on the TFIM, which is stoquastic (§6.1), so Z-basis
    snapshots alone are sufficient (no sign problem for pretraining).

Sign convention:
  Our convention: H = −J Σ Z_iZ_j − h Σ X_i (ferro for J > 0)
  NetKet Ising:   H = +J Σ Z_iZ_j − h Σ X_i
  ⇒ Use J = −1.0 in NetKet.

Requirements:
    netket >= 3.0, jax, optax, numpy, matplotlib
    Python >= 3.11

Usage:
    python hybrid_snapshot_pretraining.py

    # Quick test (fewer snapshots + iterations):
    python hybrid_snapshot_pretraining.py --N 8 --n-snapshots 500 --n-pretrain 200

    # With noise robustness test:
    python hybrid_snapshot_pretraining.py --noise-test

    # Custom:
    python hybrid_snapshot_pretraining.py --N 14 --h 1.0 --n-snapshots 5000
"""

import argparse
import csv
import os
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def first_hit_iteration(trace, E_exact, abs_err):
    """Return the first 1-based iteration where |E - E_exact| <= abs_err."""
    if trace is None or len(trace) == 0 or E_exact is None or abs_err is None:
        return float("nan")
    trace = np.asarray(trace, dtype=float)
    hits = np.where(np.abs(trace - E_exact) <= abs_err)[0]
    if len(hits) == 0:
        return float("nan")
    return float(hits[0] + 1)


def _extract_energy_trace_from_runtime_log(log):
    """Robustly extract the VMC energy trace from a NetKet RuntimeLog."""
    energy_data = getattr(log, "data", {}).get("Energy", {})
    if isinstance(energy_data, dict) and "Mean" in energy_data:
        raw_trace = energy_data["Mean"]
    else:
        raw_trace = log["Energy"]["Mean"]
    return np.real(np.asarray(raw_trace, dtype=float)).reshape(-1)


def run_vmc_driver(
    driver,
    n_iter,
    E_exact=None,
    target_abs_err=None,
    early_stop=False,
    check_every=10,
    min_stop_iter=0,
    stop_patience=1,
):
    """Run a NetKet VMC driver, optionally in chunks with early stopping."""
    import netket as nk

    use_early_stop = bool(
        early_stop
        and E_exact is not None
        and target_abs_err is not None
        and n_iter > 0
    )

    if not use_early_stop:
        log = nk.logging.RuntimeLog()
        t0 = time.time()
        _run_driver_with_warning_filters(driver, n_iter=n_iter, out=log)
        elapsed = time.time() - t0
        energy_trace = _extract_energy_trace_from_runtime_log(log)
        completed_iters = int(len(energy_trace))
        if completed_iters == 0 and n_iter > 0:
            completed_iters = int(n_iter)
        return {
            "energy_trace": energy_trace,
            "elapsed_s": elapsed,
            "completed_iters": completed_iters,
            "requested_iters": int(n_iter),
            "stopped_early": False,
            "check_every": int(max(1, check_every)),
            "stop_patience": int(max(1, stop_patience)),
        }

    check_every = int(max(1, check_every))
    min_stop_iter = int(max(0, min_stop_iter))
    stop_patience = int(max(1, stop_patience))
    trace_parts = []
    completed_iters = 0
    stopped_early = False
    consecutive_hits = 0

    t0 = time.time()
    while completed_iters < n_iter:
        chunk = min(check_every, n_iter - completed_iters)
        log = nk.logging.RuntimeLog()
        _run_driver_with_warning_filters(driver, n_iter=chunk, out=log)

        chunk_trace = _extract_energy_trace_from_runtime_log(log)
        if len(chunk_trace) > chunk:
            chunk_trace = chunk_trace[-chunk:]
        if len(chunk_trace) > 0:
            trace_parts.append(chunk_trace)
            completed_iters += int(len(chunk_trace))
        else:
            completed_iters += int(chunk)

        if trace_parts and completed_iters >= min_stop_iter:
            latest_e = float(trace_parts[-1][-1])
            if np.isfinite(latest_e) and abs(latest_e - E_exact) <= target_abs_err:
                consecutive_hits += 1
                if consecutive_hits >= stop_patience:
                    stopped_early = True
                    break
            else:
                consecutive_hits = 0

    elapsed = time.time() - t0
    energy_trace = (
        np.concatenate(trace_parts).astype(float, copy=False)
        if trace_parts else np.asarray([], dtype=float)
    )

    if len(energy_trace) > n_iter:
        energy_trace = energy_trace[:n_iter]
        completed_iters = int(len(energy_trace))

    return {
        "energy_trace": energy_trace,
        "elapsed_s": elapsed,
        "completed_iters": int(completed_iters),
        "requested_iters": int(n_iter),
        "stopped_early": bool(stopped_early),
        "check_every": check_every,
        "stop_patience": stop_patience,
    }


def compare_final_abs_errors(E_exact, hybrid_result, cold_result, tol=1e-12):
    """Compare final absolute errors and identify the winner."""
    if E_exact is None or hybrid_result is None or cold_result is None:
        return None

    hybrid_abs_err = abs(float(hybrid_result["energy"]) - float(E_exact))
    cold_abs_err = abs(float(cold_result["energy"]) - float(E_exact))
    gap_cold_minus_hybrid = cold_abs_err - hybrid_abs_err

    if gap_cold_minus_hybrid > tol:
        winner = "hybrid"
    elif gap_cold_minus_hybrid < -tol:
        winner = "cold_start"
    else:
        winner = "tie"

    return {
        "hybrid_abs_err": float(hybrid_abs_err),
        "cold_abs_err": float(cold_abs_err),
        "gap_cold_minus_hybrid": float(gap_cold_minus_hybrid),
        "winner": winner,
    }


def _winner_label(winner):
    if winner == "hybrid":
        return "hybrid closer to exact"
    if winner == "cold_start":
        return "cold-start closer to exact"
    return "tie in final absolute error"


def ed_reference(H_dense):
    """Small-system ED helper with local suppression of expected scipy eigsh fallback warnings."""
    H_arr = np.asarray(H_dense, dtype=float)
    try:
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla

        H_sp = sp.csr_matrix(H_arr)
        with warnings.catch_warnings():
            for cat in (RuntimeWarning, UserWarning):
                warnings.filterwarnings(
                    "ignore",
                    message=r".*k >= N.*eigsh.*",
                    category=cat,
                )
                warnings.filterwarnings(
                    "ignore",
                    message=r".*k >= N.*square matrix.*",
                    category=cat,
                )
            evals, evecs = spla.eigsh(H_sp, k=1, which="SA")
        idx = int(np.argmin(np.real(evals)))
        return float(np.real(evals[idx])), np.asarray(evecs[:, idx], dtype=float)
    except Exception:
        evals, evecs = np.linalg.eigh(H_arr)
        return float(np.real(evals[0])), np.asarray(evecs[:, 0], dtype=float)


def _run_driver_with_warning_filters(driver, n_iter, out):
    """Run NetKet VMC with robust local filters for known non-fatal holomorphic warnings."""
    with warnings.catch_warnings():
        for cat in (UserWarning, RuntimeWarning, FutureWarning):
            warnings.filterwarnings("ignore", message=r".*holomorphic.*", category=cat)
            warnings.filterwarnings("ignore", message=r".*not holomorphic.*", category=cat)
            warnings.filterwarnings("ignore", message=r".*complex-valued parameters.*", category=cat)
        driver.run(n_iter=n_iter, out=out)


def _make_sr_preconditioner(nk, diag_shift=0.01):
    """Construct SR with holomorphic=False when supported, with backward-compatible fallback."""
    try:
        return nk.optimizer.SR(diag_shift=diag_shift, holomorphic=False)
    except TypeError:
        return nk.optimizer.SR(diag_shift=diag_shift)


def _make_vmc_driver(nk, optax, H, vstate, lr, diag_shift=0.01):
    """Construct a VMC+SR driver for the current variational state."""
    opt = optax.sgd(learning_rate=lr)
    sr = _make_sr_preconditioner(nk, diag_shift=diag_shift)
    return nk.driver.VMC(H, opt, variational_state=vstate, preconditioner=sr)


def safe_grid_graph_2d(nk, Lx, Ly, pbc=True):
    """Cross-version-safe 2D square-lattice helper based on nk.graph.Grid."""
    pbc_vec = [bool(pbc), bool(pbc)]
    try:
        return nk.graph.Grid(extent=[int(Lx), int(Ly)], pbc=pbc_vec)
    except TypeError:
        return nk.graph.Grid(length=[int(Lx), int(Ly)], pbc=pbc_vec)


def run_2d(Lx, Ly, pbc=True):
    """Future-proof 2D graph scaffold using nk.graph.Grid instead of nk.graph.Square."""
    import netket as nk
    return safe_grid_graph_2d(nk, Lx=Lx, Ly=Ly, pbc=pbc)


# ═══════════════════════════════════════════════════════════════
# SNAPSHOT GENERATION  (§6.1: Born sampling from target state)
# ═══════════════════════════════════════════════════════════════

def generate_snapshots_ed(N, h, n_snapshots, J=1.0, pbc=True, seed=42):
    """
    Generate synthetic measurement snapshots from the ground state of
    the 1D TFIM via exact diagonalization.

    Each snapshot is a spin configuration σ ∈ {+1,−1}^N sampled from
    the Born distribution p*(σ) = |⟨σ|ψ_0⟩|² (§6.1).

    This simulates what a quantum gas microscope or Rydberg-atom array
    would produce: projective measurements in the computational (Z) basis.

    Returns:
        snapshots: array of shape (n_snapshots, N), values in {+1, −1}
        psi0: ground-state vector (2^N,)
        E0: ground-state energy
    """
    # Build TFIM: H = −J Σ ZZ − h Σ X
    dim = 2**N
    I2 = np.eye(2, dtype=complex)
    SX = np.array([[0, 1], [1, 0]], dtype=complex)
    SZ = np.array([[1, 0], [0, -1]], dtype=complex)

    def kron_chain(ops):
        result = ops[0]
        for op in ops[1:]:
            result = np.kron(result, op)
        return result

    H = np.zeros((dim, dim), dtype=complex)
    n_bonds = N if pbc else N - 1
    for i in range(n_bonds):
        j = (i + 1) % N
        ops = [I2] * N; ops[i] = SZ; ops[j] = SZ
        H -= J * kron_chain(ops)
    for i in range(N):
        ops = [I2] * N; ops[i] = SX
        H -= h * kron_chain(ops)

    E0, psi0 = ed_reference(H.real)
    psi0 = np.asarray(psi0, dtype=float).reshape(-1)

    # Born sampling: sample basis-state indices from p(σ) = |ψ_0(σ)|²
    probs = np.abs(psi0)**2
    rng = np.random.default_rng(seed)
    indices = rng.choice(dim, size=n_snapshots, p=probs)

    # Convert basis-state index → spin configuration {+1,−1}^N
    snapshots = np.zeros((n_snapshots, N), dtype=np.float64)
    for idx in range(n_snapshots):
        basis_idx = indices[idx]
        for bit in range(N):
            snapshots[idx, bit] = 1.0 - 2.0 * ((basis_idx >> bit) & 1)

    return snapshots, psi0, E0


def add_readout_noise(snapshots, epsilon, seed=123):
    """
    Apply per-site bit-flip noise to snapshots (§6.4).

    Each spin σ_i is independently flipped with probability ε:
      σ_i → −σ_i  with prob ε,   σ_i  with prob 1−ε.

    This models the dominant error source in quantum gas microscope
    experiments (readout infidelity).
    """
    rng = np.random.default_rng(seed)
    flip_mask = rng.random(snapshots.shape) < epsilon
    noisy = snapshots.copy()
    noisy[flip_mask] *= -1
    return noisy


# ═══════════════════════════════════════════════════════════════
# PHASE 1: NLL PRETRAINING  (§6.1)
# ═══════════════════════════════════════════════════════════════

def pretrain_nll(
    N,
    h,
    snapshots,
    alpha=2,
    n_iter=300,
    n_samples=2048,
    lr=0.01,
    diag_shift=0.01,
    batch_size=512,
    seed=0,
    holdout_frac=0.1,
    eval_every=10,
    lr_final=None,
    verbose=True,
):
    """
    NLL pretraining with optional learning-rate decay and held-out NLL tracking.

    The held-out metric is an exact NLL for N <= 16, computed by enumerating the
    full computational basis of the RBM and normalizing |psi|^2 exactly.
    """
    import netket as nk
    import jax
    import jax.numpy as jnp
    import jax.scipy as jsp
    import optax as ox

    g = nk.graph.Chain(length=N, pbc=True)
    hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes)
    H = nk.operator.Ising(hilbert=hi, graph=g, h=h, J=-1.0)

    model = nk.models.RBM(alpha=alpha, param_dtype=float)
    sampler = nk.sampler.MetropolisLocal(hi, n_chains=16)
    vstate = nk.vqs.MCState(sampler, model, n_samples=n_samples)

    snap_np = np.asarray(snapshots, dtype=np.float32)
    n_snap = int(snap_np.shape[0])
    rng = np.random.default_rng(seed)

    n_holdout = int(max(0, min(n_snap - 1, round(holdout_frac * n_snap)))) if n_snap > 1 else 0
    if n_holdout > 0:
        perm = rng.permutation(n_snap)
        holdout_np = snap_np[perm[:n_holdout]]
        train_np = snap_np[perm[n_holdout:]]
    else:
        holdout_np = np.empty((0, N), dtype=np.float32)
        train_np = snap_np

    snap_jax = jnp.array(train_np, dtype=jnp.float32)
    holdout_jax = jnp.array(holdout_np, dtype=jnp.float32)
    n_train = int(snap_jax.shape[0])
    batch_size = int(max(1, min(batch_size, max(1, n_train))))
    eval_every = int(max(1, eval_every))

    if lr_final is None:
        lr_final = lr
    scheduler = ox.linear_schedule(
        init_value=float(lr),
        end_value=float(lr_final),
        transition_steps=max(1, int(n_iter) - 1),
    )
    optimizer = ox.adam(scheduler)
    opt_state = optimizer.init(vstate.parameters)

    @jax.jit
    def log_amp_at(params, sigma):
        return model.apply({"params": params}, sigma).real

    @jax.jit
    def nll_loss(params, snap_batch, model_samples):
        log_amps_data = jax.vmap(lambda s: log_amp_at(params, s))(snap_batch)
        data_term = -2.0 * jnp.mean(log_amps_data)
        log_amps_model = jax.vmap(lambda s: log_amp_at(params, s))(model_samples)
        model_term = 2.0 * jnp.mean(log_amps_model)
        return data_term + model_term

    grad_fn = jax.jit(jax.value_and_grad(nll_loss))

    all_configs_jax = None
    exact_holdout_nll_fn = None
    if len(holdout_np) > 0 and N <= 16:
        dim = 2 ** N
        all_configs_np = np.empty((dim, N), dtype=np.float32)
        for idx in range(dim):
            for bit in range(N):
                all_configs_np[idx, bit] = 1.0 - 2.0 * ((idx >> bit) & 1)
        all_configs_jax = jnp.array(all_configs_np)

        @jax.jit
        def exact_holdout_nll_fn(params, heldout_batch):
            log_amps_all = jax.vmap(lambda s: log_amp_at(params, s))(all_configs_jax)
            logZ = jsp.special.logsumexp(2.0 * log_amps_all)
            log_amps_hold = jax.vmap(lambda s: log_amp_at(params, s))(heldout_batch)
            return -2.0 * jnp.mean(log_amps_hold) + logZ

    loss_trace = []
    holdout_nll_trace = []
    holdout_eval_iters = []

    if verbose:
        print(
            f"  NLL pretraining: {n_iter} iterations, {n_train} train / {n_holdout} holdout, "
            f"lr={lr}→{lr_final}"
        )
    t0 = time.time()

    for it in range(n_iter):
        batch_idx = rng.choice(n_train, size=batch_size, replace=False) if n_train > 0 else np.array([0])
        snap_batch = snap_jax[batch_idx] if n_train > 0 else jnp.array(snap_np[:1], dtype=jnp.float32)

        vstate.reset()
        vstate.sample()
        model_samples = jnp.array(vstate.samples.reshape(-1, N))

        loss, grads = grad_fn(vstate.parameters, snap_batch, model_samples)
        updates, opt_state = optimizer.update(grads, opt_state, vstate.parameters)
        vstate.parameters = ox.apply_updates(vstate.parameters, updates)

        loss_trace.append(float(loss))

        if exact_holdout_nll_fn is not None and ((it + 1) % eval_every == 0 or it == 0 or it + 1 == n_iter):
            held_nll = float(exact_holdout_nll_fn(vstate.parameters, holdout_jax))
            holdout_eval_iters.append(it + 1)
            holdout_nll_trace.append(held_nll)
        elif (it + 1) % eval_every == 0 or it == 0 or it + 1 == n_iter:
            holdout_eval_iters.append(it + 1)
            holdout_nll_trace.append(float('nan'))

        if verbose and ((it + 1) % max(1, n_iter // 10) == 0 or it == 0):
            msg = f"    iter {it + 1:4d}/{n_iter}  NLL_proxy={float(loss):.4f}"
            if holdout_nll_trace:
                last_hold = holdout_nll_trace[-1]
                if np.isfinite(last_hold):
                    msg += f"  holdout_NLL={last_hold:.4f}"
            print(msg)

    elapsed = time.time() - t0
    if verbose:
        print(f"  Pretraining done in {elapsed:.1f}s")

    try:
        E_stats = vstate.expect(H)
        E_pre = float(np.real(E_stats.mean))
        if verbose:
            print(f"  Post-pretrain energy: {E_pre:.8f}")
    except Exception:
        E_pre = None

    return vstate, {
        "loss_trace": loss_trace,
        "holdout_nll_trace": holdout_nll_trace,
        "holdout_eval_iters": holdout_eval_iters,
        "elapsed_s": elapsed,
        "E_post_pretrain": E_pre,
        "batch_size": batch_size,
        "lr": float(lr),
        "lr_final": float(lr_final),
        "n_train": n_train,
        "n_holdout": n_holdout,
    }


# ═══════════════════════════════════════════════════════════════
# PHASE 2: VMC + SR ENERGY REFINEMENT  (§6.3)
# ═══════════════════════════════════════════════════════════════

def refine_vmc(
    vstate,
    N,
    h,
    n_iter=600,
    n_samples=4096,
    lr=0.01,
    diag_shift=0.01,
    label="",
    E_exact=None,
    target_abs_err=1e-3,
    early_stop=False,
    check_every=10,
    min_stop_iter=0,
    stop_patience=1,
    lr_phase2=None,
    n_samples_phase2=None,
    phase_switch_iter=None,
    verbose=True,
):
    """VMC + SR refinement starting from the given variational state.

    When ``phase_switch_iter`` is set to an integer in ``(0, n_iter)``, the hybrid
    branch can use a two-phase schedule: phase 1 runs with ``lr`` / ``n_samples``
    and phase 2 continues from the same parameters with ``lr_phase2`` /
    ``n_samples_phase2``. This is useful when the pretrained state needs a gentle
    warm-start phase followed by a more aggressive polishing phase.
    """
    import netket as nk
    import optax

    g = nk.graph.Chain(length=N, pbc=True)
    hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes)
    H = nk.operator.Ising(hilbert=hi, graph=g, h=h, J=-1.0)

    total_iters = int(max(0, n_iter))
    base_samples = int(max(1, n_samples))
    phase2_samples = base_samples if n_samples_phase2 is None else int(max(1, n_samples_phase2))
    phase_switch = None if phase_switch_iter is None else int(phase_switch_iter)
    use_two_phase = bool(
        phase_switch is not None
        and 0 < phase_switch < total_iters
        and (lr_phase2 is not None or phase2_samples != base_samples)
    )
    lr2 = float(lr if lr_phase2 is None else lr_phase2)

    if not use_two_phase:
        vstate.n_samples = base_samples
        driver = _make_vmc_driver(nk, optax, H, vstate, lr=lr, diag_shift=diag_shift)
        run_info = run_vmc_driver(
            driver,
            n_iter=total_iters,
            E_exact=E_exact,
            target_abs_err=target_abs_err,
            early_stop=early_stop,
            check_every=check_every,
            min_stop_iter=min_stop_iter,
            stop_patience=stop_patience,
        )
        elapsed = float(run_info["elapsed_s"])
        energy_trace = np.real(np.asarray(run_info["energy_trace"], dtype=float)).reshape(-1)
        completed_iters = int(run_info.get("completed_iters", len(energy_trace)))
        requested_iters = int(run_info.get("requested_iters", total_iters))
        stopped_early = bool(run_info.get("stopped_early", False))
    else:
        phase1_iters = int(phase_switch)
        phase2_iters = int(total_iters - phase1_iters)
        stage_specs = [
            {
                "name": "phase1",
                "iters": phase1_iters,
                "lr": float(lr),
                "n_samples": base_samples,
            },
            {
                "name": "phase2",
                "iters": phase2_iters,
                "lr": float(lr2),
                "n_samples": phase2_samples,
            },
        ]

        traces = []
        elapsed = 0.0
        completed_iters = 0
        requested_iters = 0
        stopped_early = False

        for spec in stage_specs:
            if spec["iters"] <= 0:
                continue
            vstate.n_samples = int(spec["n_samples"])
            driver = _make_vmc_driver(
                nk, optax, H, vstate, lr=spec["lr"], diag_shift=diag_shift
            )
            stage_info = run_vmc_driver(
                driver,
                n_iter=int(spec["iters"]),
                E_exact=E_exact,
                target_abs_err=target_abs_err,
                early_stop=early_stop,
                check_every=check_every,
                min_stop_iter=min_stop_iter,
                stop_patience=stop_patience,
            )
            stage_trace = np.real(
                np.asarray(stage_info.get("energy_trace", []), dtype=float)
            ).reshape(-1)
            if len(stage_trace):
                traces.append(stage_trace)
            elapsed += float(stage_info.get("elapsed_s", 0.0))
            completed_iters += int(stage_info.get("completed_iters", len(stage_trace)))
            requested_iters += int(stage_info.get("requested_iters", spec["iters"]))
            if stage_info.get("stopped_early", False):
                stopped_early = True
                break

        energy_trace = (
            np.concatenate(traces).astype(float, copy=False)
            if traces else np.asarray([], dtype=float)
        )

    try:
        E_stats = vstate.expect(H)
        final_e = float(np.real(E_stats.mean))
        final_std = float(np.real(E_stats.error_of_mean))
    except Exception:
        final_e = float(energy_trace[-1]) if len(energy_trace) else float('nan')
        final_std = float("nan")

    iters_to_target = first_hit_iteration(energy_trace, E_exact, target_abs_err)
    secs_per_iter = elapsed / max(1, len(energy_trace))
    time_to_target_s = (
        float(iters_to_target * secs_per_iter)
        if np.isfinite(iters_to_target) else float("nan")
    )

    tag = f" ({label})" if label else ""
    if verbose:
        stop_msg = " [stopped early]" if stopped_early else ""
        if use_two_phase:
            schedule_msg = (
                f"lr={lr:.4g}→{lr2:.4g} @ {phase1_iters}"
                f", n_samples={base_samples}→{phase2_samples}"
            )
        else:
            schedule_msg = f"lr={lr}"
        print(
            f"  VMC refinement{tag}: {completed_iters}/{total_iters} iters, "
            f"{elapsed:.1f}s, {schedule_msg}{stop_msg}"
        )
        print(f"    E = {final_e:.8f} ± {final_std:.2e}")
        if E_exact is not None and target_abs_err is not None:
            if np.isfinite(iters_to_target):
                print(
                    f"    Reached |E−E_exact| ≤ {target_abs_err:.1e} at "
                    f"iter {int(iters_to_target)} (~{time_to_target_s:.2f}s)"
                )
            else:
                print(f"    Did not reach |E−E_exact| ≤ {target_abs_err:.1e}")

    return {
        "energy": final_e,
        "std": final_std,
        "n_params": int(vstate.n_parameters),
        "trace": energy_trace,
        "elapsed_s": elapsed,
        "iters_to_target": iters_to_target,
        "time_to_target_s": time_to_target_s,
        "target_abs_err": target_abs_err,
        "lr": lr,
        "lr_phase2": float(lr2) if use_two_phase else np.nan,
        "n_samples": int(base_samples),
        "n_samples_phase2": int(phase2_samples) if use_two_phase else np.nan,
        "phase_switch_iter": int(phase_switch) if use_two_phase else 0,
        "two_phase": bool(use_two_phase),
        "completed_iters": int(completed_iters),
        "requested_iters": int(requested_iters if requested_iters else total_iters),
        "stopped_early": bool(stopped_early),
    }


# ═══════════════════════════════════════════════════════════════
# COLD-START VMC (baseline for comparison)
# ═══════════════════════════════════════════════════════════════

def cold_start_vmc(
    N,
    h,
    alpha=2,
    n_iter=600,
    n_samples=4096,
    lr=0.01,
    diag_shift=0.01,
    E_exact=None,
    target_abs_err=1e-3,
    early_stop=False,
    check_every=10,
    min_stop_iter=0,
    stop_patience=1,
    verbose=True,
):
    """Standard VMC + SR from random initialization (no pretraining)."""
    import netket as nk
    import optax

    g = nk.graph.Chain(length=N, pbc=True)
    hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes)
    H = nk.operator.Ising(hilbert=hi, graph=g, h=h, J=-1.0)

    model = nk.models.RBM(alpha=alpha, param_dtype=float)
    sampler = nk.sampler.MetropolisLocal(hi, n_chains=16)
    vstate = nk.vqs.MCState(sampler, model, n_samples=n_samples)

    opt = optax.sgd(learning_rate=lr)
    sr = _make_sr_preconditioner(nk, diag_shift=diag_shift)
    driver = nk.driver.VMC(H, opt, variational_state=vstate, preconditioner=sr)

    run_info = run_vmc_driver(
        driver,
        n_iter=n_iter,
        E_exact=E_exact,
        target_abs_err=target_abs_err,
        early_stop=early_stop,
        check_every=check_every,
        min_stop_iter=min_stop_iter,
        stop_patience=stop_patience,
    )
    elapsed = float(run_info["elapsed_s"])
    energy_trace = np.real(np.asarray(run_info["energy_trace"], dtype=float)).reshape(-1)

    try:
        E_stats = vstate.expect(H)
        final_e = float(np.real(E_stats.mean))
        final_std = float(np.real(E_stats.error_of_mean))
    except Exception:
        final_e = float(energy_trace[-1]) if len(energy_trace) else float('nan')
        final_std = float("nan")

    iters_to_target = first_hit_iteration(energy_trace, E_exact, target_abs_err)
    secs_per_iter = elapsed / max(1, len(energy_trace))
    time_to_target_s = (
        float(iters_to_target * secs_per_iter)
        if np.isfinite(iters_to_target) else float("nan")
    )

    if verbose:
        stop_msg = " [stopped early]" if run_info.get("stopped_early") else ""
        print(
            f"  Cold-start VMC: {run_info['completed_iters']}/{n_iter} iters, "
            f"{elapsed:.1f}s, lr={lr}{stop_msg}"
        )
        print(f"    E = {final_e:.8f} ± {final_std:.2e}")
        if E_exact is not None and target_abs_err is not None:
            if np.isfinite(iters_to_target):
                print(
                    f"    Reached |E−E_exact| ≤ {target_abs_err:.1e} at "
                    f"iter {int(iters_to_target)} (~{time_to_target_s:.2f}s)"
                )
            else:
                print(f"    Did not reach |E−E_exact| ≤ {target_abs_err:.1e}")

    return {
        "energy": final_e,
        "std": final_std,
        "n_params": int(vstate.n_parameters),
        "trace": energy_trace,
        "elapsed_s": elapsed,
        "iters_to_target": iters_to_target,
        "time_to_target_s": time_to_target_s,
        "target_abs_err": target_abs_err,
        "lr": lr,
        "completed_iters": int(run_info.get("completed_iters", len(energy_trace))),
        "requested_iters": int(run_info.get("requested_iters", n_iter)),
        "stopped_early": bool(run_info.get("stopped_early", False)),
    }


# ═══════════════════════════════════════════════════════════════
# FIGURES
# ═══════════════════════════════════════════════════════════════

def make_figure(E_exact, pretrain_log, hybrid_result, cold_result,
                N, h, n_snapshots, noisy_results=None, output_prefix=None):
    """
    Three-panel figure (or four if noise test):
      (a) NLL pretraining loss convergence
      (b) VMC energy convergence: pretrained vs cold-start
      (c) Final energy bar chart
      (d) [Optional] Noise robustness: energy vs noise level
    """
    n_panels = 4 if noisy_results else 3
    fig, axes = plt.subplots(1, n_panels,
                             figsize=(5 * n_panels, 5))

    loss_trace = []
    if pretrain_log is not None and isinstance(pretrain_log, dict):
        loss_trace = list(pretrain_log.get("loss_trace", []))

    # ── (a) Pretraining NLL loss ──
    ax = axes[0]
    if len(loss_trace) > 0:
        ax.plot(loss_trace, color="#9C27B0", lw=1.2)
    else:
        ax.text(
            0.5, 0.5, "Pretraining unavailable",
            transform=ax.transAxes, ha="center", va="center", fontsize=11
        )
    ax.set_xlabel("Pretraining iteration", fontsize=11)
    ax.set_ylabel("NLL proxy loss", fontsize=11)
    ax.set_title(f"(a) Snapshot pretraining\n({n_snapshots} snapshots)",
                 fontsize=11)
    ax.grid(True, alpha=0.3)

    # ── (b) VMC convergence comparison ──
    ax = axes[1]
    if hybrid_result is not None:
        pretrain_iters = len(loss_trace)
        if pretrain_iters > 0:
            # Show pretraining phase as lighter prefix.
            ax.axvspan(
                0, pretrain_iters, alpha=0.08, color="#9C27B0",
                label="_pretrain region"
            )
            x_hybrid = np.arange(pretrain_iters, pretrain_iters + len(hybrid_result["trace"]))
            label = "Pretrained → VMC+SR"
        else:
            x_hybrid = np.arange(len(hybrid_result["trace"]))
            label = "Hybrid VMC+SR"
        ax.plot(x_hybrid, hybrid_result["trace"], color="#2196F3", lw=1.5, label=label)

    if cold_result is not None:
        ax.plot(cold_result["trace"], color="#FF9800", lw=1.5,
                label="Cold-start VMC+SR")

    if E_exact is not None:
        ax.axhline(E_exact, color="k", ls="--", lw=1, alpha=0.5,
                    label=f"ED exact = {E_exact:.4f}")

    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel("Energy", fontsize=11)
    ax.set_title("(b) VMC convergence", fontsize=11)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    # ── (c) Final energy bar chart ──
    ax = axes[2]
    labels, energies, errs, colors = [], [], [], []

    if E_exact is not None:
        labels.append("ED exact")
        energies.append(E_exact)
        errs.append(0)
        colors.append("#E53935")

    if hybrid_result is not None:
        labels.append("Hybrid\n(pretrain+VMC)")
        energies.append(hybrid_result["energy"])
        errs.append(hybrid_result["std"] if not np.isnan(hybrid_result["std"]) else 0)
        colors.append("#2196F3")

    if cold_result is not None:
        labels.append("Cold-start\nVMC")
        energies.append(cold_result["energy"])
        errs.append(cold_result["std"] if not np.isnan(cold_result["std"]) else 0)
        colors.append("#FF9800")

    x = np.arange(len(labels))
    ax.barh(x, energies, xerr=errs, color=colors, alpha=0.8, height=0.5)
    ax.set_yticks(x)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Energy", fontsize=11)
    ax.set_title("(c) Final energies", fontsize=11)
    ax.grid(True, alpha=0.3, axis="x")

    # ── (d) Noise robustness ──
    if noisy_results and len(axes) > 3:
        ax = axes[3]
        eps_vals = [r["epsilon"] for r in noisy_results]
        e_vals = [r["energy"] for r in noisy_results]
        e_stds = [r["std"] for r in noisy_results]

        ax.errorbar(eps_vals, e_vals, yerr=e_stds, fmt="o-",
                     color="#4CAF50", ms=6, lw=1.5, capsize=3)
        if E_exact is not None:
            ax.axhline(E_exact, color="k", ls="--", lw=1, alpha=0.5,
                        label="ED exact")
        ax.set_xlabel("Readout noise ε (per site)", fontsize=11)
        ax.set_ylabel("Final energy after refinement", fontsize=11)
        ax.set_title("(d) Noise robustness (§6.4)", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"Hybrid snapshot pretraining: 1D TFIM, N={N}, h={h}",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    fname = output_prefix or f"hybrid_N{N}_h{h:.1f}"
    plt.savefig(fname + ".png", dpi=150, bbox_inches="tight")
    plt.savefig(fname + ".pdf", bbox_inches="tight")
    print(f"\nSaved: {fname}.png / .pdf")


def _aligned_trace_stats(traces):
    """Return x, mean, sem for variable-length traces aligned by iteration index."""
    valid = [
        np.asarray(trace, dtype=float).reshape(-1)
        for trace in traces
        if trace is not None and len(np.asarray(trace).reshape(-1)) > 0
    ]
    if not valid:
        return (
            np.asarray([], dtype=float),
            np.asarray([], dtype=float),
            np.asarray([], dtype=float),
        )

    max_len = max(len(trace) for trace in valid)
    arr = np.full((len(valid), max_len), np.nan, dtype=float)
    for idx, trace in enumerate(valid):
        arr[idx, :len(trace)] = trace

    counts = np.sum(np.isfinite(arr), axis=0)
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(arr, axis=0)
        std = np.nanstd(arr, axis=0)
    sem = np.divide(
        std,
        np.sqrt(np.maximum(counts, 1)),
        out=np.full(max_len, np.nan, dtype=float),
        where=counts > 0,
    )
    x = np.arange(1, max_len + 1, dtype=float)
    return x, mean, sem


def make_multiseed_aggregate_figure(seed_runs, N, h, n_snapshots, output_prefix=None):
    """Create a multi-seed aggregate figure with per-seed and mean-over-seed diagnostics."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # (a) Held-out pretraining NLL across seeds
    ax = axes[0]
    holdout_xs = []
    holdout_ys = []
    for run in seed_runs:
        plog = run.get("pretrain_log") or {}
        xs = np.asarray(plog.get("holdout_eval_iters", []), dtype=float)
        ys = np.asarray(plog.get("holdout_nll_trace", []), dtype=float)
        if len(xs) and len(ys):
            mask = np.isfinite(xs) & np.isfinite(ys)
            xs = xs[mask]
            ys = ys[mask]
            if len(xs):
                holdout_xs.append(xs)
                holdout_ys.append(ys)
                ax.plot(xs, ys, lw=1.0, alpha=0.35)
    if holdout_xs:
        same_grid = all(
            len(xs) == len(holdout_xs[0]) and np.allclose(xs, holdout_xs[0])
            for xs in holdout_xs[1:]
        )
        if same_grid:
            arr = np.vstack([np.asarray(ys, dtype=float) for ys in holdout_ys])
            with np.errstate(invalid="ignore"):
                mean = np.nanmean(arr, axis=0)
                sem = np.nanstd(arr, axis=0) / np.sqrt(max(1, arr.shape[0]))
            ax.plot(holdout_xs[0], mean, color="#9C27B0", lw=2.0, label="mean")
            ax.fill_between(holdout_xs[0], mean - sem, mean + sem, color="#9C27B0", alpha=0.15)
            ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "Held-out NLL unavailable", transform=ax.transAxes,
                ha="center", va="center", fontsize=11)
    ax.set_xlabel("Pretraining iteration", fontsize=11)
    ax.set_ylabel("Held-out NLL", fontsize=11)
    ax.set_title("(a) Pretraining generalization", fontsize=11)
    ax.grid(True, alpha=0.3)

    # (b) Final absolute error per seed
    ax = axes[1]
    seed_labels = []
    hybrid_abs = []
    cold_abs = []
    for run in seed_runs:
        comp = compare_final_abs_errors(run["E_exact"], run.get("hybrid_result"), run.get("cold_result"))
        if comp is None:
            continue
        seed_labels.append(str(run["seed"]))
        hybrid_abs.append(comp["hybrid_abs_err"])
        cold_abs.append(comp["cold_abs_err"])
    if seed_labels:
        x = np.arange(len(seed_labels), dtype=float)
        width = 0.36
        ax.bar(x - width / 2, hybrid_abs, width=width, color="#2196F3", alpha=0.8, label="Hybrid")
        ax.bar(x + width / 2, cold_abs, width=width, color="#FF9800", alpha=0.8, label="Cold-start")
        if all(val > 0 for val in hybrid_abs + cold_abs):
            ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(seed_labels)
        ax.set_xlabel("Seed", fontsize=11)
        ax.set_ylabel("Final |E − E_exact|", fontsize=11)
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No per-seed error data", transform=ax.transAxes,
                ha="center", va="center", fontsize=11)
    ax.set_title("(b) Final absolute error by seed", fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")

    # (c) Iterations to target per seed
    ax = axes[2]
    hybrid_iters = []
    cold_iters = []
    iter_labels = []
    for run in seed_runs:
        hr = run.get("hybrid_result") or {}
        cr = run.get("cold_result") or {}
        h_it = hr.get("iters_to_target", np.nan)
        c_it = cr.get("iters_to_target", np.nan)
        if np.isfinite(h_it) or np.isfinite(c_it):
            iter_labels.append(str(run["seed"]))
            hybrid_iters.append(float(h_it) if np.isfinite(h_it) else np.nan)
            cold_iters.append(float(c_it) if np.isfinite(c_it) else np.nan)
    if iter_labels:
        x = np.arange(len(iter_labels), dtype=float)
        width = 0.36
        ax.bar(x - width / 2, hybrid_iters, width=width, color="#2196F3", alpha=0.8, label="Hybrid")
        ax.bar(x + width / 2, cold_iters, width=width, color="#FF9800", alpha=0.8, label="Cold-start")
        ax.set_xticks(x)
        ax.set_xticklabels(iter_labels)
        ax.set_xlabel("Seed", fontsize=11)
        ax.set_ylabel("Iterations to target", fontsize=11)
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "Target threshold not reached", transform=ax.transAxes,
                ha="center", va="center", fontsize=11)
    ax.set_title("(c) Convergence speed by seed", fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")

    # (d) Mean VMC convergence aligned by refinement iteration
    ax = axes[3]
    hybrid_traces = [
        (run.get("hybrid_result") or {}).get("trace", [])
        for run in seed_runs
    ]
    cold_traces = [
        (run.get("cold_result") or {}).get("trace", [])
        for run in seed_runs
    ]
    x_h, mean_h, sem_h = _aligned_trace_stats(hybrid_traces)
    x_c, mean_c, sem_c = _aligned_trace_stats(cold_traces)

    if len(x_h):
        ax.plot(x_h, mean_h, color="#2196F3", lw=1.8, label="Hybrid mean")
        ax.fill_between(x_h, mean_h - sem_h, mean_h + sem_h, color="#2196F3", alpha=0.15)
    if len(x_c):
        ax.plot(x_c, mean_c, color="#FF9800", lw=1.8, label="Cold-start mean")
        ax.fill_between(x_c, mean_c - sem_c, mean_c + sem_c, color="#FF9800", alpha=0.15)
    if seed_runs:
        ax.axhline(float(seed_runs[0]["E_exact"]), color="k", ls="--", lw=1, alpha=0.5,
                   label=f"ED exact = {float(seed_runs[0]['E_exact']):.4f}")
    if len(x_h) or len(x_c):
        ax.legend(fontsize=8, loc="upper right")
    else:
        ax.text(0.5, 0.5, "No VMC traces available", transform=ax.transAxes,
                ha="center", va="center", fontsize=11)
    ax.set_xlabel("Refinement iteration", fontsize=11)
    ax.set_ylabel("Energy", fontsize=11)
    ax.set_title("(d) Mean VMC convergence", fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.suptitle(
        f"Hybrid snapshot pretraining: 1D TFIM, N={N}, h={h} (multi-seed aggregate)",
        fontsize=13, y=1.02
    )
    plt.tight_layout()
    fname = output_prefix or f"hybrid_N{N}_h{h:.1f}_multiseed"
    plt.savefig(fname + ".png", dpi=150, bbox_inches="tight")
    plt.savefig(fname + ".pdf", bbox_inches="tight")
    print(f"Saved: {fname}.png / .pdf")


# ═══════════════════════════════════════════════════════════════
# CSV LOGGING
# ═══════════════════════════════════════════════════════════════

def save_trace_csv(pretrain_log, hybrid_result, cold_result, N, h, csv_path=None):
    """Save per-iteration traces (pretrain NLL, hybrid VMC, cold-start VMC)."""
    if csv_path is None:
        csv_path = f"hybrid_N{N}_h{h:.1f}_results.csv"

    pretrain_trace = np.asarray(
        (pretrain_log or {}).get("loss_trace", []), dtype=float
    )
    hybrid_trace = np.asarray(
        (hybrid_result or {}).get("trace", []), dtype=float
    )
    cold_trace = np.asarray(
        (cold_result or {}).get("trace", []), dtype=float
    )

    max_len = int(max(len(pretrain_trace), len(hybrid_trace), len(cold_trace), 1))
    pre_col = np.full(max_len, np.nan)
    hy_col = np.full(max_len, np.nan)
    cold_col = np.full(max_len, np.nan)
    pre_col[:len(pretrain_trace)] = pretrain_trace
    hy_col[:len(hybrid_trace)] = hybrid_trace
    cold_col[:len(cold_trace)] = cold_trace

    iters = np.arange(1, max_len + 1, dtype=float)
    data = np.column_stack([iters, pre_col, hy_col, cold_col])
    header = "iteration,pretrain_nll_proxy,hybrid_vmc_energy,coldstart_vmc_energy"
    np.savetxt(csv_path, data, delimiter=",", header=header, comments="", fmt="%.10f")
    return csv_path


def save_summary_csv(
    E_exact,
    pretrain_log,
    hybrid_result,
    cold_result,
    noisy_results,
    N,
    h,
    n_snapshots,
    summary_csv_path=None,
):
    """Save run-level summary metrics and optional noise sweep to CSV."""
    if summary_csv_path is None:
        summary_csv_path = f"hybrid_N{N}_h{h:.1f}_summary.csv"

    rows = []
    rows.append(
        {
            "section": "meta",
            "name": "run_config",
            "N": N,
            "h": h,
            "n_snapshots": n_snapshots,
            "epsilon": "",
            "energy": "",
            "std": "",
            "rel_err": "",
            "elapsed_s": "",
            "n_params": "",
            "value": "",
        }
    )
    rows.append(
        {
            "section": "summary",
            "name": "ed_exact",
            "N": N,
            "h": h,
            "n_snapshots": n_snapshots,
            "epsilon": "",
            "energy": float(E_exact),
            "std": "",
            "rel_err": 0.0,
            "elapsed_s": "",
            "n_params": "",
            "value": "",
        }
    )

    if pretrain_log:
        rows.append(
            {
                "section": "summary",
                "name": "pretraining",
                "N": N,
                "h": h,
                "n_snapshots": n_snapshots,
                "epsilon": "",
                "energy": pretrain_log.get("E_post_pretrain", ""),
                "std": "",
                "rel_err": "",
                "elapsed_s": pretrain_log.get("elapsed_s", ""),
                "n_params": "",
                "value": len(pretrain_log.get("loss_trace", [])),
            }
        )

    if hybrid_result:
        rel_h = abs(hybrid_result["energy"] - E_exact) / abs(E_exact)
        rows.append(
            {
                "section": "summary",
                "name": "hybrid_refined",
                "N": N,
                "h": h,
                "n_snapshots": n_snapshots,
                "epsilon": "",
                "energy": hybrid_result["energy"],
                "std": hybrid_result["std"],
                "rel_err": rel_h,
                "elapsed_s": hybrid_result.get("elapsed_s", ""),
                "n_params": hybrid_result.get("n_params", ""),
                "value": "",
            }
        )

    if cold_result:
        rel_c = abs(cold_result["energy"] - E_exact) / abs(E_exact)
        rows.append(
            {
                "section": "summary",
                "name": "cold_start",
                "N": N,
                "h": h,
                "n_snapshots": n_snapshots,
                "epsilon": "",
                "energy": cold_result["energy"],
                "std": cold_result["std"],
                "rel_err": rel_c,
                "elapsed_s": cold_result.get("elapsed_s", ""),
                "n_params": cold_result.get("n_params", ""),
                "value": "",
            }
        )

    comparison = compare_final_abs_errors(E_exact, hybrid_result, cold_result)
    if comparison is not None:
        rows.append(
            {
                "section": "summary",
                "name": "hybrid_abs_error",
                "N": N,
                "h": h,
                "n_snapshots": n_snapshots,
                "epsilon": "",
                "energy": "",
                "std": "",
                "rel_err": "",
                "elapsed_s": "",
                "n_params": "",
                "value": comparison["hybrid_abs_err"],
            }
        )
        rows.append(
            {
                "section": "summary",
                "name": "cold_abs_error",
                "N": N,
                "h": h,
                "n_snapshots": n_snapshots,
                "epsilon": "",
                "energy": "",
                "std": "",
                "rel_err": "",
                "elapsed_s": "",
                "n_params": "",
                "value": comparison["cold_abs_err"],
            }
        )
        rows.append(
            {
                "section": "summary",
                "name": "abs_error_gap_cold_minus_hybrid",
                "N": N,
                "h": h,
                "n_snapshots": n_snapshots,
                "epsilon": "",
                "energy": "",
                "std": "",
                "rel_err": "",
                "elapsed_s": "",
                "n_params": "",
                "value": comparison["gap_cold_minus_hybrid"],
            }
        )
        rows.append(
            {
                "section": "summary",
                "name": "final_abs_error_winner",
                "N": N,
                "h": h,
                "n_snapshots": n_snapshots,
                "epsilon": "",
                "energy": "",
                "std": "",
                "rel_err": "",
                "elapsed_s": "",
                "n_params": "",
                "value": comparison["winner"],
            }
        )

    if noisy_results:
        for row in noisy_results:
            eps = row.get("epsilon", np.nan)
            e = row.get("energy", np.nan)
            rel = np.nan if not np.isfinite(e) else abs(e - E_exact) / abs(E_exact)
            rows.append(
                {
                    "section": "noise_sweep",
                    "name": "noise_point",
                    "N": N,
                    "h": h,
                    "n_snapshots": n_snapshots,
                    "epsilon": eps,
                    "energy": e,
                    "std": row.get("std", np.nan),
                    "rel_err": rel,
                    "elapsed_s": "",
                    "n_params": "",
                    "value": "",
                }
            )

    fieldnames = [
        "section", "name", "N", "h", "n_snapshots", "epsilon",
        "energy", "std", "rel_err", "elapsed_s", "n_params", "value",
    ]
    with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return summary_csv_path


def with_seed_suffix(path, seed, default_path):
    """Return a file path with _seed<seed> inserted before the extension."""
    base = path if path is not None else default_path
    root, ext = os.path.splitext(base)
    if not ext:
        return f"{base}_seed{seed}"
    return f"{root}_seed{seed}{ext}"


def save_multiseed_summary_csv(seed_runs, N, h, n_snapshots, csv_path=None):
    """Save per-seed and aggregate metrics for a multi-seed benchmark."""
    if csv_path is None:
        csv_path = f"hybrid_N{N}_h{h:.1f}_multiseed.csv"

    rows = []
    for run in seed_runs:
        comparison = compare_final_abs_errors(
            run["E_exact"], run.get("hybrid_result"), run.get("cold_result")
        )
        hybrid_result = run.get("hybrid_result") or {}
        cold_result = run.get("cold_result") or {}
        E_exact = float(run["E_exact"])
        hybrid_energy = float(hybrid_result.get("energy", np.nan))
        cold_energy = float(cold_result.get("energy", np.nan))
        hybrid_rel = np.nan if not np.isfinite(hybrid_energy) else abs(hybrid_energy - E_exact) / abs(E_exact)
        cold_rel = np.nan if not np.isfinite(cold_energy) else abs(cold_energy - E_exact) / abs(E_exact)
        hybrid_iters = hybrid_result.get("iters_to_target", np.nan)
        cold_iters = cold_result.get("iters_to_target", np.nan)
        hybrid_time = hybrid_result.get("time_to_target_s", np.nan)
        cold_time = cold_result.get("time_to_target_s", np.nan)
        pretrain_elapsed = (run.get("pretrain_log") or {}).get("elapsed_s", np.nan)

        hybrid_faster = int(
            np.isfinite(hybrid_iters) and np.isfinite(cold_iters) and hybrid_iters < cold_iters
        )
        if comparison is None:
            hybrid_better = 0
            winner = ""
            hybrid_abs = np.nan
            cold_abs = np.nan
            gap = np.nan
        else:
            hybrid_better = int(comparison["winner"] == "hybrid")
            winner = comparison["winner"]
            hybrid_abs = comparison["hybrid_abs_err"]
            cold_abs = comparison["cold_abs_err"]
            gap = comparison["gap_cold_minus_hybrid"]

        rows.append({
            "seed": int(run["seed"]),
            "E_exact": E_exact,
            "pretrain_energy": (run.get("pretrain_log") or {}).get("E_post_pretrain", np.nan),
            "pretrain_elapsed_s": pretrain_elapsed,
            "hybrid_energy": hybrid_energy,
            "hybrid_std": hybrid_result.get("std", np.nan),
            "hybrid_abs_err": hybrid_abs,
            "hybrid_rel_err": hybrid_rel,
            "hybrid_elapsed_s": hybrid_result.get("elapsed_s", np.nan),
            "hybrid_iters_to_target": hybrid_iters,
            "hybrid_time_to_target_s": hybrid_time,
            "hybrid_completed_iters": hybrid_result.get("completed_iters", np.nan),
            "cold_energy": cold_energy,
            "cold_std": cold_result.get("std", np.nan),
            "cold_abs_err": cold_abs,
            "cold_rel_err": cold_rel,
            "cold_elapsed_s": cold_result.get("elapsed_s", np.nan),
            "cold_iters_to_target": cold_iters,
            "cold_time_to_target_s": cold_time,
            "cold_completed_iters": cold_result.get("completed_iters", np.nan),
            "abs_error_gap_cold_minus_hybrid": gap,
            "winner_final_abs_error": winner,
            "hybrid_better_final_abs_error": hybrid_better,
            "hybrid_faster_to_target": hybrid_faster,
        })

    if rows:
        mean_row = {"seed": "mean"}
        numeric_keys = [
            k for k, v in rows[0].items()
            if k != "seed" and isinstance(v, (int, float, np.floating))
        ]
        for key in numeric_keys:
            values = [row[key] for row in rows]
            arr = np.asarray(values, dtype=float)
            mean_row[key] = float(np.nanmean(arr))
        mean_row["winner_final_abs_error"] = "mixed"
        rows.append(mean_row)

    fieldnames = [
        "seed", "E_exact", "pretrain_energy", "pretrain_elapsed_s",
        "hybrid_energy", "hybrid_std", "hybrid_abs_err", "hybrid_rel_err",
        "hybrid_elapsed_s", "hybrid_iters_to_target", "hybrid_time_to_target_s",
        "hybrid_completed_iters", "cold_energy", "cold_std", "cold_abs_err",
        "cold_rel_err", "cold_elapsed_s", "cold_iters_to_target",
        "cold_time_to_target_s", "cold_completed_iters",
        "abs_error_gap_cold_minus_hybrid", "winner_final_abs_error",
        "hybrid_better_final_abs_error", "hybrid_faster_to_target",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def run_single_seed(args, seed):
    """Run the full pipeline for one seed and return all outputs."""
    N = args.N

    print(f"Hybrid Snapshot Pretraining: N={N}, h={args.h}, M={args.n_snapshots} snapshots")
    print(f"Architecture: RBM, alpha={args.alpha}")
    print(f"Seed: {seed}")
    print()

    print("=" * 55)
    print("STEP 0: Generate synthetic snapshots (Born sampling)")
    print("=" * 55)
    snapshots, psi0, E_exact = generate_snapshots_ed(
        N, args.h, args.n_snapshots, seed=seed
    )
    print(f"  ED ground state: E0 = {E_exact:.10f}  "
          f"(E0/N = {E_exact/N:.10f})")
    print(f"  Generated {args.n_snapshots} Z-basis snapshots")

    mz_per_snap = np.mean(snapshots, axis=1)
    print(f"  Snapshot ⟨m_z⟩ = {np.mean(mz_per_snap):.4f} ± "
          f"{np.std(mz_per_snap) / np.sqrt(len(mz_per_snap)):.4f}")
    print()

    print("=" * 55)
    print("STEP 1: NLL pretraining from snapshots (§6.1)")
    print("=" * 55)
    pretrain_log = None
    hybrid_result = None
    try:
        vstate_pre, pretrain_log = pretrain_nll(
            N, args.h, snapshots, alpha=args.alpha,
            n_iter=args.n_pretrain, n_samples=args.pretrain_samples,
            lr=args.pretrain_lr, lr_final=args.pretrain_lr_final,
            batch_size=args.pretrain_batch_size, seed=seed,
            holdout_frac=args.pretrain_holdout_frac,
            eval_every=args.pretrain_eval_every
        )
        print()

        print("=" * 55)
        print("STEP 2: VMC+SR refinement from pretrained state (§6.3)")
        print("=" * 55)
        hybrid_phase2_lr = None
        hybrid_phase2_samples = None
        hybrid_phase_switch = None
        if args.hybrid_two_phase:
            hybrid_phase2_lr = (
                args.refine_lr_hybrid_phase2
                if args.refine_lr_hybrid_phase2 is not None
                else args.refine_lr_cold
            )
            hybrid_phase2_samples = (
                args.refine_samples_hybrid_phase2
                if args.refine_samples_hybrid_phase2 is not None
                else max(args.refine_samples, 2 * args.refine_samples)
            )
            hybrid_phase_switch = args.hybrid_phase_switch_iter

        hybrid_result = refine_vmc(
            vstate_pre, N, args.h, n_iter=args.n_refine,
            n_samples=args.refine_samples, lr=args.refine_lr_hybrid,
            label="hybrid", E_exact=E_exact, target_abs_err=args.target_abs_err,
            early_stop=args.hybrid_early_stop,
            check_every=args.early_stop_check_every,
            min_stop_iter=args.min_refine_iters_hybrid,
            stop_patience=args.early_stop_patience,
            lr_phase2=hybrid_phase2_lr,
            n_samples_phase2=hybrid_phase2_samples,
            phase_switch_iter=hybrid_phase_switch,
        )
        print()

    except Exception as e:
        print(f"  Pretraining/refinement failed: {e}")
        print("  Check that netket is installed.")
        print()

    print("=" * 55)
    print("STEP 3: Cold-start VMC baseline (random init)")
    print("=" * 55)
    cold_result = None
    try:
        cold_result = cold_start_vmc(
            N, args.h, alpha=args.alpha, n_iter=args.n_refine,
            n_samples=args.refine_samples, lr=args.refine_lr_cold,
            E_exact=E_exact, target_abs_err=args.target_abs_err,
            early_stop=args.cold_early_stop,
            check_every=args.early_stop_check_every,
            min_stop_iter=args.min_refine_iters_cold,
            stop_patience=args.early_stop_patience
        )
    except Exception as e:
        print(f"  Cold-start VMC failed: {e}")
    print()

    noisy_results = None
    if args.noise_test:
        print("=" * 55)
        print("STEP 4: Noise robustness test (§6.4)")
        print("=" * 55)
        print("  Per-site bit-flip noise: ε = 0, 0.02, 0.05, 0.10, 0.20")
        noisy_results = []

        for eps in [0.0, 0.02, 0.05, 0.10, 0.20]:
            print(f"\n  ── ε = {eps:.2f} ──")
            noisy_snaps = add_readout_noise(snapshots, eps) if eps > 0 else snapshots

            try:
                vstate_n, _ = pretrain_nll(
                    N, args.h, noisy_snaps, alpha=args.alpha,
                    n_iter=args.n_pretrain, n_samples=args.pretrain_samples,
                    lr=args.pretrain_lr, lr_final=args.pretrain_lr_final,
                    batch_size=args.pretrain_batch_size, seed=seed,
                    holdout_frac=args.pretrain_holdout_frac,
                    eval_every=args.pretrain_eval_every
                )
                noise_phase2_lr = None
                noise_phase2_samples = None
                noise_phase_switch = None
                if args.hybrid_two_phase:
                    noise_phase2_lr = (
                        args.refine_lr_hybrid_phase2
                        if args.refine_lr_hybrid_phase2 is not None
                        else args.refine_lr_cold
                    )
                    noise_phase2_samples = (
                        args.refine_samples_hybrid_phase2
                        if args.refine_samples_hybrid_phase2 is not None
                        else max(args.refine_samples, 2 * args.refine_samples)
                    )
                    noise_phase_switch = args.hybrid_phase_switch_iter

                result_n = refine_vmc(
                    vstate_n, N, args.h, n_iter=args.n_refine,
                    n_samples=args.refine_samples, lr=args.refine_lr_hybrid,
                    label=f"ε={eps:.2f}", E_exact=E_exact,
                    target_abs_err=args.target_abs_err,
                    lr_phase2=noise_phase2_lr,
                    n_samples_phase2=noise_phase2_samples,
                    phase_switch_iter=noise_phase_switch,
                )
                noisy_results.append({
                    "epsilon": eps,
                    "energy": result_n["energy"],
                    "std": result_n["std"],
                })
            except Exception as e:
                print(f"    Failed at ε={eps}: {e}")
                noisy_results.append({
                    "epsilon": eps,
                    "energy": float("nan"),
                    "std": float("nan"),
                })
        print()

    print("=" * 55)
    print("SUMMARY")
    print("=" * 55)
    print(f"  ED exact:    {E_exact:.10f}")
    if hybrid_result:
        rel_h = abs(hybrid_result["energy"] - E_exact) / abs(E_exact)
        print(f"  Hybrid:      {hybrid_result['energy']:.10f}  "
              f"(rel err = {rel_h:.2e})")
    if cold_result:
        rel_c = abs(cold_result["energy"] - E_exact) / abs(E_exact)
        print(f"  Cold-start:  {cold_result['energy']:.10f}  "
              f"(rel err = {rel_c:.2e})")

    comparison = compare_final_abs_errors(E_exact, hybrid_result, cold_result)
    if comparison is not None:
        print(f"  Hybrid abs err: {comparison['hybrid_abs_err']:.2e}")
        print(f"  Cold abs err:   {comparison['cold_abs_err']:.2e}")
        print(
            f"  Abs-error gap (cold−hybrid): {comparison['gap_cold_minus_hybrid']:.2e} "
            f"({_winner_label(comparison['winner'])})"
        )
    if noisy_results:
        print(f"  Noise sweep: {len(noisy_results)} points (see figure panel d)")
    print()

    return {
        "seed": int(seed),
        "N": N,
        "h": args.h,
        "n_snapshots": args.n_snapshots,
        "snapshots": snapshots,
        "psi0": psi0,
        "E_exact": E_exact,
        "pretrain_log": pretrain_log,
        "hybrid_result": hybrid_result,
        "cold_result": cold_result,
        "noisy_results": noisy_results,
    }


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Hybrid snapshot pretraining for 1D TFIM")
    parser.add_argument("--N", type=int, default=10,
                        help="Chain length")
    parser.add_argument("--h", type=float, default=1.0,
                        help="Transverse field")
    parser.add_argument("--n-snapshots", type=int, default=2000,
                        help="Number of synthetic snapshots")
    parser.add_argument("--alpha", type=int, default=2,
                        help="RBM hidden density")
    parser.add_argument("--n-pretrain", type=int, default=300,
                        help="NLL pretraining iterations")
    parser.add_argument("--n-refine", type=int, default=600,
                        help="VMC refinement iterations")
    parser.add_argument("--noise-test", action="store_true",
                        help="Run noise robustness sweep (§6.4)")
    parser.add_argument("--results-csv-path", type=str, default=None,
                        help="Optional path for per-iteration results CSV")
    parser.add_argument("--summary-csv-path", type=str, default=None,
                        help="Optional path for summary CSV")
    parser.add_argument("--pretrain-samples", type=int, default=2048,
                        help="MC samples used during NLL pretraining")
    parser.add_argument("--refine-samples", type=int, default=4096,
                        help="MC samples used during VMC refinement")
    parser.add_argument("--pretrain-lr", type=float, default=0.01,
                        help="Initial learning rate for pretraining")
    parser.add_argument("--pretrain-lr-final", type=float, default=None,
                        help="Final pretraining LR for linear decay (defaults to constant)")
    parser.add_argument("--pretrain-batch-size", type=int, default=512,
                        help="Mini-batch size for snapshot pretraining")
    parser.add_argument("--pretrain-holdout-frac", type=float, default=0.1,
                        help="Fraction of snapshots reserved for held-out NLL")
    parser.add_argument("--pretrain-eval-every", type=int, default=10,
                        help="Evaluate held-out NLL every this many pretrain steps")
    parser.add_argument("--refine-lr-hybrid", type=float, default=0.006,
                        help="VMC LR for the pretrained branch")
    parser.add_argument("--refine-lr-cold", type=float, default=0.01,
                        help="VMC LR for the cold-start branch")
    parser.add_argument("--hybrid-two-phase", action="store_true",
                        help="Use a two-phase VMC schedule for the pretrained branch")
    parser.add_argument("--hybrid-phase-switch-iter", type=int, default=100,
                        help="Hybrid VMC iteration where phase 2 starts")
    parser.add_argument("--refine-lr-hybrid-phase2", type=float, default=None,
                        help="Phase-2 VMC LR for the pretrained branch (defaults to --refine-lr-cold)")
    parser.add_argument("--refine-samples-hybrid-phase2", type=int, default=None,
                        help="Phase-2 MC samples for the pretrained branch (defaults to 2x --refine-samples)")
    parser.add_argument("--target-abs-err", type=float, default=1e-3,
                        help="Target |E-E_exact| for convergence checks")
    parser.add_argument("--hybrid-early-stop", action="store_true",
                        help="Enable early stopping for the hybrid branch")
    parser.add_argument("--cold-early-stop", action="store_true",
                        help="Enable early stopping for the cold-start branch")
    parser.add_argument("--early-stop-check-every", type=int, default=10,
                        help="Check convergence every this many VMC iters")
    parser.add_argument("--early-stop-patience", type=int, default=3,
                        help="Require this many consecutive threshold hits before stopping")
    parser.add_argument("--min-refine-iters-hybrid", type=int, default=0,
                        help="Minimum hybrid VMC iterations before early stop can trigger")
    parser.add_argument("--min-refine-iters-cold", type=int, default=0,
                        help="Minimum cold-start VMC iterations before early stop can trigger")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base seed for the first run")
    parser.add_argument("--n-seeds", type=int, default=1,
                        help="Number of sequential seeds to run")
    parser.add_argument("--seed-stride", type=int, default=1,
                        help="Increment between consecutive seeds")
    args = parser.parse_args()

    N = args.N
    if N > 16:
        print(f"WARNING: N={N} is too large for ED snapshot generation "
              f"(2^{N} = {2**N:,}).")
        print("  For N > 16, use DMRG snapshots from baselines/ instead.")
        return

    n_seeds = int(max(1, args.n_seeds))
    seed_stride = int(max(1, args.seed_stride))
    seeds = [int(args.seed + i * seed_stride) for i in range(n_seeds)]

    seed_runs = []
    for idx, seed in enumerate(seeds, start=1):
        if n_seeds > 1:
            print()
            print("#" * 72)
            print(f"RUN {idx}/{n_seeds}  (seed={seed})")
            print("#" * 72)
        run = run_single_seed(args, seed)
        seed_runs.append(run)

        default_results = f"hybrid_N{N}_h{args.h:.1f}_results.csv"
        default_summary = f"hybrid_N{N}_h{args.h:.1f}_summary.csv"
        results_csv = save_trace_csv(
            run.get("pretrain_log"), run.get("hybrid_result"), run.get("cold_result"),
            N, args.h, csv_path=with_seed_suffix(args.results_csv_path, seed, default_results)
        )
        summary_csv = save_summary_csv(
            run["E_exact"], run.get("pretrain_log"), run.get("hybrid_result"),
            run.get("cold_result"), run.get("noisy_results"),
            N, args.h, args.n_snapshots,
            summary_csv_path=with_seed_suffix(args.summary_csv_path, seed, default_summary)
        )
        if n_seeds == 1:
            print(f"Saved: {results_csv}")
            print(f"Saved: {summary_csv}")
            print()

    if n_seeds == 1:
        run = seed_runs[0]
        if run.get("pretrain_log") or run.get("cold_result"):
            make_figure(
                run["E_exact"], run.get("pretrain_log"), run.get("hybrid_result"),
                run.get("cold_result"), N, args.h, args.n_snapshots, run.get("noisy_results")
            )
        return

    aggregate_csv = save_multiseed_summary_csv(
        seed_runs, N, args.h, args.n_snapshots, csv_path=args.summary_csv_path
    )

    hybrid_better = 0
    hybrid_faster = 0
    hybrid_rel = []
    cold_rel = []
    hybrid_iters = []
    cold_iters = []
    for run in seed_runs:
        comparison = compare_final_abs_errors(
            run["E_exact"], run.get("hybrid_result"), run.get("cold_result")
        )
        if comparison is not None and comparison["winner"] == "hybrid":
            hybrid_better += 1

        hr = run.get("hybrid_result") or {}
        cr = run.get("cold_result") or {}
        if hr and np.isfinite(hr.get("iters_to_target", np.nan)) and cr and np.isfinite(cr.get("iters_to_target", np.nan)):
            if hr["iters_to_target"] < cr["iters_to_target"]:
                hybrid_faster += 1
        if hr and np.isfinite(hr.get("energy", np.nan)):
            hybrid_rel.append(abs(hr["energy"] - run["E_exact"]) / abs(run["E_exact"]))
        if cr and np.isfinite(cr.get("energy", np.nan)):
            cold_rel.append(abs(cr["energy"] - run["E_exact"]) / abs(run["E_exact"]))
        if hr and np.isfinite(hr.get("iters_to_target", np.nan)):
            hybrid_iters.append(hr["iters_to_target"])
        if cr and np.isfinite(cr.get("iters_to_target", np.nan)):
            cold_iters.append(cr["iters_to_target"])

    print("=" * 55)
    print("MULTI-SEED SUMMARY")
    print("=" * 55)
    print(f"  Seeds run: {seeds}")
    print(f"  Hybrid better final abs error: {hybrid_better}/{n_seeds}")
    print(f"  Hybrid faster to target:      {hybrid_faster}/{n_seeds}")
    if hybrid_rel:
        print(f"  Mean hybrid rel err: {np.nanmean(np.asarray(hybrid_rel, dtype=float)):.2e}")
    if cold_rel:
        print(f"  Mean cold rel err:   {np.nanmean(np.asarray(cold_rel, dtype=float)):.2e}")
    if hybrid_iters:
        print(f"  Mean hybrid iters-to-target: {np.nanmean(np.asarray(hybrid_iters, dtype=float)):.1f}")
    if cold_iters:
        print(f"  Mean cold iters-to-target:   {np.nanmean(np.asarray(cold_iters, dtype=float)):.1f}")
    print(f"Saved: {aggregate_csv}")
    print()

    first_run = seed_runs[0]
    if first_run.get("pretrain_log") or first_run.get("cold_result"):
        prefix = f"hybrid_N{N}_h{args.h:.1f}_seed{first_run['seed']}"
        make_figure(
            first_run["E_exact"], first_run.get("pretrain_log"),
            first_run.get("hybrid_result"), first_run.get("cold_result"),
            N, args.h, args.n_snapshots, first_run.get("noisy_results"),
            output_prefix=prefix,
        )

    if seed_runs:
        make_multiseed_aggregate_figure(
            seed_runs,
            N,
            args.h,
            args.n_snapshots,
            output_prefix=f"hybrid_N{N}_h{args.h:.1f}_multiseed",
        )


if __name__ == "__main__":
    main()
