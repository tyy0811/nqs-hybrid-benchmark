"""
03: Quench Dynamics with TDVP
============================
1D TFIM quench: prepare the ground state of H(h_i), quench to H(h_f),
and compare NQS TDVP dynamics against exact diagonalization (ED).

This merged version keeps:
  - NetKet TDVP compatibility fallbacks across old/new APIs
  - explicit holomorphic=False handling for SR/QGT when supported
  - symmetry-aware ansatz selection (--ansatz auto|rbm|translation|z2|translation_z2)
  - CSV output with symmetry-safe observables
  - parity-even validation metrics (m_x, c_zz, energy)
  - richer summary diagnostics

Primary validation observables:
  - m_x(t)  = (1/N) sum_i <X_i>_t
  - c_zz(t) = (1/N) sum_i <Z_i Z_{i+1}>_t
  - <H_f>_t

Diagnostic only:
  - m_z(t)  = (1/N) sum_i <Z_i>_t

Requirements:
    netket >= 3.0, jax, optax, scipy, numpy, matplotlib
    Python >= 3.11
"""

import argparse
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np

try:
    import jax.numpy as jnp
    from flax import linen as nn
except Exception:  # pragma: no cover - only relevant if jax/flax are missing
    jnp = None
    nn = None


def _complex_logaddexp_pair(a, b):
    """Numerically stable log(exp(a) + exp(b)) for complex-valued log-amplitudes."""
    if jnp is None:
        raise RuntimeError("jax is required for the explicit Z2-symmetric ansatz")
    pivot = jnp.where(jnp.real(a) >= jnp.real(b), a, b)
    return pivot + jnp.log(jnp.exp(a - pivot) + jnp.exp(b - pivot))


if nn is not None:
    class _Z2SymmetrizedModel(nn.Module):
        """Even-parity wrapper: psi(s) = phi(s) + phi(-s), in log-amplitude form."""

        base_model: nn.Module

        def __call__(self, x):
            x = jnp.asarray(x)
            log_phi = self.base_model(x)
            log_phi_flip = self.base_model(-x)
            return _complex_logaddexp_pair(log_phi, log_phi_flip)
else:  # pragma: no cover - only relevant if flax is missing
    _Z2SymmetrizedModel = None


def make_output_stem(N, h_i, h_f):
    """Consistent base filename for figure/data outputs."""
    return f"quench_N{N}_h{h_i}-{h_f}"


def _series_on_grid(result, key, target_times):
    """Return a result series on a target time grid, or NaNs if absent."""
    values = None if result is None else result.get(key)
    times = None if result is None else result.get("times")
    if values is None or times is None:
        return np.full(len(target_times), np.nan)

    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    target_times = np.asarray(target_times, dtype=float)

    if len(times) == len(target_times) and np.allclose(times, target_times):
        return values
    if len(times) == 1:
        return np.full(len(target_times), values[0])

    return np.interp(target_times, times, values)


def _safe_rel_mae(reference, abs_err):
    """Relative MAE against the mean absolute scale of the reference series."""
    reference = np.asarray(reference, dtype=float)
    abs_err = np.asarray(abs_err, dtype=float)
    mask = np.isfinite(reference) & np.isfinite(abs_err)
    if not np.any(mask):
        return np.nan
    scale = np.mean(np.abs(reference[mask]))
    if scale <= 0:
        return np.nan
    return float(np.mean(abs_err[mask]) / scale)


def _split_halves_mae(abs_err):
    """Return early-half and late-half MAE to separate bias from drift."""
    abs_err = np.asarray(abs_err, dtype=float)
    mask = np.isfinite(abs_err)
    valid = abs_err[mask]
    if len(valid) == 0:
        return np.nan, np.nan
    mid = max(1, len(valid) // 2)
    early = float(np.mean(valid[:mid]))
    late = float(np.mean(valid[mid:])) if len(valid[mid:]) > 0 else early
    return early, late


def save_results_csv(ed_result, nqs_result, N, h_i, h_f, csv_path=None):
    """Save all available trajectories plus error columns to a CSV file."""
    if ed_result is None and nqs_result is None:
        return None

    time_grids = []
    if ed_result is not None and "times" in ed_result:
        time_grids.append(np.asarray(ed_result["times"], dtype=float))
    if nqs_result is not None and "times" in nqs_result:
        time_grids.append(np.asarray(nqs_result["times"], dtype=float))
    target_times = np.unique(np.concatenate(time_grids))

    ed_mz = _series_on_grid(ed_result, "mz_t", target_times)
    ed_mx = _series_on_grid(ed_result, "mx_t", target_times)
    ed_czz = _series_on_grid(ed_result, "czz_t", target_times)
    ed_energy = _series_on_grid(ed_result, "energy_t", target_times)

    nqs_mz = _series_on_grid(nqs_result, "mz_t", target_times)
    nqs_mx = _series_on_grid(nqs_result, "mx_t", target_times)
    nqs_czz = _series_on_grid(nqs_result, "czz_t", target_times)
    nqs_energy = _series_on_grid(nqs_result, "energy_t", target_times)

    abs_err_mx = np.where(np.isfinite(ed_mx) & np.isfinite(nqs_mx), np.abs(ed_mx - nqs_mx), np.nan)
    abs_err_czz = np.where(np.isfinite(ed_czz) & np.isfinite(nqs_czz), np.abs(ed_czz - nqs_czz), np.nan)
    abs_err_energy = np.where(
        np.isfinite(ed_energy) & np.isfinite(nqs_energy),
        np.abs(ed_energy - nqs_energy),
        np.nan,
    )

    if np.isfinite(nqs_energy).any():
        first_idx = np.where(np.isfinite(nqs_energy))[0][0]
        e0 = nqs_energy[first_idx]
        if np.abs(e0) > 0:
            rel_energy_drift = np.where(np.isfinite(nqs_energy), np.abs(nqs_energy - e0) / np.abs(e0), np.nan)
        else:
            rel_energy_drift = np.where(np.isfinite(nqs_energy), np.abs(nqs_energy - e0), np.nan)
    else:
        rel_energy_drift = np.full(len(target_times), np.nan)

    data = np.column_stack(
        [
            target_times,
            ed_mz,
            ed_mx,
            ed_czz,
            ed_energy,
            nqs_mz,
            nqs_mx,
            nqs_czz,
            nqs_energy,
            abs_err_mx,
            abs_err_czz,
            abs_err_energy,
            rel_energy_drift,
        ]
    )
    header = ",".join(
        [
            "time",
            "ed_mz",
            "ed_mx",
            "ed_czz",
            "ed_energy",
            "nqs_mz",
            "nqs_mx",
            "nqs_czz",
            "nqs_energy",
            "abs_err_mx",
            "abs_err_czz",
            "abs_err_energy",
            "rel_energy_drift",
        ]
    )

    if csv_path is None:
        csv_path = make_output_stem(N, h_i, h_f) + ".csv"

    np.savetxt(csv_path, data, delimiter=",", header=header, comments="", fmt="%.10f")
    return csv_path


def _make_sr(nk, diag_shift):
    """Create SR with explicit holomorphic=False when supported."""
    try:
        return nk.optimizer.SR(diag_shift=diag_shift, holomorphic=False)
    except TypeError:
        sr = nk.optimizer.SR(diag_shift=diag_shift)
        if hasattr(sr, "holomorphic"):
            try:
                sr.holomorphic = False
            except Exception:
                pass
        return sr


def _make_qgt(nk, diag_shift):
    """Create a QGT factory with explicit holomorphic=False when supported."""
    qgt_ctor = nk.optimizer.qgt.QGTJacobianDense
    try:
        return qgt_ctor(diag_shift=diag_shift, holomorphic=False)
    except TypeError:
        try:
            return qgt_ctor(holomorphic=False, diag_shift=diag_shift)
        except TypeError:
            return qgt_ctor(diag_shift=diag_shift)


def _translation_permutations(N):
    """Return cyclic translations as a Flax-hashable tuple-of-tuples."""
    sites = np.arange(N, dtype=int)
    perms = np.stack([np.roll(sites, -shift) for shift in range(N)], axis=0)
    # Avoid ndarray attributes inside Flax modules: some builds fail hashing them.
    return tuple(tuple(int(x) for x in row) for row in perms)


def _make_model(nk, N, alpha, ansatz):
    """Build an RBM-family ansatz with optional translation and explicit Z2 symmetry."""
    requested = str(ansatz).lower()
    valid = {"auto", "rbm", "translation", "z2", "translation_z2"}
    if requested not in valid:
        raise ValueError(
            f"Unsupported ansatz '{ansatz}'. Choose auto|rbm|translation|z2|translation_z2."
        )

    def _plain_rbm_model():
        return nk.models.RBM(alpha=alpha, param_dtype=complex)

    def _plain_rbm(reason=None):
        if reason:
            print(f"  Warning: {reason}")
        if requested == "rbm":
            print("  Ansatz: plain RBM")
        else:
            print("  Ansatz fallback: plain RBM")
        return _plain_rbm_model()

    def _strict_translation_error(reason):
        raise RuntimeError(
            "--ansatz translation and --ansatz translation_z2 require a working "
            "translation-symmetric nk.models.RBMSymm. "
            f"{reason}"
        )

    def _translation_base(strict=False):
        rbm_symm = getattr(nk.models, "RBMSymm", None)
        if rbm_symm is None:
            if strict:
                _strict_translation_error("This NetKet version does not provide RBMSymm.")
            return None, "NetKet has no nk.models.RBMSymm in this version."

        perms = _translation_permutations(N)
        try:
            model = rbm_symm(symmetries=perms, alpha=alpha, param_dtype=complex)
            return model, None
        except TypeError:
            try:
                model = rbm_symm(perms, alpha=alpha, param_dtype=complex)
                return model, None
            except Exception as exc:
                if strict:
                    _strict_translation_error(f"RBMSymm constructor failed ({exc}).")
                return None, f"RBMSymm constructor failed ({exc})."
        except Exception as exc:
            if strict:
                _strict_translation_error(f"RBMSymm setup failed ({exc}).")
            return None, f"RBMSymm setup failed ({exc})."

    def _z2_wrap(base_model, label):
        if _Z2SymmetrizedModel is None:
            raise RuntimeError(
                "--ansatz z2/translation_z2 requires jax and flax so the explicit "
                "Z2-symmetrized wrapper can be constructed."
            )
        print(f"  Ansatz: {label}")
        return _Z2SymmetrizedModel(base_model=base_model)

    if requested == "rbm":
        print("  Ansatz: plain RBM")
        return _plain_rbm_model()

    if requested == "z2":
        return _z2_wrap(_plain_rbm_model(), "explicit Z2-symmetric RBM")

    if requested == "translation":
        model, _ = _translation_base(strict=True)
        print("  Ansatz: translation-symmetric RBMSymm")
        return model

    if requested == "translation_z2":
        base_model, _ = _translation_base(strict=True)
        return _z2_wrap(base_model, "explicit Z2-symmetric translation-symmetric RBMSymm")

    if requested == "auto":
        base_model, reason = _translation_base(strict=False)
        if base_model is not None:
            print("  Ansatz: translation-symmetric RBMSymm")
            return base_model
        return _plain_rbm(reason)

    raise AssertionError(f"Unhandled ansatz branch: {requested}")


def build_tfim_matrix(N, h, J=1.0, pbc=True):
    """Build full 2^N × 2^N TFIM matrix: H = -J Σ Z_i Z_{i+1} - h Σ X_i."""
    dim = 2 ** N
    H = np.zeros((dim, dim), dtype=complex)
    I2 = np.eye(2, dtype=complex)
    SX = np.array([[0, 1], [1, 0]], dtype=complex)
    SZ = np.array([[1, 0], [0, -1]], dtype=complex)

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


def build_observables_matrices(N, pbc=True):
    """Build ED operators for m_z, m_x, and nearest-neighbour C_zz."""
    I2 = np.eye(2, dtype=complex)
    SX = np.array([[0, 1], [1, 0]], dtype=complex)
    SZ = np.array([[1, 0], [0, -1]], dtype=complex)

    def kron_chain(ops):
        result = ops[0]
        for op in ops[1:]:
            result = np.kron(result, op)
        return result

    z_ops = []
    x_ops = []
    zz_ops = []

    for i in range(N):
        ops = [I2] * N
        ops[i] = SZ
        z_ops.append(kron_chain(ops))

    for i in range(N):
        ops = [I2] * N
        ops[i] = SX
        x_ops.append(kron_chain(ops))

    n_bonds = N if pbc else N - 1
    for i in range(n_bonds):
        j = (i + 1) % N
        ops = [I2] * N
        ops[i] = SZ
        ops[j] = SZ
        zz_ops.append(kron_chain(ops))

    return z_ops, x_ops, zz_ops


def ed_dynamics(N, h_i, h_f, T, dt, J=1.0):
    """Exact quench dynamics via full matrix exponentiation."""
    from scipy.linalg import expm

    print(f"  Building H(h_i={h_i}) ...")
    H_i = build_tfim_matrix(N, h_i, J)
    evals, evecs = np.linalg.eigh(H_i)
    psi0 = evecs[:, 0]
    E0_i = evals[0]
    print(f"  E0(h_i={h_i}) = {E0_i:.8f}")

    print(f"  Building H(h_f={h_f}) ...")
    H_f = build_tfim_matrix(N, h_f, J)

    z_ops, x_ops, zz_ops = build_observables_matrices(N)

    print(f"  Computing exp(-iH_f·dt) for dt={dt} ...")
    U_dt = expm(-1j * H_f * dt)

    times = np.arange(0, T + dt / 2, dt)
    mz_t = np.zeros(len(times))
    mx_t = np.zeros(len(times))
    czz_t = np.zeros(len(times))
    energy_t = np.zeros(len(times))

    psi = psi0.copy()
    for k, _ in enumerate(times):
        mz_t[k] = np.mean([np.real(psi.conj() @ Zi @ psi) for Zi in z_ops])
        mx_t[k] = np.mean([np.real(psi.conj() @ Xi @ psi) for Xi in x_ops])
        czz_t[k] = np.mean([np.real(psi.conj() @ ZZi @ psi) for ZZi in zz_ops])
        energy_t[k] = np.real(psi.conj() @ H_f @ psi)
        if k < len(times) - 1:
            psi = U_dt @ psi

    print(f"  E(t=0) = {energy_t[0]:.8f},  E(t={T}) = {energy_t[-1]:.8f}")
    print(f"  Energy drift: {abs(energy_t[-1] - energy_t[0]):.2e}")

    return {
        "times": times,
        "mz_t": mz_t,
        "mx_t": mx_t,
        "czz_t": czz_t,
        "energy_t": energy_t,
        "psi0": psi0,
        "E0_initial": E0_i,
    }


def nqs_ground_state(
    N,
    h,
    alpha=4,
    n_samples=4096,
    n_iter=100,
    lr=0.01,
    diag_shift=0.01,
    n_chains=16,
    ansatz="auto",
):
    """Find the 1D TFIM ground state at field h using VMC + SR."""
    import netket as nk
    import optax

    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message=".*HolomorphicUndeclaredWarning.*",
    )

    g = nk.graph.Chain(length=N, pbc=True)
    hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes)
    H = nk.operator.Ising(hilbert=hi, graph=g, h=h, J=-1.0)

    model = _make_model(nk, N, alpha, ansatz)
    sampler = nk.sampler.MetropolisLocal(hi, n_chains=n_chains)
    vstate = nk.vqs.MCState(sampler, model, n_samples=n_samples)

    opt = optax.sgd(learning_rate=lr)
    sr = _make_sr(nk, diag_shift)
    driver = nk.driver.VMC(H, opt, variational_state=vstate, preconditioner=sr)

    log = nk.logging.RuntimeLog()
    driver.run(n_iter=n_iter, out=log)

    try:
        final_e = float(np.real(vstate.expect(H).mean))
    except Exception:
        energy_data = log.data.get("Energy", {})
        raw = energy_data["Mean"] if isinstance(energy_data, dict) else log["Energy"]["Mean"]
        final_e = float(np.real(np.asarray(raw)[-1]))

    return vstate, H, final_e, log


def _make_nqs_observables(nk, hilbert, N):
    """Build NetKet operators for tracked observables."""
    mz_terms = [nk.operator.spin.sigmaz(hilbert, i) for i in range(N)]
    mx_terms = [nk.operator.spin.sigmax(hilbert, i) for i in range(N)]
    zz_terms = [
        nk.operator.spin.sigmaz(hilbert, i) * nk.operator.spin.sigmaz(hilbert, (i + 1) % N)
        for i in range(N)
    ]

    mz_op = mz_terms[0]
    mx_op = mx_terms[0]
    czz_op = zz_terms[0]
    for op in mz_terms[1:]:
        mz_op = mz_op + op
    for op in mx_terms[1:]:
        mx_op = mx_op + op
    for op in zz_terms[1:]:
        czz_op = czz_op + op

    return {"mz": mz_op / N, "mx": mx_op / N, "czz": czz_op / N}


def _expect_real(vstate, operator):
    """Return the real part of an expectation value as float."""
    return float(np.real(vstate.expect(operator).mean))


def _measure_nqs(vstate, H_f, obs):
    """Measure all tracked observables at the current NQS state."""
    return {
        "mz": _expect_real(vstate, obs["mz"]),
        "mx": _expect_real(vstate, obs["mx"]),
        "czz": _expect_real(vstate, obs["czz"]),
        "energy": _expect_real(vstate, H_f),
    }


def _try_set_n_samples(vstate, n_samples):
    """Best-effort update of sampling budget for measurement-heavy evolution."""
    for attr in ("n_samples", "n_samples_per_rank"):
        if hasattr(vstate, attr):
            try:
                setattr(vstate, attr, n_samples)
            except Exception:
                pass


def nqs_tdvp_dynamics(vstate, N, h_f, T, dt, n_samples=4096, diag_shift=0.01):
    """TDVP real-time evolution after a quench to h_f."""
    import netket as nk

    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message=".*HolomorphicUndeclaredWarning.*",
    )

    g = nk.graph.Chain(length=N, pbc=True)
    hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes)
    H_f = nk.operator.Ising(hilbert=hi, graph=g, h=h_f, J=-1.0)
    obs = _make_nqs_observables(nk, hi, N)
    _try_set_n_samples(vstate, n_samples)

    try:
        import netket.experimental as nkx

        integrator = nkx.dynamics.Heun(dt=dt)
        qgt = _make_qgt(nk, diag_shift)

        try:
            te_driver = nkx.TDVP(
                operator=H_f,
                variational_state=vstate,
                ode_solver=integrator,
                qgt=qgt,
                propagation_type="real",
            )
        except TypeError:
            te_driver = nkx.TDVP(
                operator=H_f,
                variational_state=vstate,
                integrator=integrator,
                qgt=qgt,
            )

        n_steps = int(T / dt)
        times = np.zeros(n_steps + 1)
        mz_t = np.zeros(n_steps + 1)
        mx_t = np.zeros(n_steps + 1)
        czz_t = np.zeros(n_steps + 1)
        energy_t = np.zeros(n_steps + 1)

        snap = _measure_nqs(vstate, H_f, obs)
        mz_t[0] = snap["mz"]
        mx_t[0] = snap["mx"]
        czz_t[0] = snap["czz"]
        energy_t[0] = snap["energy"]

        print(f"  Running TDVP ({n_steps} steps, dt={dt}) ...")
        for k in range(n_steps):
            te_driver.advance(dt)
            times[k + 1] = (k + 1) * dt

            snap = _measure_nqs(vstate, H_f, obs)
            mz_t[k + 1] = snap["mz"]
            mx_t[k + 1] = snap["mx"]
            czz_t[k + 1] = snap["czz"]
            energy_t[k + 1] = snap["energy"]

            if (k + 1) % max(1, n_steps // 10) == 0:
                print(
                    f"    t = {times[k + 1]:.3f}  "
                    f"m_x = {mx_t[k + 1]:.6f}  "
                    f"c_zz = {czz_t[k + 1]:.6f}  "
                    f"E = {energy_t[k + 1]:.6f}"
                )

        return {
            "times": times,
            "mz_t": mz_t,
            "mx_t": mx_t,
            "czz_t": czz_t,
            "energy_t": energy_t,
        }

    except (ImportError, AttributeError, TypeError) as exc:
        print(f"\n  NetKet experimental TDVP not available: {exc}")
        print("  Falling back to manual Euler TDVP...\n")
        return _manual_euler_tdvp(vstate, H_f, obs, T, dt, diag_shift)


def _manual_euler_tdvp(vstate, H_f, obs, T, dt, diag_shift):
    """Manual Euler TDVP fallback for older NetKet setups."""
    import jax
    import netket as nk

    qgt_factory = _make_qgt(nk, diag_shift)

    n_steps = int(T / dt)
    times = np.zeros(n_steps + 1)
    mz_t = np.zeros(n_steps + 1)
    mx_t = np.zeros(n_steps + 1)
    czz_t = np.zeros(n_steps + 1)
    energy_t = np.zeros(n_steps + 1)

    snap = _measure_nqs(vstate, H_f, obs)
    mz_t[0] = snap["mz"]
    mx_t[0] = snap["mx"]
    czz_t[0] = snap["czz"]
    energy_t[0] = snap["energy"]

    print(f"  Running manual Euler TDVP ({n_steps} steps, dt={dt}) ...")

    pinv_solver = getattr(getattr(nk.optimizer, "solver", object()), "pinv_smooth", None)

    for k in range(n_steps):
        _, grad = vstate.expect_and_grad(H_f)
        force = jax.tree.map(lambda g: -1j * g, grad)
        S = qgt_factory(vstate)

        if pinv_solver is not None:
            theta_dot = S.solve(pinv_solver, force)
        else:
            theta_dot = S.solve(force)
        if isinstance(theta_dot, tuple):
            theta_dot = theta_dot[0]

        new_params = jax.tree.map(lambda p, dp: p + dt * dp, vstate.parameters, theta_dot)
        vstate.parameters = new_params

        times[k + 1] = (k + 1) * dt
        snap = _measure_nqs(vstate, H_f, obs)
        mz_t[k + 1] = snap["mz"]
        mx_t[k + 1] = snap["mx"]
        czz_t[k + 1] = snap["czz"]
        energy_t[k + 1] = snap["energy"]

        if (k + 1) % max(1, n_steps // 10) == 0:
            print(
                f"    t = {times[k + 1]:.3f}  "
                f"m_x = {mx_t[k + 1]:.6f}  "
                f"c_zz = {czz_t[k + 1]:.6f}  "
                f"E = {energy_t[k + 1]:.6f}"
            )

    return {
        "times": times,
        "mz_t": mz_t,
        "mx_t": mx_t,
        "czz_t": czz_t,
        "energy_t": energy_t,
    }


def _print_initial_snapshot(vstate, N, h_f, ed_result=None):
    """Print the NQS observable values at t=0 before TDVP starts."""
    import netket as nk

    g = nk.graph.Chain(length=N, pbc=True)
    hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes)
    H_f = nk.operator.Ising(hilbert=hi, graph=g, h=h_f, J=-1.0)
    obs = _make_nqs_observables(nk, hi, N)
    snap = _measure_nqs(vstate, H_f, obs)

    print("  Initial-state diagnostics before TDVP:")
    print(f"    NQS m_x(0)  = {snap['mx']:.6f}")
    print(f"    NQS c_zz(0) = {snap['czz']:.6f}")
    print(f"    NQS m_z(0)  = {snap['mz']:.6f} (diagnostic)")
    print(f"    NQS <H_f>(0)= {snap['energy']:.6f}")

    if ed_result is not None:
        print(f"    ED  m_x(0)  = {ed_result['mx_t'][0]:.6f}")
        print(f"    ED  c_zz(0) = {ed_result['czz_t'][0]:.6f}")
        print(f"    ED  m_z(0)  = {ed_result['mz_t'][0]:.6f} (diagnostic)")
        print(f"    ED  <H_f>(0)= {ed_result['energy_t'][0]:.6f}")
        print(f"    |Δm_x(0)|   = {abs(snap['mx'] - ed_result['mx_t'][0]):.2e}")
        print(f"    |Δc_zz(0)|  = {abs(snap['czz'] - ed_result['czz_t'][0]):.2e}")
        print(f"    |ΔE(0)|     = {abs(snap['energy'] - ed_result['energy_t'][0]):.2e}")

    if abs(snap["mz"]) > 0.05:
        print("    Warning: |m_z(0)| is noticeably nonzero; the ansatz may be symmetry biased.")
    print()


def _print_series_summary(label, ed_series, nqs_series):
    """Print max/mean/final errors plus relative and early-vs-late MAE."""
    ed_series = np.asarray(ed_series, dtype=float)
    nqs_series = np.asarray(nqs_series, dtype=float)
    common_len = min(len(ed_series), len(nqs_series))
    ed_series = ed_series[:common_len]
    nqs_series = nqs_series[:common_len]

    abs_err = np.abs(ed_series - nqs_series)
    rel_mae = _safe_rel_mae(ed_series, abs_err)
    early_mae, late_mae = _split_halves_mae(abs_err)

    print(f"  {label}: max={np.max(abs_err):.4e}  mean={np.mean(abs_err):.4e}  final={abs_err[-1]:.4e}")
    if np.isfinite(rel_mae):
        print(f"    relative MAE = {rel_mae:.2%}")
    print(f"    early-half MAE = {early_mae:.4e}  late-half MAE = {late_mae:.4e}")


def make_figure(ed_result, nqs_result, N, h_i, h_f):
    """Make a four-panel validation figure."""
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))

    ax = axes[0]
    if ed_result is not None:
        ax.plot(ed_result["times"], ed_result["mx_t"], "k-", lw=2, label="ED exact", alpha=0.7)
    if nqs_result is not None:
        ax.plot(nqs_result["times"], nqs_result["mx_t"], "o-", ms=3, lw=1.2, label="NQS TDVP")
    ax.set_xlabel("Time $t$")
    ax.set_ylabel(r"$m_x(t) = \frac{1}{N}\sum_i \langle X_i \rangle$")
    ax.set_title(f"Transverse magnetization: $h={h_i}\\to{h_f}$, N={N}")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    if ed_result is not None:
        ax.plot(ed_result["times"], ed_result["czz_t"], "k-", lw=2, label="ED exact", alpha=0.7)
    if nqs_result is not None:
        ax.plot(nqs_result["times"], nqs_result["czz_t"], "o-", ms=3, lw=1.2, label="NQS TDVP")
    ax.set_xlabel("Time $t$")
    ax.set_ylabel(r"$c_{zz}(t) = \frac{1}{N}\sum_i \langle Z_i Z_{i+1} \rangle$")
    ax.set_title("Nearest-neighbor correlator")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    if ed_result is not None:
        ax.plot(ed_result["times"], ed_result["energy_t"], "k-", lw=2, label="ED (exact conservation)", alpha=0.7)
    if nqs_result is not None:
        ax.plot(nqs_result["times"], nqs_result["energy_t"], "o-", ms=3, lw=1.2, label="NQS TDVP")
    ax.set_xlabel("Time $t$")
    ax.set_ylabel(r"$\langle H_f \rangle_t$")
    ax.set_title("Energy conservation")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    ax = axes[3]
    if nqs_result is not None:
        e0 = nqs_result["energy_t"][0]
        if abs(e0) > 0:
            drift = np.abs(nqs_result["energy_t"] - e0) / abs(e0)
            ylabel = r"$|\langle H_f \rangle_t - \langle H_f \rangle_0| / |\langle H_f \rangle_0|$"
        else:
            drift = np.abs(nqs_result["energy_t"] - e0)
            ylabel = r"$|\langle H_f \rangle_t - \langle H_f \rangle_0|$"
        ax.semilogy(nqs_result["times"], drift + 1e-16, "o-", ms=3, lw=1.2)
        ax.set_ylabel(ylabel)
        ax.set_title("TDVP energy drift")
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No NQS data", transform=ax.transAxes, ha="center", va="center", fontsize=14)
    ax.set_xlabel("Time $t$")

    plt.tight_layout()
    fname = make_output_stem(N, h_i, h_f)
    plt.savefig(fname + ".png", dpi=150, bbox_inches="tight")
    plt.savefig(fname + ".pdf", bbox_inches="tight")
    print(f"\nSaved: {fname}.png / .pdf")


def main():
    parser = argparse.ArgumentParser(description="TFIM quench dynamics: ED vs NQS TDVP")
    parser.add_argument("--N", type=int, default=8, help="Chain length (default: 8)")
    parser.add_argument("--hi", type=float, default=0.5, help="Initial transverse field")
    parser.add_argument("--hf", type=float, default=2.0, help="Final transverse field")
    parser.add_argument("--T", type=float, default=5.0, help="Total evolution time")
    parser.add_argument("--dt", type=float, default=0.05, help="TDVP time step")
    parser.add_argument("--alpha", type=int, default=4, help="RBM hidden density")
    parser.add_argument(
        "--ansatz",
        type=str,
        default="auto",
        choices=["auto", "rbm", "translation", "z2", "translation_z2"],
        help="Ansatz type (auto prefers translation if available; z2 variants enforce global spin-flip symmetry)",
    )
    parser.add_argument("--n-samples", type=int, default=4096, help="Monte Carlo samples")
    parser.add_argument("--n-iter", type=int, default=100, help="VMC iterations")
    parser.add_argument("--diag-shift", type=float, default=0.01, help="QGT/SR diagonal shift")
    parser.add_argument("--n-chains", type=int, default=16, help="Metropolis chains")
    parser.add_argument("--ed-only", action="store_true", help="Only run ED dynamics")
    parser.add_argument("--csv-path", type=str, default=None, help="Optional output CSV path")
    args = parser.parse_args()

    print(f"TFIM Quench Dynamics: N={args.N}, h = {args.hi} → {args.hf}")
    print(f"Evolution time T={args.T}, dt={args.dt}")
    print("1D chain, PBC")
    print()

    ed_result = None
    if args.N <= 14:
        print("=" * 50)
        print("PHASE 1: Exact diagonalization dynamics")
        print("=" * 50)
        t0 = time.time()
        ed_result = ed_dynamics(args.N, args.hi, args.hf, args.T, args.dt)
        print(f"  ED time: {time.time() - t0:.1f}s")
        print(f"  m_x(0) = {ed_result['mx_t'][0]:.6f}")
        print(f"  m_x(T) = {ed_result['mx_t'][-1]:.6f}")
        print(f"  c_zz(0) = {ed_result['czz_t'][0]:.6f}")
        print(f"  c_zz(T) = {ed_result['czz_t'][-1]:.6f}")
        print(f"  m_z(0) = {ed_result['mz_t'][0]:.6f} (diagnostic)")
        print(f"  m_z(T) = {ed_result['mz_t'][-1]:.6f} (diagnostic)")
        print()
    else:
        print(f"N={args.N} > 14: skipping ED (2^{args.N} = {2 ** args.N:,} states)")
        print()

    nqs_result = None
    if not args.ed_only:
        print("=" * 50)
        print("PHASE 2a: NQS ground state of H(h_i) via VMC + SR")
        print("=" * 50)
        try:
            t0 = time.time()
            vstate, _, E_nqs, _ = nqs_ground_state(
                args.N,
                args.hi,
                alpha=args.alpha,
                n_samples=args.n_samples,
                n_iter=args.n_iter,
                diag_shift=args.diag_shift,
                n_chains=args.n_chains,
                ansatz=args.ansatz,
            )
            print(f"  NQS E0(h_i={args.hi}) = {E_nqs:.8f}  ({time.time() - t0:.1f}s)")
            if ed_result is not None:
                print(f"  ED  E0(h_i={args.hi}) = {ed_result['E0_initial']:.8f}")
                print(f"  Preparation error: {abs(E_nqs - ed_result['E0_initial']):.2e}")
            print()

            _print_initial_snapshot(vstate, args.N, args.hf, ed_result=ed_result)

            print("=" * 50)
            print("PHASE 2b: TDVP evolution under H(h_f)")
            print("=" * 50)
            t0 = time.time()
            nqs_result = nqs_tdvp_dynamics(
                vstate,
                args.N,
                args.hf,
                args.T,
                args.dt,
                n_samples=args.n_samples,
                diag_shift=args.diag_shift,
            )
            elapsed = time.time() - t0
            print(f"  TDVP time: {elapsed:.1f}s")
            print(f"  m_x(0) = {nqs_result['mx_t'][0]:.6f}")
            print(f"  m_x(T) = {nqs_result['mx_t'][-1]:.6f}")
            print(f"  c_zz(0) = {nqs_result['czz_t'][0]:.6f}")
            print(f"  c_zz(T) = {nqs_result['czz_t'][-1]:.6f}")
            print(f"  m_z(0) = {nqs_result['mz_t'][0]:.6f} (diagnostic)")
            print(f"  m_z(T) = {nqs_result['mz_t'][-1]:.6f} (diagnostic)")
            drift = abs(nqs_result["energy_t"][-1] - nqs_result["energy_t"][0])
            print(f"  Energy drift: {drift:.2e}")
            print()
        except Exception as exc:
            print(f"  NQS TDVP failed: {exc}")
            print("  Check that netket is installed and compatible.")
            print()

    print("=" * 50)
    print("SUMMARY")
    print("=" * 50)
    if ed_result is not None and nqs_result is not None:
        _print_series_summary("m_x", ed_result["mx_t"], nqs_result["mx_t"])
        _print_series_summary("c_zz", ed_result["czz_t"], nqs_result["czz_t"])
        _print_series_summary("Energy", ed_result["energy_t"], nqs_result["energy_t"])

        common_len = min(len(ed_result["mz_t"]), len(nqs_result["mz_t"]))
        mz_diff = np.abs(ed_result["mz_t"][:common_len] - nqs_result["mz_t"][:common_len])
        print(f"  m_z diagnostic: max={np.max(mz_diff):.4e}  mean={np.mean(mz_diff):.4e}")
    elif ed_result is not None:
        print("  ED dynamics completed (NQS skipped)")
    elif nqs_result is not None:
        print("  NQS TDVP completed (no ED reference)")
    else:
        print("  No results - check dependencies")
    print()

    if ed_result is not None or nqs_result is not None:
        csv_path = save_results_csv(ed_result, nqs_result, args.N, args.hi, args.hf, csv_path=args.csv_path)
        if csv_path is not None:
            print(f"Saved: {csv_path}")
        make_figure(ed_result, nqs_result, args.N, args.hi, args.hf)


if __name__ == "__main__":
    main()
