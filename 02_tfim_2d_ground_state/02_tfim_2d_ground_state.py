"""
02: 2D TFIM Ground State with NQS + TN baselines (MPS proxy + TeNPy DMRG)
============================================================================

What this script provides:
  1. Exact diagonalization (ED) on small systems (N <= 20)
  2. RBM + ARNN neural quantum states via NetKet VMC + SR
  3. A lightweight exact-state-compression MPS proxy (small systems only)
  4. A true finite-DMRG baseline via TeNPy on a snake-mapped 2D lattice
  5. Comparison against either ED (when available) or the best-found energy

Typical small controlled benchmark:
    python 02_tfim_2d_ground_state_mps_baseline.py \
      --Lx 4 --Ly 4 --h 3.0 --alpha 2 --n-samples 1024 --n-iter 200 \
      --budget-mode params --with-dmrg --dmrg-chi-sweep 4 6 8 12 16

Recommended 6x6 near-critical benchmark with best-found reference:
    python 02_tfim_2d_ground_state_mps_baseline.py \
      --Lx 6 --Ly 6 --h 3.044 --alpha 3 --n-samples 512 --n-iter 300 \
      --budget-mode params --skip-mps --with-dmrg \
      --dmrg-chi-sweep 4 6 8 10 12 14 --obc --reference-mode best \
      --param-scaling-plot
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import pathlib
import sys
import time
import warnings
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np


# ═══════════════════════════════════════════════════════════════
# PROBLEM SETUP
# ═══════════════════════════════════════════════════════════════


def square_lattice_graph(Lx, Ly, pbc=True):
    """Construct an Lx×Ly square grid graph (works across NetKet versions)."""
    import netket as nk

    return nk.graph.Grid(extent=[Lx, Ly], pbc=pbc)



def make_problem(Lx, Ly, h, pbc=True, J_netket=-1.0):
    """Return (graph, hilbert, Hamiltonian) for the 2D TFIM."""
    import netket as nk

    g = square_lattice_graph(Lx, Ly, pbc=pbc)
    hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes)
    H = nk.operator.Ising(hilbert=hi, graph=g, h=h, J=J_netket)
    return g, hi, H


# ═══════════════════════════════════════════════════════════════
# EXACT DIAGONALIZATION REFERENCE (small lattices)
# ═══════════════════════════════════════════════════════════════


def ed_ground_state(Lx, Ly, h, pbc=True, J_netket=-1.0, return_state=False):
    """
    Lanczos ED for 2D TFIM on Lx×Ly square lattice.

    Returns:
        E0                       if return_state=False
        (E0, psi0_normalized)    if return_state=True
        None                     if N > 20
    """
    import netket as nk

    N = Lx * Ly
    if N > 20:
        return None

    _, _, H = make_problem(Lx, Ly, h, pbc=pbc, J_netket=J_netket)

    if return_state:
        evals, evecs = nk.exact.lanczos_ed(H, k=1, compute_eigenvectors=True)
        E0 = np.asarray(evals).reshape(-1)[0]
        psi0 = np.asarray(evecs)
        if psi0.ndim == 2:
            psi0 = psi0[:, 0]
        psi0 = psi0.reshape(-1).astype(np.complex128)
        norm = np.linalg.norm(psi0)
        if norm == 0:
            raise RuntimeError("ED returned a zero-norm eigenvector.")
        psi0 /= norm
        return float(np.real(E0)), psi0

    E0 = nk.exact.lanczos_ed(H, k=1, compute_eigenvectors=False)[0]
    return float(np.real(E0))


# ═══════════════════════════════════════════════════════════════
# LIGHTWEIGHT MPS BASELINE (TT-SVD compression of exact state)
# ═══════════════════════════════════════════════════════════════


def compress_state_to_mps(psi, N, max_bond_dim):
    """
    Convert a 2^N statevector into an open-boundary MPS via sequential SVD.

    Returns:
        cores: list of tensors with shapes (r_left, 2, r_right)
        discarded_weight: total discarded singular-value weight
    """
    psi = np.asarray(psi, dtype=np.complex128).reshape([2] * N)
    cores = []
    discarded_weight = 0.0

    left_rank = 1
    work = psi

    for _site in range(N - 1):
        mat = work.reshape(left_rank * 2, -1)
        U, S, Vh = np.linalg.svd(mat, full_matrices=False)
        keep = min(max_bond_dim, S.size)

        if keep < S.size:
            discarded_weight += float(np.sum(np.abs(S[keep:]) ** 2))

        U = U[:, :keep]
        S = S[:keep]
        Vh = Vh[:keep, :]

        cores.append(U.reshape(left_rank, 2, keep))
        work = (S[:, None] * Vh)
        left_rank = keep

    cores.append(work.reshape(left_rank, 2, 1))
    return cores, discarded_weight



def mps_to_state(cores):
    """Reconstruct a dense statevector from MPS cores (small-N only)."""
    psi = cores[0]
    for core in cores[1:]:
        psi = np.tensordot(psi, core, axes=([-1], [0]))
    return np.asarray(psi).reshape(-1)



def mps_parameter_count(cores):
    """Count free scalar entries in the MPS tensors."""
    return int(sum(int(np.prod(c.shape)) for c in cores))



def choose_mps_bond_dim_for_budget(psi_exact, N, target_params, chi_max):
    """Pick the MPS bond dimension whose parameter count is closest to target."""
    best = None
    best_score = None

    for chi in range(1, max(1, chi_max) + 1):
        cores, discarded = compress_state_to_mps(psi_exact, N, chi)
        params = mps_parameter_count(cores)
        score = (params > target_params, abs(params - target_params), discarded)
        if best_score is None or score < best_score:
            best = (chi, params)
            best_score = score

    return best



def run_mps_baseline_from_exact_state(Lx, Ly, h, psi_exact, max_bond_dim, pbc=True):
    """Build a finite-MPS approximation by compressing the exact ED ground state."""
    N = Lx * Ly
    _, _, H = make_problem(Lx, Ly, h, pbc=pbc, J_netket=-1.0)

    t0 = time.time()
    cores, discarded_weight = compress_state_to_mps(psi_exact, N, max_bond_dim)
    psi_mps = mps_to_state(cores).astype(np.complex128)
    norm = np.linalg.norm(psi_mps)
    if norm == 0:
        raise RuntimeError("MPS reconstruction produced a zero-norm state.")
    psi_mps /= norm

    H_lin = None
    for attr in ("to_sparse", "to_linear_operator"):
        if hasattr(H, attr):
            H_lin = getattr(H, attr)()
            break
    if H_lin is None:
        raise RuntimeError("Could not convert Hamiltonian to a linear operator.")

    Hpsi = H_lin @ psi_mps
    energy = float(np.real(np.vdot(psi_mps, Hpsi)))
    overlap = float(np.abs(np.vdot(psi_exact, psi_mps)))
    elapsed = time.time() - t0

    return {
        "ansatz": "MPS",
        "bond_dim": int(max_bond_dim),
        "energy": energy,
        "std": 0.0,
        "n_params": mps_parameter_count(cores),
        "trace": None,
        "elapsed_s": elapsed,
        "discarded_weight": float(discarded_weight),
        "fidelity": overlap ** 2,
    }


# ═══════════════════════════════════════════════════════════════
# NQS GROUND STATE: RBM
# ═══════════════════════════════════════════════════════════════


def run_rbm(Lx, Ly, h, alpha=2, n_samples=4096, n_iter=600,
            lr=0.01, diag_shift=0.01, pbc=True):
    """VMC + SR with a complex RBM ansatz on the 2D TFIM."""
    import netket as nk
    import optax

    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message=".*HolomorphicUndeclaredWarning.*",
    )

    _, hi, H = make_problem(Lx, Ly, h, pbc=pbc, J_netket=-1.0)

    model = nk.models.RBM(alpha=alpha, param_dtype=complex)
    sampler = nk.sampler.MetropolisLocal(hi, n_chains=16)
    vstate = nk.vqs.MCState(sampler, model, n_samples=n_samples)

    opt = optax.sgd(learning_rate=lr)
    sr = nk.optimizer.SR(diag_shift=diag_shift, holomorphic=False)
    driver = nk.driver.VMC(H, opt, variational_state=vstate, preconditioner=sr)

    log = nk.logging.RuntimeLog()
    t0 = time.time()
    driver.run(n_iter=n_iter, out=log)
    elapsed = time.time() - t0

    energy_data = log.data.get("Energy", {})
    raw_trace = energy_data["Mean"] if isinstance(energy_data, dict) and "Mean" in energy_data else log["Energy"]["Mean"]
    energy_trace = np.real(np.asarray(raw_trace))

    try:
        E_stats = vstate.expect(H)
        final_e = float(np.real(E_stats.mean))
        final_std = float(np.real(E_stats.error_of_mean))
    except Exception:
        final_e = float(energy_trace[-1])
        final_std = float("nan")

    return {
        "ansatz": "RBM",
        "alpha": int(alpha),
        "energy": final_e,
        "std": final_std,
        "n_params": int(vstate.n_parameters),
        "trace": energy_trace,
        "elapsed_s": elapsed,
    }


# ═══════════════════════════════════════════════════════════════
# NQS GROUND STATE: AUTOREGRESSIVE (ARNN / TRANSFORMER)
# ═══════════════════════════════════════════════════════════════


def _load_transformer_builder():
    """
    Resolve build_transformer_model from a local transformer module.

    Supports:
      - transformer_ansatz.py
      - 02.1_transformer_ansatz.py  (non-importable name via normal import)
      - 02_1_transformer_ansatz.py
    """
    try:
        from transformer_ansatz import build_transformer_model
        return build_transformer_model
    except Exception as exc:
        last_exc = exc

    base_dir = pathlib.Path(__file__).resolve().parent
    candidates = [
        base_dir / "transformer_ansatz.py",
        base_dir / "02.1_transformer_ansatz.py",
        base_dir / "02_1_transformer_ansatz.py",
    ]

    for path in candidates:
        if not path.exists():
            continue
        module_name = f"_local_transformer_ansatz_{path.stem.replace('.', '_')}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        # Flax dataclass machinery expects the module to be present in sys.modules.
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        builder = getattr(module, "build_transformer_model", None)
        if callable(builder):
            return builder

    searched = ", ".join(str(p.name) for p in candidates)
    raise ModuleNotFoundError(
        f"Could not load transformer ansatz module. Searched: {searched}"
    ) from last_exc


def _build_autoregressive_model(hi, *, ar_model="dense", features=32, layers=1,
                                transformer_d_model=None, transformer_heads=4,
                                transformer_ff_mult=4):
    """Return (model, ansatz_label, metadata) for the requested autoregressive ansatz."""
    import netket as nk

    ar_model = (ar_model or "dense").lower()
    if ar_model == "transformer":
        build_transformer_model = _load_transformer_builder()

        d_model = int(transformer_d_model if transformer_d_model is not None else max(64, features))
        model = build_transformer_model(
            hi,
            d_model=d_model,
            n_layers=layers,
            n_heads=transformer_heads,
            ff_mult=transformer_ff_mult,
            param_dtype=float,
        )
        return model, "TransformerAR", {
            "ar_model": "transformer",
            "features": int(d_model),
            "d_model": int(d_model),
            "heads": int(transformer_heads),
            "ff_mult": int(transformer_ff_mult),
        }

    width = int(features)
    model = nk.models.ARNNDense(
        hilbert=hi,
        layers=layers,
        features=width,
        param_dtype=float,
    )
    return model, "ARNN", {
        "ar_model": "dense",
        "features": width,
        "d_model": "",
        "heads": "",
        "ff_mult": "",
    }



def _make_autoregressive_sampler(hi, ansatz_label):
    """Prefer exact direct sampling; fall back gracefully for custom models."""
    import netket as nk

    if ansatz_label == "ARNN":
        return nk.sampler.ARDirectSampler(hi), "ARDirectSampler"
    if ansatz_label == "TransformerAR":
        # Custom transformer module is not a full NetKet ARNN subclass in all builds.
        # MetropolisLocal is robust across versions and avoids ARDirect reorder hooks.
        return nk.sampler.MetropolisLocal(hi, n_chains=16), "MetropolisLocal"

    try:
        return nk.sampler.ARDirectSampler(hi), "ARDirectSampler"
    except Exception as exc:
        warnings.warn(
            f"ARDirectSampler rejected the custom transformer ansatz ({exc}); "
            "falling back to MetropolisLocal.",
            RuntimeWarning,
        )
        return nk.sampler.MetropolisLocal(hi, n_chains=16), "MetropolisLocal"



def estimate_arnn_n_parameters(Lx, Ly, features, layers, pbc=True, *,
                               ar_model="dense", transformer_d_model=None,
                               transformer_heads=4, transformer_ff_mult=4):
    """Instantiate a tiny autoregressive state to estimate parameter count."""
    import netket as nk

    _, hi, _ = make_problem(Lx, Ly, h=0.0, pbc=pbc, J_netket=-1.0)
    model, ansatz_label, _ = _build_autoregressive_model(
        hi,
        ar_model=ar_model,
        features=features,
        layers=layers,
        transformer_d_model=transformer_d_model,
        transformer_heads=transformer_heads,
        transformer_ff_mult=transformer_ff_mult,
    )
    sampler, _ = _make_autoregressive_sampler(hi, ansatz_label)
    probe_samples = 16 if ansatz_label == "ARNN" else 64
    vstate = nk.vqs.MCState(sampler, model, n_samples=probe_samples)
    return int(vstate.n_parameters)



def choose_arnn_for_budget(Lx, Ly, target_params, pbc=True,
                           layers_override=None, features_override=None, *,
                           ar_model="dense", transformer_d_model_override=None,
                           transformer_heads=4, transformer_ff_mult=4):
    """Choose a modest autoregressive configuration near a target parameter count."""
    N = Lx * Ly

    if transformer_d_model_override is not None:
        layers = 1 if layers_override is None else layers_override
        params = estimate_arnn_n_parameters(
            Lx, Ly,
            features=max(8, N),
            layers=layers,
            pbc=pbc,
            ar_model=ar_model,
            transformer_d_model=transformer_d_model_override,
            transformer_heads=transformer_heads,
            transformer_ff_mult=transformer_ff_mult,
        )
        return transformer_d_model_override, layers, params

    if features_override is not None:
        layers = 1 if layers_override is None else layers_override
        params = estimate_arnn_n_parameters(
            Lx, Ly, features_override, layers, pbc=pbc,
            ar_model=ar_model,
            transformer_d_model=None,
            transformer_heads=transformer_heads,
            transformer_ff_mult=transformer_ff_mult,
        )
        return features_override, layers, params

    if (ar_model or "dense").lower() == "transformer":
        feature_candidates = sorted({
            max(32, N),
            max(48, int(1.5 * N)),
            max(64, 2 * N),
            max(96, 3 * N),
            max(128, 4 * N),
        })
    else:
        feature_candidates = sorted({
            max(4, N // 8),
            max(4, N // 4),
            max(8, N // 2),
            max(8, 3 * N // 4),
            N,
            int(1.5 * N),
            2 * N,
        })
    layer_candidates = [1] if layers_override is None else [layers_override]

    best = None
    best_score = None
    for layers in layer_candidates:
        for features in feature_candidates:
            try:
                params = estimate_arnn_n_parameters(
                    Lx, Ly, features, layers, pbc=pbc,
                    ar_model=ar_model,
                    transformer_d_model=(features if (ar_model or "dense").lower() == "transformer" else None),
                    transformer_heads=transformer_heads,
                    transformer_ff_mult=transformer_ff_mult,
                )
            except Exception:
                continue
            score = (params > target_params, abs(params - target_params), params)
            if best_score is None or score < best_score:
                best = (features, layers, params)
                best_score = score

    return best



def run_arnn(Lx, Ly, h, n_samples=4096, n_iter=600,
             lr=0.01, diag_shift=0.01,
             arnn_features=None, arnn_layers=1, pbc=True, *,
             ar_model="dense", transformer_d_model=None,
             transformer_heads=4, transformer_ff_mult=4):
    """VMC + SR ground-state search with an autoregressive NQS on 2D TFIM."""
    import netket as nk
    import optax

    N = Lx * Ly
    _, hi, H = make_problem(Lx, Ly, h, pbc=pbc, J_netket=-1.0)

    features = arnn_features if arnn_features is not None else 2 * N
    model, ansatz_label, meta = _build_autoregressive_model(
        hi,
        ar_model=ar_model,
        features=features,
        layers=arnn_layers,
        transformer_d_model=transformer_d_model,
        transformer_heads=transformer_heads,
        transformer_ff_mult=transformer_ff_mult,
    )
    sampler, sampler_name = _make_autoregressive_sampler(hi, ansatz_label)
    vstate = nk.vqs.MCState(sampler, model, n_samples=n_samples)

    opt = optax.sgd(learning_rate=lr)
    sr = nk.optimizer.SR(diag_shift=diag_shift, holomorphic=False)
    driver = nk.driver.VMC(H, opt, variational_state=vstate, preconditioner=sr)

    log = nk.logging.RuntimeLog()
    t0 = time.time()
    driver.run(n_iter=n_iter, out=log)
    elapsed = time.time() - t0

    energy_data = log.data.get("Energy", {})
    raw_trace = energy_data["Mean"] if isinstance(energy_data, dict) and "Mean" in energy_data else log["Energy"]["Mean"]
    energy_trace = np.real(np.asarray(raw_trace))

    try:
        E_stats = vstate.expect(H)
        final_e = float(np.real(E_stats.mean))
        final_std = float(np.real(E_stats.error_of_mean))
    except Exception:
        final_e = float(energy_trace[-1])
        final_std = float("nan")

    return {
        "ansatz": ansatz_label,
        "ar_model": meta["ar_model"],
        "features": int(meta["features"]),
        "layers": int(arnn_layers),
        "d_model": meta["d_model"],
        "heads": meta["heads"],
        "ff_mult": meta["ff_mult"],
        "sampler": sampler_name,
        "energy": final_e,
        "std": final_std,
        "n_params": int(vstate.n_parameters),
        "trace": energy_trace,
        "elapsed_s": elapsed,
    }


# ═══════════════════════════════════════════════════════════════
# TRUE TN BASELINE: TeNPy DMRG on a snake-mapped 2D TFIM
# ═══════════════════════════════════════════════════════════════


def _count_finite_mps_real_params(psi_opt):
    """Gauge-corrected finite-MPS parameter count from actual converged bond dims."""
    chis = [int(c) for c in psi_opt.chi]
    n_sites = len(psi_opt.sites)

    # local dimension; robust to different TeNPy site internals
    try:
        d = int(psi_opt.sites[0].dim)
    except Exception:
        d = 2

    bonds = [1] + chis + [1]
    raw_params = 0
    gauge_params = 0
    for i in range(n_sites):
        chi_l = int(bonds[i])
        chi_r = int(bonds[i + 1])
        raw_params += chi_l * d * chi_r
        if i < n_sites - 1:
            gauge_params += chi_r * chi_r

    return max(2 * (raw_params - gauge_params), 1)



def run_dmrg_tfim(Lx, Ly, h, chi_max=16, n_sweeps=10, svd_min=1e-10, pbc=True):
    """
    Finite two-site DMRG for 2D TFIM using a Square lattice + finite MPS ordering.

    The underlying MPS ordering is whatever TeNPy uses for the Square lattice
    object; when supported, we request a snake ordering. This is the standard
    finite-MPS approach to 2D DMRG on small cylinders/rectangles.
    """
    from tenpy.algorithms import dmrg
    from tenpy.models.lattice import Square
    from tenpy.models.model import CouplingMPOModel
    from tenpy.networks.mps import MPS
    from tenpy.networks.site import SpinHalfSite

    class TFIM2D(CouplingMPOModel):
        def init_lattice(self, model_params):
            site = SpinHalfSite(conserve=None)
            bc = "periodic" if model_params.get("pbc", True) else "open"
            square_kwargs = {
                "bc": bc,
                "bc_MPS": model_params.get("bc_MPS", "finite"),
            }
            try:
                return Square(
                    model_params["Lx"],
                    model_params["Ly"],
                    site,
                    order="snake",
                    **square_kwargs,
                )
            except TypeError:
                return Square(
                    model_params["Lx"],
                    model_params["Ly"],
                    site,
                    **square_kwargs,
                )

        def init_terms(self, model_params):
            Jzz = model_params.get("Jzz", -1.0)
            hx = model_params.get("h", 3.0)
            # NetKet convention here is effectively H = J Σ zz - h Σ x with J=-1,
            # i.e. ferromagnetic -Σ zz - h Σ x.
            for u in range(len(self.lat.unit_cell)):
                self.add_onsite(-hx, u, "Sigmax")
            for u1, u2, dx in self.lat.pairs["nearest_neighbors"]:
                self.add_coupling(Jzz, u1, "Sigmaz", u2, "Sigmaz", dx)

    model = TFIM2D({
        "Lx": int(Lx),
        "Ly": int(Ly),
        "h": float(h),
        "Jzz": -1.0,
        "pbc": bool(pbc),
        "bc_MPS": "finite",
    })

    sites = model.lat.mps_sites()
    product_state = ["up"] * len(sites)
    psi = MPS.from_product_state(sites, product_state, bc="finite")

    dmrg_params = {
        "trunc_params": {
            "chi_max": int(chi_max),
            "svd_min": float(svd_min),
        },
        "max_sweeps": int(n_sweeps),
        "max_trunc_err": 1.0,
        "mixer": True,
        "mixer_params": {
            "amplitude": 1e-3,
            "decay": 1.5,
            "disable_after": max(1, int(n_sweeps) - 3),
        },
    }

    t0 = time.time()
    engine = dmrg.TwoSiteDMRGEngine(psi, model, dmrg_params)
    out = engine.run()
    elapsed = time.time() - t0

    if isinstance(out, tuple) and len(out) >= 2:
        E0 = out[0]
        psi_opt = out[1]
    else:
        E0 = out
        psi_opt = getattr(engine, "psi", psi)

    n_params = _count_finite_mps_real_params(psi_opt)

    return {
        "ansatz": "DMRG",
        "bond_dim": int(chi_max),
        "energy": float(np.real(E0)),
        "std": 0.0,
        "n_params": int(n_params),
        "trace": None,
        "elapsed_s": elapsed,
        "chis": [int(c) for c in getattr(psi_opt, "chi", [])],
    }


# ═══════════════════════════════════════════════════════════════
# COMPARISON HELPERS
# ═══════════════════════════════════════════════════════════════


def default_csv_filename(Lx, Ly, h, sweep=False):
    suffix = "chi_sweep_summary" if sweep else "summary"
    return f"tfim_2d_{Lx}x{Ly}_h{h:.1f}_{suffix}.csv"



def choose_reference_energy(results, E_exact, mode="auto"):
    """Pick the energy used for reported relative errors / Δ annotations."""
    mode = (mode or "auto").lower()
    best_found = min((float(r["energy"]) for r in results), default=None)

    if mode == "ed":
        if E_exact is not None:
            return float(E_exact), "ED exact", "ed"
        if best_found is None:
            return None, "reference unavailable", "none"
        return float(best_found), "best found", "best"

    if mode == "best":
        if best_found is None:
            if E_exact is None:
                return None, "reference unavailable", "none"
            return float(E_exact), "ED exact", "ed"
        return float(best_found), "best found", "best"

    # auto
    if E_exact is not None:
        return float(E_exact), "ED exact", "ed"
    if best_found is not None:
        return float(best_found), "best found", "best"
    return None, "reference unavailable", "none"



def rel_error_to_ref(energy, E_ref):
    if E_ref is None or not np.isfinite(E_ref) or E_ref == 0:
        return float("nan")
    return abs(float(energy) - float(E_ref)) / abs(float(E_ref))



def delta_to_ref(energy, E_ref):
    if E_ref is None or not np.isfinite(E_ref):
        return float("nan")
    return float(energy) - float(E_ref)



def write_csv_summary(path, results, E_exact, E_ref, ref_label, ref_mode, Lx, Ly, h, pbc=True):
    """Write a flat CSV summary of all baselines / ansätze."""
    fieldnames = [
        "Lx", "Ly", "N", "h", "boundary",
        "E_exact", "E_ref", "ref_label", "ref_mode",
        "ansatz", "ar_model", "alpha", "bond_dim", "features", "layers",
        "d_model", "heads", "ff_mult", "sampler",
        "energy", "std", "delta_to_ref", "rel_error_ref", "rel_error_exact",
        "n_params", "elapsed_s", "fidelity", "discarded_weight",
    ]

    boundary = "PBC" if pbc else "OBC"
    N = Lx * Ly

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        if E_exact is not None:
            writer.writerow({
                "Lx": Lx,
                "Ly": Ly,
                "N": N,
                "h": h,
                "boundary": boundary,
                "E_exact": E_exact,
                "E_ref": E_ref if E_ref is not None else "",
                "ref_label": ref_label,
                "ref_mode": ref_mode,
                "ansatz": "ED",
                "energy": E_exact,
                "delta_to_ref": delta_to_ref(E_exact, E_ref),
                "rel_error_ref": rel_error_to_ref(E_exact, E_ref),
                "rel_error_exact": 0.0,
            })

        for r in results:
            rel_err_exact = (abs(r["energy"] - E_exact) / abs(E_exact)
                             if E_exact is not None else "")
            writer.writerow({
                "Lx": Lx,
                "Ly": Ly,
                "N": N,
                "h": h,
                "boundary": boundary,
                "E_exact": E_exact if E_exact is not None else "",
                "E_ref": E_ref if E_ref is not None else "",
                "ref_label": ref_label,
                "ref_mode": ref_mode,
                "ansatz": r.get("ansatz", ""),
                "ar_model": r.get("ar_model", ""),
                "alpha": r.get("alpha", ""),
                "bond_dim": r.get("bond_dim", ""),
                "features": r.get("features", ""),
                "layers": r.get("layers", ""),
                "d_model": r.get("d_model", ""),
                "heads": r.get("heads", ""),
                "ff_mult": r.get("ff_mult", ""),
                "sampler": r.get("sampler", ""),
                "energy": r.get("energy", ""),
                "std": r.get("std", ""),
                "delta_to_ref": delta_to_ref(r.get("energy"), E_ref),
                "rel_error_ref": rel_error_to_ref(r.get("energy"), E_ref),
                "rel_error_exact": rel_err_exact,
                "n_params": r.get("n_params", ""),
                "elapsed_s": r.get("elapsed_s", ""),
                "fidelity": r.get("fidelity", ""),
                "discarded_weight": r.get("discarded_weight", ""),
            })

    print(f"CSV summary written: {path}")


# ═══════════════════════════════════════════════════════════════
# FIGURES
# ═══════════════════════════════════════════════════════════════


def _result_label(r):
    tag = r["ansatz"]
    if r["ansatz"] == "RBM":
        tag += f" α={r['alpha']}"
    elif r["ansatz"] in {"MPS", "DMRG"}:
        tag += f" χ={r['bond_dim']}"
    elif r["ansatz"] == "ARNN":
        tag += f" f={r['features']} L={r['layers']}"
    elif r["ansatz"] == "TransformerAR":
        tag += f" d={r['d_model']} H={r['heads']} L={r['layers']}"
    return tag



def make_figure(results, E_exact, E_ref, ref_label, Lx, Ly, h, pbc=True):
    """Two-panel figure: traces + final energy comparison."""
    boundary = "PBC" if pbc else "OBC"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    colors = {
        "RBM": "#2196F3",
        "ARNN": "#4CAF50",
        "TransformerAR": "#00ACC1",
        "MPS": "#FF9800",
        "DMRG": "#9C27B0",
    }

    for r in results:
        if r.get("trace") is None:
            continue
        label = _result_label(r) + f" [{r['n_params']} params]"
        ax1.plot(r["trace"], color=colors.get(r["ansatz"], "#999"), lw=1.5, label=label)

    if E_exact is not None:
        ax1.axhline(E_exact, color="k", ls="--", lw=1, alpha=0.5, label=f"ED exact = {E_exact:.6f}")
    elif E_ref is not None:
        ax1.axhline(E_ref, color="k", ls="--", lw=1, alpha=0.5, label=f"{ref_label} = {E_ref:.6f}")

    ax1.set_xlabel("VMC iteration", fontsize=12)
    ax1.set_ylabel("Energy", fontsize=12)
    ax1.set_title(f"2D TFIM {Lx}×{Ly}, h/J = {h:.2f}, {boundary}", fontsize=13)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    labels = []
    energies = []
    errors = []
    bar_colors = []

    if E_exact is not None:
        labels.append("ED exact")
        energies.append(E_exact)
        errors.append(0.0)
        bar_colors.append("#E53935")

    for r in results:
        labels.append(_result_label(r))
        energies.append(r["energy"])
        errors.append(r["std"] if not np.isnan(r["std"]) else 0.0)
        bar_colors.append(colors.get(r["ansatz"], "#999"))

    y = np.arange(len(labels))
    ax2.barh(y, energies, xerr=errors, color=bar_colors, alpha=0.8, height=0.55)
    ax2.set_yticks(y)
    ax2.set_yticklabels(labels, fontsize=10)
    ax2.set_xlabel("Ground-state energy", fontsize=12)
    ax2.set_title("Final energy comparison", fontsize=13)
    ax2.grid(True, alpha=0.3, axis="x")

    if E_ref is not None:
        start = 1 if E_exact is not None else 0
        for i, item in enumerate(results, start=start):
            rel_err = rel_error_to_ref(item["energy"], E_ref)
            ax2.annotate(f"  Δ = {rel_err:.2e}", xy=(item["energy"], i), fontsize=8, va="center")

    plt.tight_layout()
    fname = f"tfim_2d_{Lx}x{Ly}_h{h:.1f}"
    plt.savefig(fname + ".png", dpi=150, bbox_inches="tight")
    plt.savefig(fname + ".pdf", bbox_inches="tight")
    print(f"\nSaved: {fname}.png / .pdf")



def make_param_scaling_plot(results, E_ref, ref_label, Lx, Ly, h, pbc=True, out_base=None):
    """Parameter-count vs relative-error plot against the chosen reference."""
    if E_ref is None:
        print("Skipping parameter-scaling plot: no comparison reference available.")
        return

    boundary = "PBC" if pbc else "OBC"
    fig, ax = plt.subplots(figsize=(7.6, 5.4))

    colors = {
        "RBM": "#2196F3",
        "ARNN": "#4CAF50",
        "TransformerAR": "#00ACC1",
        "MPS": "#FF9800",
        "DMRG": "#9C27B0",
    }
    markers = {
        "RBM": "o",
        "ARNN": "s",
        "TransformerAR": "P",
        "MPS": "^",
        "DMRG": "D",
    }

    groups = {}
    for r in results:
        groups.setdefault(r["ansatz"], []).append(r)

    order = ["RBM", "DMRG", "ARNN", "TransformerAR", "MPS"]
    for ansatz in order:
        items = groups.get(ansatz, [])
        if not items:
            continue
        items = sorted(items, key=lambda x: x["n_params"])
        xs = [r["n_params"] for r in items]
        ys = [max(rel_error_to_ref(r["energy"], E_ref), 1e-16) for r in items]
        if len(items) > 1:
            ax.plot(xs, ys, marker=markers.get(ansatz, "o"), color=colors.get(ansatz, "#666"), label=ansatz)
        else:
            ax.scatter(xs, ys, marker=markers.get(ansatz, "o"), s=70, color=colors.get(ansatz, "#666"), label=ansatz)

    ax.set_yscale("log")
    ax.set_xlabel("Variational parameter count", fontsize=12)
    ax.set_ylabel(f"Relative energy error vs {ref_label}", fontsize=12)
    ax.set_title(f"2D TFIM {Lx}×{Ly}, h/J = {h:.2f}: accuracy vs budget", fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    plt.tight_layout()
    base = out_base or f"tfim_2d_{Lx}x{Ly}_h{h:.1f}_param_scaling"
    plt.savefig(base + ".png", dpi=150, bbox_inches="tight")
    plt.savefig(base + ".pdf", bbox_inches="tight")
    print(f"Saved: {base}.png / .pdf")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="2D TFIM ground state with NQS + MPS proxy + TeNPy DMRG baseline"
    )

    # Physics / lattice
    parser.add_argument("--Lx", type=int, default=4)
    parser.add_argument("--Ly", type=int, default=4)
    parser.add_argument("--h", type=float, default=3.0,
                        help="Transverse field (critical ≈ 3.044 in 2D)")
    parser.add_argument("--obc", action="store_true",
                        help="Use open instead of periodic boundaries")

    # RBM controls
    parser.add_argument("--alpha", type=int, default=2,
                        help="RBM hidden density")
    parser.add_argument("--n-samples", type=int, default=4096,
                        help="RBM samples per iteration")
    parser.add_argument("--n-iter", type=int, default=600,
                        help="RBM iterations")
    parser.add_argument("--rbm-only", action="store_true",
                        help="Skip ARNN (faster)")

    # ARNN controls
    parser.add_argument("--arnn-features", type=int, default=None,
                        help="ARNN hidden width; also used as transformer d_model when no explicit transformer width is given")
    parser.add_argument("--arnn-layers", type=int, default=1,
                        help="Number of autoregressive hidden layers / transformer blocks")
    parser.add_argument("--arnn-n-samples", type=int, default=None,
                        help="Autoregressive-model samples per iteration (default: same as RBM)")
    parser.add_argument("--arnn-n-iter", type=int, default=None,
                        help="Autoregressive-model iterations (default: same as RBM)")
    parser.add_argument("--ar-model", choices=["dense", "transformer"], default="dense",
                        help="Autoregressive ansatz: NetKet ARNNDense or a custom transformer-style model")
    parser.add_argument("--transformer-d-model", type=int, default=None,
                        help="Transformer embedding width (default: use --arnn-features or auto-match budget)")
    parser.add_argument("--transformer-heads", type=int, default=4,
                        help="Number of attention heads in the transformer ansatz")
    parser.add_argument("--transformer-ff-mult", type=int, default=4,
                        help="Feed-forward width multiplier inside each transformer block")

    # Lightweight MPS proxy baseline
    parser.add_argument("--skip-mps", action="store_true",
                        help="Skip the lightweight exact-state MPS proxy")
    parser.add_argument("--mps-bond", type=int, default=None,
                        help="MPS bond dimension χ (default: auto or 8)")
    parser.add_argument("--mps-chi-sweep", type=int, nargs="+", default=None,
                        help="Run the lightweight MPS proxy for each listed χ")
    parser.add_argument("--mps-chi-sweep-default", action="store_true",
                        help="Use a built-in χ sweep: 2 3 4 5 6 8 12 16 20 24 32")
    parser.add_argument("--mps-chi-max", type=int, default=16,
                        help="Max χ to consider when auto-matching parameters")

    # True DMRG baseline
    parser.add_argument("--with-dmrg", action="store_true",
                        help="Run TeNPy finite DMRG baseline")
    parser.add_argument("--dmrg-chi", type=int, default=8,
                        help="Single DMRG χ if not sweeping")
    parser.add_argument("--dmrg-chi-sweep", type=int, nargs="+", default=None,
                        help="Run DMRG for each listed χ")
    parser.add_argument("--dmrg-chi-sweep-default", action="store_true",
                        help="Use a built-in DMRG χ sweep: 4 6 8 10 12 14 16")
    parser.add_argument("--dmrg-sweeps", type=int, default=10,
                        help="DMRG max sweeps")
    parser.add_argument("--dmrg-svd-min", type=float, default=1e-10,
                        help="DMRG truncation cutoff")

    # Fairness / reporting
    parser.add_argument("--budget-mode", choices=["none", "params"], default="none",
                        help="'params' matches ARNN/MPS roughly to the RBM parameter count")
    parser.add_argument("--reference-mode", choices=["auto", "ed", "best"], default="auto",
                        help="Comparison reference: ED exact, best found, or auto (ED if available else best)")
    parser.add_argument("--csv-out", type=str, default=None,
                        help="Write a CSV summary (default auto-name when sweeping)")
    parser.add_argument("--param-scaling-plot", action="store_true",
                        help="Write a parameter-count vs relative-error plot")
    parser.add_argument("--param-plot-out", type=str, default=None,
                        help="Custom basename for the parameter-scaling plot")

    args = parser.parse_args()
    pbc = not args.obc
    boundary = "PBC" if pbc else "OBC"

    N = args.Lx * args.Ly
    print(f"2D TFIM ground state: {args.Lx}×{args.Ly} = {N} sites, h/J = {args.h:.3f}, {boundary}")
    print(f"Hilbert space dim = 2^{N} = {2 ** N:,}")
    print()

    need_exact_state = (not args.skip_mps) and (N <= 20)
    psi_exact = None

    print("Computing ED reference...")
    try:
        if need_exact_state:
            ed = ed_ground_state(args.Lx, args.Ly, args.h, pbc=pbc, return_state=True)
            if ed is None:
                E_exact = None
            else:
                E_exact, psi_exact = ed
        else:
            E_exact = ed_ground_state(args.Lx, args.Ly, args.h, pbc=pbc, return_state=False)

        if E_exact is not None:
            print(f"  E0 = {E_exact:.10f}  (E0/N = {E_exact / N:.10f})")
        else:
            print(f"  Skipped (N={N} > 20, Hilbert space too large)")
    except Exception as e:
        print(f"  Failed: {e}")
        E_exact = None
        psi_exact = None
    print()

    results = []

    print(f"Running RBM (alpha={args.alpha})...")
    try:
        rbm = run_rbm(
            args.Lx, args.Ly, args.h,
            alpha=args.alpha,
            n_samples=args.n_samples,
            n_iter=args.n_iter,
            pbc=pbc,
        )
        results.append(rbm)
        target_params = rbm["n_params"] if args.budget_mode == "params" else None
        rel_exact = rel_error_to_ref(rbm["energy"], E_exact)
        print(f"  E = {rbm['energy']:.8f} ± {rbm['std']:.2e}")
        print(f"  params = {rbm['n_params']}, time = {rbm['elapsed_s']:.1f}s")
        if E_exact is not None:
            print(f"  relative error vs ED = {rel_exact:.2e}")
    except Exception as e:
        print(f"  Failed: {e}")
        rbm = None
        target_params = None
    print()

    if not args.skip_mps:
        mps_sweep = None
        if args.mps_chi_sweep_default:
            mps_sweep = [2, 3, 4, 5, 6, 8, 12, 16, 20, 24, 32]
        elif args.mps_chi_sweep:
            mps_sweep = list(dict.fromkeys(int(x) for x in args.mps_chi_sweep))

        print("Running lightweight MPS baseline (exact-state compression)...")
        try:
            if psi_exact is None:
                if N > 20:
                    raise RuntimeError("MPS baseline requires exact state, so it is only enabled for N <= 20.")
                raise RuntimeError("Exact state unavailable, so the MPS baseline could not be built.")

            if mps_sweep is None:
                mps_bonds = [args.mps_bond] if args.mps_bond is not None else []
                if not mps_bonds:
                    if target_params is not None:
                        choice = choose_mps_bond_dim_for_budget(psi_exact, N, target_params, args.mps_chi_max)
                        mps_bonds = [choice[0]]
                        print(f"  auto-selected χ = {mps_bonds[0]} to target ~{target_params} parameters")
                    else:
                        mps_bonds = [min(8, args.mps_chi_max)]
            else:
                mps_bonds = mps_sweep

            for chi in mps_bonds:
                mps = run_mps_baseline_from_exact_state(
                    args.Lx, args.Ly, args.h, psi_exact,
                    max_bond_dim=chi, pbc=pbc,
                )
                results.append(mps)
                rel_exact = rel_error_to_ref(mps["energy"], E_exact)
                print(f"  χ={chi:>2d}: E = {mps['energy']:.8f}, params = {mps['n_params']}, time = {mps['elapsed_s']:.3f}s")
                if E_exact is not None:
                    print(f"         relative error vs ED = {rel_exact:.2e}")
        except Exception as e:
            print(f"  Skipped: {e}")
        print()

    if args.with_dmrg:
        dmrg_sweep: Optional[Sequence[int]]
        if args.dmrg_chi_sweep_default:
            dmrg_sweep = [4, 6, 8, 10, 12, 14, 16]
        elif args.dmrg_chi_sweep:
            dmrg_sweep = list(dict.fromkeys(int(x) for x in args.dmrg_chi_sweep))
        else:
            dmrg_sweep = [int(args.dmrg_chi)]

        print("Running DMRG baseline (TeNPy)...")
        try:
            for chi in dmrg_sweep:
                dres = run_dmrg_tfim(
                    args.Lx, args.Ly, args.h,
                    chi_max=chi,
                    n_sweeps=args.dmrg_sweeps,
                    svd_min=args.dmrg_svd_min,
                    pbc=pbc,
                )
                results.append(dres)
                rel_exact = rel_error_to_ref(dres["energy"], E_exact)
                print(f"  χ={chi:>2d}: E = {dres['energy']:.8f}, params = {dres['n_params']}, time = {dres['elapsed_s']:.1f}s")
                if E_exact is not None:
                    print(f"         relative error vs ED = {rel_exact:.2e}")
        except Exception as e:
            print(f"  Failed: {e}")
        print()

    if not args.rbm_only:
        print(f"Running {args.ar_model} autoregressive model...")
        try:
            arnn_features = args.arnn_features
            arnn_layers = args.arnn_layers
            transformer_d_model = args.transformer_d_model

            if args.budget_mode == "params" and rbm is not None and arnn_features is None and transformer_d_model is None:
                choice = choose_arnn_for_budget(
                    args.Lx, args.Ly,
                    target_params=rbm["n_params"],
                    pbc=pbc,
                    layers_override=arnn_layers,
                    features_override=None,
                    ar_model=args.ar_model,
                    transformer_d_model_override=None,
                    transformer_heads=args.transformer_heads,
                    transformer_ff_mult=args.transformer_ff_mult,
                )
                if choice is not None:
                    selected_width, arnn_layers, est_params = choice
                    if args.ar_model == "transformer":
                        transformer_d_model = selected_width
                        print(f"  auto-selected d_model={transformer_d_model}, layers={arnn_layers} (~{est_params} params)")
                    else:
                        arnn_features = selected_width
                        print(f"  auto-selected features={arnn_features}, layers={arnn_layers} (~{est_params} params)")
                else:
                    print("  auto-selection failed; falling back to default autoregressive width")

            arnn = run_arnn(
                args.Lx, args.Ly, args.h,
                n_samples=args.arnn_n_samples if args.arnn_n_samples is not None else args.n_samples,
                n_iter=args.arnn_n_iter if args.arnn_n_iter is not None else args.n_iter,
                arnn_features=arnn_features,
                arnn_layers=arnn_layers,
                pbc=pbc,
                ar_model=args.ar_model,
                transformer_d_model=transformer_d_model,
                transformer_heads=args.transformer_heads,
                transformer_ff_mult=args.transformer_ff_mult,
            )
            results.append(arnn)
            rel_exact = rel_error_to_ref(arnn["energy"], E_exact)
            print(f"  E = {arnn['energy']:.8f} ± {arnn['std']:.2e}")
            print(f"  params = {arnn['n_params']}, time = {arnn['elapsed_s']:.1f}s, sampler = {arnn.get('sampler', '')}")
            if E_exact is not None:
                print(f"  relative error vs ED = {rel_exact:.2e}")
        except Exception as e:
            print(f"  Failed: {e}")
        print()

    if not results:
        print("No results — check that netket is installed (and tenpy if using DMRG).")
        return

    E_ref, ref_label, ref_mode = choose_reference_energy(results, E_exact, mode=args.reference_mode)
    print(f"Comparison reference: {ref_label} = {E_ref:.10f}" if E_ref is not None else "Comparison reference unavailable")
    print()

    sweep_active = any([
        bool(args.mps_chi_sweep),
        bool(args.mps_chi_sweep_default),
        bool(args.dmrg_chi_sweep),
        bool(args.dmrg_chi_sweep_default),
    ])
    csv_path = args.csv_out or (default_csv_filename(args.Lx, args.Ly, args.h, sweep=sweep_active) if sweep_active else None)
    if csv_path:
        write_csv_summary(csv_path, results, E_exact, E_ref, ref_label, ref_mode, args.Lx, args.Ly, args.h, pbc=pbc)
        print()

    print("=" * 72)
    print(f"SUMMARY: 2D TFIM {args.Lx}×{args.Ly}, h/J = {args.h:.3f}, {boundary}")
    print("=" * 72)
    if E_exact is not None:
        print(f"  ED exact:        {E_exact:.10f}  (E/N = {E_exact / N:.10f})")
    if E_ref is not None and (E_exact is None or abs(E_ref - E_exact) > 0):
        print(f"  {ref_label:14s}: {E_ref:.10f}")
    for r in results:
        tag = _result_label(r)
        std_txt = f" ± {r['std']:.2e}" if not np.isnan(r["std"]) else ""
        rel_ref = rel_error_to_ref(r["energy"], E_ref)
        d_ref = delta_to_ref(r["energy"], E_ref)
        extra = ""
        if r["ansatz"] == "MPS":
            extra = f", F²={r['fidelity']:.6f}"
        print(
            f"  {tag:18s}: {r['energy']:.10f}{std_txt}  "
            f"({r['n_params']} params, {r['elapsed_s']:.2f}s{extra})  "
            f"ΔE_ref={d_ref:.6g}, rel={rel_ref:.2e}"
        )
    print()

    if rbm is not None and args.budget_mode == "params":
        print("Budget note: parameter-matched mode used the RBM parameter count as the target.")
        print()

    make_figure(results, E_exact, E_ref, ref_label, args.Lx, args.Ly, args.h, pbc=pbc)
    if args.param_scaling_plot:
        make_param_scaling_plot(
            results, E_ref, ref_label,
            args.Lx, args.Ly, args.h,
            pbc=pbc,
            out_base=args.param_plot_out,
        )


if __name__ == "__main__":
    main()
