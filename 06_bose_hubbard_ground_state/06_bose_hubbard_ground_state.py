"""
06: Bose–Hubbard Ground State with NQS
=======================================
Demonstrates that the VMC + SR pipeline (§2–§4) transfers directly from
spin models to bosonic systems — the target model for FOR 5919 Project 01.

This version makes the U(1) number-sector reduction explicit.  At unit
filling, all benchmark demos work in the fixed-particle sector
N_tot = N_sites by constructing the NetKet Hilbert space as

    nk.hilbert.Fock(n_max=n_max, N=N_sites, n_particles=N_sites)

This is the direct Bose–Hubbard analog of the charge-sector projection used
in the Schwinger-model benchmarks: the variational search is confined to the
physically relevant symmetry sector, and the exact / sampling backends only
see the reduced basis.

Three demonstrations:

  1. **n_max convergence** (§1.4): sweep the Fock-space truncation n_max
     at fixed U/t and verify that the energy converges.  This is the key
     bosonic-specific check — the spin pipeline had no analog.

  2. **U/t phase diagram**: sweep U/t from superfluid to Mott insulator,
     tracking density fluctuations ⟨δn²⟩ = ⟨n²⟩ − ⟨n⟩² as the order
     parameter.  Large fluctuations = superfluid; suppressed = Mott.

  3. **2D lattice** (Project 01 geometry): one VMC+SR run on a small
     square lattice to show the machinery works in 2D.

All runs validated against exact diagonalization (Lanczos) for small
systems.

Physics (§1.4):
  H_BH = −t Σ (a†_i a_j + h.c.) + (U/2) Σ n_i(n_i−1) − μ Σ n_i

  - U/t ≫ 1:  Mott insulator (integer filling, suppressed fluctuations)
  - U/t ≪ 1:  Superfluid (delocalized, large fluctuations)
  - 2D critical point at unit filling: (U/t)_c ≈ 16.7  [Capogrosso-Sansone
    et al., PRB 77, 015602 (2008)]

Key Bose–Hubbard details (§1.4):
  - Local Hilbert space is truncated Fock space {|0⟩,...,|n_max⟩}, not {↑,↓}
  - Hopping matrix elements are occupation-dependent: √((n_i+1)·n_j)
  - Total particle number is conserved, so unit-filling runs should exploit
    the fixed-N sector whenever possible
  - For NQS, this changes local-energy evaluation but NOT the VMC/SR
    infrastructure — the same O_k, S, gradient machinery applies.

Requirements:
    netket >= 3.0, jax, optax, numpy, matplotlib
    Python >= 3.11

Usage:
    python bose_hubbard_ground_state.py

    # Quick test:
    python bose_hubbard_ground_state.py --N 6 --quick

    # U/t sweep only:
    python bose_hubbard_ground_state.py --mode sweep-only

    # 2D lattice:
    python bose_hubbard_ground_state.py --Lx 3 --Ly 3 --mode 2d
"""

import argparse
import csv
import time
from functools import lru_cache
import numpy as np
import matplotlib.pyplot as plt


@lru_cache(maxsize=None)
def bounded_fock_sector_dim(n_sites, n_particles, n_max):
    """Count fixed-number Fock states with 0 <= n_i <= n_max and Σ_i n_i = n_particles."""
    n_sites = int(n_sites)
    n_particles = int(n_particles)
    n_max = int(n_max)
    if n_sites < 0 or n_particles < 0 or n_max < 0:
        return 0
    if n_particles > n_sites * n_max:
        return 0

    dp = [0] * (n_particles + 1)
    dp[0] = 1
    for _ in range(n_sites):
        nxt = [0] * (n_particles + 1)
        for total, count in enumerate(dp):
            if count == 0:
                continue
            max_occ = min(n_max, n_particles - total)
            for occ in range(max_occ + 1):
                nxt[total + occ] += count
        dp = nxt
    return int(dp[n_particles])


def hilbert_space_stats(n_sites, n_max, n_particles=None):
    """Return (full_dim, sector_dim_or_None, reduction_factor)."""
    n_sites = int(n_sites)
    n_max = int(n_max)
    full_dim = int((n_max + 1) ** n_sites)
    if n_particles is None:
        return full_dim, None, 1.0

    sector_dim = bounded_fock_sector_dim(n_sites, int(n_particles), n_max)
    if sector_dim <= 0:
        reduction = float("inf")
    else:
        reduction = float(full_dim) / float(sector_dim)
    return full_dim, int(sector_dim), reduction


def format_hilbert_space_stats(n_sites, n_max, n_particles=None):
    """Human-readable Hilbert-space summary."""
    full_dim, sector_dim, reduction = hilbert_space_stats(n_sites, n_max, n_particles)
    if n_particles is None:
        return f"full dim={full_dim:,} (grand-canonical)"
    return (
        f"full dim={full_dim:,}, fixed-N sector dim={sector_dim:,} "
        f"(N_tot={int(n_particles)}), reduction={reduction:.1f}×"
    )


# ═══════════════════════════════════════════════════════════════
# BOSE–HUBBARD HAMILTONIAN SETUP  (§1.4)
# ═══════════════════════════════════════════════════════════════

def build_bh_netket(graph, n_max, U, t_hop, mu=0.0, n_particles=None):
    """
    Build the Bose–Hubbard Hamiltonian and Hilbert space in NetKet.

    H = −t Σ_{⟨i,j⟩} (a†_i a_j + a†_j a_i) + (U/2) Σ_i n_i(n_i−1) − μ Σ_i n_i

    Uses NetKet's built-in BoseHubbard operator, which handles the
    Fock-space bookkeeping natively (§1.4): occupation-number basis,
    truncation at n_max, and occupation-dependent matrix elements
    √((n_i+1)·n_j) for the hopping terms.

    Crucially, passing ``n_particles`` activates an explicit U(1)
    number-sector projection.  At unit filling this is the direct bosonic
    analog of the Schwinger-model charge-sector projection: the variational
    search, exact diagonalization, and exact sampler all operate only in the
    fixed-N block rather than the full Fock space.

    Args:
        graph: NetKet graph (Chain, Square, etc.)
        n_max: maximum occupation per site (local dim = n_max + 1)
        U: on-site interaction
        t_hop: hopping amplitude
        mu: chemical potential (default 0 for unit filling)
        n_particles: if set, restricts to canonical ensemble (fixed N)

    Returns:
        (hilbert, H) — the Fock-space Hilbert space and Hamiltonian
    """
    import netket as nk

    N_sites = graph.n_nodes

    if n_particles is not None:
        hi = nk.hilbert.Fock(n_max=n_max, N=N_sites,
                             n_particles=n_particles)
    else:
        hi = nk.hilbert.Fock(n_max=n_max, N=N_sites)

    H = nk.operator.BoseHubbard(
        hilbert=hi,
        graph=graph,
        U=U,
        J=t_hop,
        mu=mu,
    )

    return hi, H


def build_observables(hi, N_sites):
    """
    Build density and density-fluctuation operators for measurements.

    ⟨n_i⟩ = local density
    ⟨δn²⟩ = ⟨n²⟩ − ⟨n⟩² = density fluctuation (order parameter:
             large in superfluid, suppressed in Mott insulator)
    """
    import netket as nk

    # n_i operators (number operator at each site)
    n_ops = [nk.operator.boson.number(hi, i) for i in range(N_sites)]

    # Total number operator: N_tot = Σ n_i
    N_tot = sum(n_ops)

    # n² operator for fluctuation: Σ n_i²
    # NetKet doesn't have a built-in n² so we construct it
    # We'll measure ⟨n_i⟩ and ⟨n_i²⟩ from samples instead
    return n_ops, N_tot


# ═══════════════════════════════════════════════════════════════
# ED REFERENCE
# ═══════════════════════════════════════════════════════════════

def ed_reference(graph, n_max, U, t_hop, mu=0.0, n_particles=None):
    """
    Exact diagonalization for small Bose–Hubbard systems.

    The full Fock-space dimension is (n_max + 1)^N, but in the fixed-number
    sector the relevant size is the bounded-composition count for
    Σ_i n_i = n_particles with 0 <= n_i <= n_max.  When ``n_particles`` is
    supplied, this function uses the reduced sector size to decide whether ED
    is tractable.

    Returns E0 or None if too large.
    """
    import netket as nk

    N_sites = graph.n_nodes
    full_dim, sector_dim, reduction = hilbert_space_stats(N_sites, n_max, n_particles)
    effective_dim = full_dim if n_particles is None else sector_dim

    # Exact diagonalization is practical on substantially larger constrained
    # sectors than on the full space, so gate on the effective block size.
    dim_limit = 100000 if n_particles is None else 200000
    if effective_dim is None or effective_dim > dim_limit:
        return None

    hi, H = build_bh_netket(graph, n_max, U, t_hop, mu, n_particles)

    try:
        E0 = nk.exact.lanczos_ed(H, k=1, compute_eigenvectors=False)[0]
        return float(np.real(E0))
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════
# NQS GROUND STATE  (VMC + SR — same pipeline as spin models)
# ═══════════════════════════════════════════════════════════════


def _round_up_to_multiple(n, multiple):
    n = int(max(1, n))
    multiple = int(max(1, multiple))
    return int(((n + multiple - 1) // multiple) * multiple)


def _extract_r_hat(stats):
    """Best-effort extraction of R-hat from a NetKet Stats object."""
    for name in ("R_hat", "Rhat", "r_hat"):
        value = getattr(stats, name, None)
        if value is not None:
            try:
                arr = np.asarray(value)
                if arr.size:
                    return float(np.max(np.real(arr)))
            except Exception:
                try:
                    return float(value)
                except Exception:
                    pass
    return float("nan")


def _repeat_fraction(values, decimals=8):
    arr = np.asarray(values)
    if arr.size == 0:
        return float("nan")
    arr = np.round(np.real(arr), decimals=decimals)
    unique = np.unique(arr)
    return 1.0 - float(len(unique)) / float(len(arr))


def _exact_energy_from_vstate(vstate, H):
    """Return the exact variational energy using the dense statevector."""
    psi = np.asarray(vstate.to_array(normalize=False)).reshape(-1)
    norm = float(np.vdot(psi, psi).real)
    if norm <= 0 or not np.isfinite(norm):
        raise RuntimeError("Failed to reconstruct a finite dense statevector.")
    psi = psi / np.sqrt(norm)
    Hmat = H.to_sparse()
    energy = float(np.real(np.vdot(psi, Hmat @ psi)))
    return energy, norm, psi


def make_sampler(hi, H, graph, n_particles=None, sampler_kind="auto",
                 n_chains=32, sweep_multiplier=4):
    """Choose a sampler that is robust in the fixed-particle Fock sector.

    For tiny / moderate indexable Hilbert spaces, `ExactSampler` is the most
    reliable option because it draws independent samples from |ψ|² and avoids
    Markov-chain mixing failures entirely. This is ideal for the 1D demos and
    for the tiny validation harness.

    For larger spaces where exact sampling is too expensive, fixed-particle
    runs prefer `MetropolisHamiltonian`, which proposes moves connected by the
    off-diagonal hopping terms of the Bose–Hubbard Hamiltonian.
    """
    import netket as nk

    n_chains = int(max(1, n_chains))
    sweep_size = int(max(1, sweep_multiplier * graph.n_nodes))

    try:
        n_states = int(getattr(hi, "n_states"))
    except Exception:
        n_states = None

    exact_threshold = 50000
    if sampler_kind in ("auto", "exact"):
        exact = getattr(nk.sampler, "ExactSampler", None)
        use_exact = sampler_kind == "exact" or (n_states is not None and n_states <= exact_threshold)
        if exact is not None and use_exact:
            for kwargs in (
                {"hilbert": hi},
                {},
            ):
                try:
                    if kwargs:
                        return exact(**kwargs), "exact", 0
                    return exact(hi), "exact", 0
                except Exception:
                    pass

    if n_particles is not None and sampler_kind in ("auto", "hamiltonian"):
        mh = getattr(nk.sampler, "MetropolisHamiltonian", None)
        if mh is not None:
            for kwargs in (
                {"hilbert": hi, "hamiltonian": H, "n_chains": n_chains, "sweep_size": sweep_size},
                {"hilbert": hi, "operator": H, "n_chains": n_chains, "sweep_size": sweep_size},
                {"hilbert": hi, "rule": getattr(getattr(nk.sampler, "rules", object()), "HamiltonianRule", lambda op: None)(H), "n_chains": n_chains, "sweep_size": sweep_size},
                {"hilbert": hi, "rule": getattr(getattr(nk.sampler, "rules", object()), "HamiltonianRuleNumpy", lambda op: None)(H), "n_chains": n_chains, "sweep_size": sweep_size},
            ):
                try:
                    if kwargs.get("rule", "ok") is None:
                        continue
                    return mh(**kwargs), "hamiltonian", sweep_size
                except Exception:
                    pass
            try:
                return mh(hi, H, n_chains=n_chains, sweep_size=sweep_size), "hamiltonian", sweep_size
            except Exception:
                pass

    if n_particles is not None and sampler_kind in ("auto", "exchange"):
        me = getattr(nk.sampler, "MetropolisExchange", None)
        if me is not None:
            try:
                return me(hi, graph=graph, n_chains=n_chains, sweep_size=sweep_size), "exchange", sweep_size
            except Exception:
                try:
                    return me(hi, graph=graph, n_chains=n_chains), "exchange", sweep_size
                except Exception:
                    pass

    ml = nk.sampler.MetropolisLocal
    try:
        return ml(hi, n_chains=n_chains, sweep_size=sweep_size), "local", sweep_size
    except Exception:
        return ml(hi, n_chains=n_chains), "local", sweep_size



def make_variational_state(hi, H, graph, model, *, n_particles=None,
                           n_samples=1024, sampler_kind="auto",
                           n_chains=32, sweep_multiplier=4,
                           n_discard_per_chain=None, fullsum_threshold=50000,
                           allow_fullsum=False):
    """Create a variational state.

    By default, the demos always use MCState (optionally with ExactSampler on
    small spaces) so the optimization path stays aligned with the original VMC
    workflow. FullSumState is reserved for exact diagnostics / validation.
    """
    import netket as nk

    try:
        n_states = int(getattr(hi, "n_states"))
    except Exception:
        n_states = None

    fullsum_ctor = getattr(getattr(nk, "vqs", object()), "FullSumState", None)
    use_fullsum = bool(allow_fullsum) and fullsum_ctor is not None and n_states is not None and n_states <= int(fullsum_threshold)

    if use_fullsum:
        attempts = (
            ((hi, model), {}),
            ((), {"hilbert": hi, "model": model}),
            ((model,), {"hilbert": hi}),
            ((hi,), {"model": model}),
        )
        for args, kwargs in attempts:
            try:
                vstate = fullsum_ctor(*args, **kwargs)
                return vstate, "fullsum", 0, 0
            except Exception:
                pass

    sampler, sampler_used, sweep_size = make_sampler(
        hi, H, graph, n_particles=n_particles, sampler_kind=sampler_kind,
        n_chains=n_chains, sweep_multiplier=sweep_multiplier,
    )
    if sampler_used == "exact":
        n_discard = 0
    elif n_discard_per_chain is None:
        n_discard = int(max(4 * sweep_size, graph.n_nodes))
    else:
        n_discard = int(max(0, n_discard_per_chain))

    n_chains = int(max(1, n_chains))
    n_samples = _round_up_to_multiple(max(n_samples, n_chains), n_chains)
    vstate = nk.vqs.MCState(
        sampler,
        model,
        n_samples=n_samples,
        n_discard_per_chain=n_discard,
    )
    return vstate, sampler_used, sweep_size, n_discard

def run_vmc(graph, n_max, U, t_hop, mu=0.0, n_particles=None,
            alpha=4, n_samples=4096, n_iter=600, lr=0.001,
            diag_shift=0.10, sampler_kind="auto", n_chains=32,
            sweep_multiplier=4, n_discard_per_chain=None,
            use_complex_rbm=False, sr_mode="auto",
            return_internal=False, allow_fullsum=False):
    """
    VMC ground-state search for Bose–Hubbard.

    In `sr_mode=auto`, stochastic reconfiguration is disabled for exact /
    full-summation states so the small-system debugging path uses a plain
    descent update without any QGT preconditioning.

    When ``n_particles`` is set (the default choice for all unit-filling demos),
    the variational search is confined to the fixed total-number sector.
    """
    import netket as nk
    import optax

    hi, H = build_bh_netket(graph, n_max, U, t_hop, mu, n_particles)
    N_sites = graph.n_nodes
    full_dim, sector_dim, reduction_factor = hilbert_space_stats(N_sites, n_max, n_particles)

    param_dtype = complex if use_complex_rbm else float
    model = nk.models.RBM(alpha=alpha, param_dtype=param_dtype)

    n_chains = int(max(1, n_chains))
    n_samples = _round_up_to_multiple(max(n_samples, n_chains), n_chains)
    vstate, sampler_used, sweep_size, n_discard = make_variational_state(
        hi, H, graph, model,
        n_particles=n_particles, n_samples=n_samples,
        sampler_kind=sampler_kind, n_chains=n_chains,
        sweep_multiplier=sweep_multiplier,
        n_discard_per_chain=n_discard_per_chain,
        allow_fullsum=allow_fullsum,
    )

    if sr_mode not in {"auto", "on", "off"}:
        raise ValueError(f"Unsupported sr_mode={sr_mode!r}")
    use_sr = (sr_mode == "on") or (sr_mode == "auto" and sampler_used not in ("exact", "fullsum"))

    if use_sr:
        opt = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(learning_rate=lr),
        )
        preconditioner = nk.optimizer.SR(diag_shift=diag_shift)
        optimizer_name = "adam+sr"
    else:
        opt = optax.sgd(learning_rate=lr)
        preconditioner = None
        optimizer_name = "sgd"

    driver_kwargs = {"variational_state": vstate}
    if preconditioner is not None:
        driver_kwargs["preconditioner"] = preconditioner
    driver = nk.driver.VMC(H, opt, **driver_kwargs)

    log = nk.logging.RuntimeLog()
    t0 = time.time()
    driver.run(n_iter=n_iter, out=log)
    elapsed = time.time() - t0

    energy_data = log.data.get("Energy", {})
    if isinstance(energy_data, dict) and "Mean" in energy_data:
        raw_trace = energy_data["Mean"]
    else:
        raw_trace = log["Energy"]["Mean"]
    energy_trace = np.real(np.asarray(raw_trace))

    E_stats = vstate.expect(H)
    final_e = float(np.real(E_stats.mean))
    err_attr = getattr(E_stats, "error_of_mean", None)
    if err_attr is None:
        final_std = 0.0
    else:
        try:
            final_std = float(np.real(err_attr))
        except Exception:
            final_std = float(np.real(err_attr()))
    r_hat = _extract_r_hat(E_stats) if sampler_used not in ("fullsum", "exact") else float("nan")

    samples = np.empty((0, N_sites))
    if sampler_used == "fullsum":
        exact_e, _, psi_exact = _exact_energy_from_vstate(vstate, H)
        probs = np.abs(psi_exact) ** 2
        states = np.asarray(hi.all_states())
        occ_mean = np.mean(states, axis=1)
        occ2_mean = np.mean(states ** 2, axis=1)
        n_mean = float(np.sum(probs * occ_mean))
        n2_mean = float(np.sum(probs * occ2_mean))
        delta_n2 = float(n2_mean - n_mean ** 2)
        # fullsum states have exact expectation, so report zero statistical error
        final_e = exact_e
        final_std = 0.0
    else:
        try:
            samples = np.asarray(vstate.samples).reshape(-1, N_sites)
            n_mean = float(np.mean(samples))
            n2_mean = float(np.mean(samples ** 2))
            delta_n2 = float(n2_mean - n_mean ** 2)
        except Exception:
            n_mean = float("nan")
            delta_n2 = float("nan")

    out = {
        "energy": final_e,
        "std": final_std,
        "n_params": int(vstate.n_parameters),
        "trace": energy_trace,
        "elapsed_s": elapsed,
        "n_max": n_max,
        "full_dim": int(full_dim),
        "sector_dim": int(sector_dim if sector_dim is not None else full_dim),
        "reduction_factor": float(reduction_factor),
        "U": U,
        "t_hop": t_hop,
        "n_mean": n_mean,
        "delta_n2": delta_n2,
        "n_samples": int(n_samples if sampler_used != "fullsum" else 0),
        "n_iter": int(n_iter),
        "sampler": sampler_used,
        "optimizer": optimizer_name,
        "sr_enabled": bool(use_sr),
        "n_chains": int(n_chains),
        "sweep_size": int(sweep_size),
        "n_discard_per_chain": int(n_discard),
        "r_hat": r_hat,
        "repeat_frac": float("nan"),
    }
    if sampler_used != "fullsum" and samples.size:
        try:
            local_vals = np.asarray(vstate.local_estimators(H)).reshape(-1)
            out["repeat_frac"] = _repeat_fraction(local_vals)
        except Exception:
            pass

    if return_internal:
        out.update({
            "hilbert": hi,
            "hamiltonian": H,
            "vstate": vstate,
            "model": model,
        })

    return out


def one_step_exact_descent_test(alpha=4, lr=1e-2, use_complex_rbm=False):
    """Run a single exact gradient-descent step on the tiniest nontrivial case."""
    import netket as nk
    import optax

    g = nk.graph.Chain(length=2, pbc=True)
    hi, H = build_bh_netket(g, n_max=2, U=4.0, t_hop=1.0, n_particles=2)

    param_dtype = complex if use_complex_rbm else float
    model = nk.models.RBM(alpha=alpha, param_dtype=param_dtype)
    vstate, sampler_used, sweep_size, n_discard = make_variational_state(
        hi, H, g, model,
        n_particles=2, n_samples=512,
        sampler_kind="auto", n_chains=8,
        sweep_multiplier=4,
        n_discard_per_chain=0,
        allow_fullsum=True,
    )

    e_before, _, _ = _exact_energy_from_vstate(vstate, H)
    opt = optax.sgd(learning_rate=lr)
    driver = nk.driver.VMC(H, opt, variational_state=vstate)
    driver.run(n_iter=1, out=nk.logging.RuntimeLog())
    e_after, _, _ = _exact_energy_from_vstate(vstate, H)

    return {
        "initial_energy": e_before,
        "final_energy": e_after,
        "delta_energy": e_after - e_before,
        "improved": bool(e_after < e_before),
        "sampler": sampler_used,
        "sweep_size": int(sweep_size),
        "n_discard_per_chain": int(n_discard),
        "optimizer": "sgd",
        "lr": float(lr),
        "E_ed": ed_reference(g, 2, 4.0, 1.0, n_particles=2),
    }

# ═══════════════════════════════════════════════════════════════
# DEMONSTRATION 0: TINY VALIDATION
# ═══════════════════════════════════════════════════════════════

def tiny_deterministic_validation(alpha=4, n_iter=200, n_samples=4096,
                                  lr=0.001, diag_shift=0.10,
                                  sampler_kind="auto", n_chains=32,
                                  sweep_multiplier=8,
                                  n_discard_per_chain=None,
                                  use_complex_rbm=False, sr_mode="auto"):
    """Tiny exact-vs-MC check plus a one-step exact descent diagnostic."""
    import netket as nk

    print("=" * 55)
    print("DEMO 0: Tiny deterministic validation")
    print("=" * 55)

    descent = one_step_exact_descent_test(
        alpha=alpha,
        lr=max(lr, 1e-2),
        use_complex_rbm=use_complex_rbm,
    )
    print("  Step A — one-step exact descent (N=2, n_max=2, n_particles=2, U/t=4.0)")
    if descent["E_ed"] is not None:
        print(f"    ED ground energy   = {descent['E_ed']:.8f}")
    print(f"    E_before          = {descent['initial_energy']:.8f}")
    print(f"    E_after 1 step    = {descent['final_energy']:.8f}")
    print(f"    ΔE (after-before) = {descent['delta_energy']:.8f}")
    print(f"    Optimizer         = {descent['optimizer']}  (lr={descent['lr']:.3g})")
    print(f"    State backend     = {descent['sampler']}")
    if descent["improved"]:
        print("    Result            = PASS (single exact descent step lowered the energy)")
    else:
        print("    Result            = FAIL (single exact descent step raised the energy)")
        print("                        This strongly suggests a sign / update-rule bug.")

    print()
    print("  Step B — exact-vs-estimator consistency (N=3, n_max=3, n_particles=3, U/t=10.0)")

    g = nk.graph.Chain(length=3, pbc=True)
    r = run_vmc(
        g, 3, 10.0, 1.0, n_particles=3,
        alpha=alpha, n_samples=n_samples, n_iter=n_iter, lr=lr,
        diag_shift=diag_shift, sampler_kind=sampler_kind, n_chains=n_chains,
        sweep_multiplier=sweep_multiplier,
        n_discard_per_chain=n_discard_per_chain,
        use_complex_rbm=use_complex_rbm, sr_mode=sr_mode,
        return_internal=True,
        allow_fullsum=False,
    )

    H = r["hamiltonian"]
    vstate = r["vstate"]
    dense_e, norm_before, psi = _exact_energy_from_vstate(vstate, H)
    Hmat = H.to_sparse()
    Hpsi = Hmat @ psi
    E_ed = ed_reference(g, 3, 10.0, 1.0, n_particles=3)

    samples = np.empty((0, g.n_nodes), dtype=int)
    mc_loc = np.asarray([], dtype=complex)
    dense_loc_sample = np.asarray([], dtype=complex)
    if r["sampler"] != "fullsum":
        samples = np.asarray(vstate.samples).reshape(-1, g.n_nodes)
        mc_loc = np.asarray(vstate.local_estimators(H)).reshape(-1)
        sample_ids = np.asarray(r["hilbert"].states_to_numbers(samples[:3]), dtype=int)
        dense_loc_sample = Hpsi[sample_ids] / psi[sample_ids]

    gap = abs(r["energy"] - dense_e)
    tol = max(3.0 * max(r["std"], 1e-12), 1e-6)
    consistent = bool(np.isfinite(gap) and gap <= tol)

    print(f"    ED ground energy   = {E_ed:.8f}" if E_ed is not None else "    ED ground energy   = n/a")
    print(f"    Reported estimate  = {r['energy']:.8f} ± {r['std']:.2e}")
    print(f"    Dense ⟨ψ|H|ψ⟩      = {dense_e:.8f}  (basis size = {psi.size})")
    print(f"    ||ψ||² before norm = {norm_before:.8f}")
    print(f"    |estimate-dense|   = {gap:.8f}  (tolerance = {tol:.8f})")
    dense_sample = ", ".join(f"{z.real:.4f}{z.imag:+.4f}j" for z in (Hpsi[:3] / psi[:3]))
    print(f"    Sample dense E_loc = {dense_sample}")
    if dense_loc_sample.size:
        print("    Dense E_loc@sample = " + ", ".join(f"{z.real:.4f}{z.imag:+.4f}j" for z in dense_loc_sample))
        print("    MC    E_loc@sample = " + ", ".join(f"{z.real:.4f}{z.imag:+.4f}j" for z in mc_loc[:len(dense_loc_sample)]))
        print(f"    max |ΔE_loc|       = {float(np.max(np.abs(dense_loc_sample - mc_loc[:len(dense_loc_sample)]))):.8f}")
    else:
        print("    Dense E_loc@sample = n/a (fullsum state)")
        print("    MC    E_loc@sample = n/a (fullsum state)")
        print("    max |ΔE_loc|       = n/a")
    print(f"    Sampler            = {r['sampler']}")
    print(f"    Optimizer          = {r.get('optimizer', '?')}")
    print(f"    SR enabled         = {r.get('sr_enabled', False)}")
    print(f"    R_hat (final)      = {r['r_hat']:.4f}")
    print(f"    MC repeat frac.    = {r['repeat_frac']:.4f}")

    print()
    if not descent["improved"]:
        print("  Diagnosis: the exact one-step descent test failed.")
        print("             Focus on the parameter-update rule before any large runs.")
    elif r["sampler"] == "fullsum":
        if consistent:
            print("  Diagnosis: tiny case passes the exact consistency check.")
            print("             Full-summation variational expectations match the dense reconstruction.")
        else:
            print("  Diagnosis: tiny case FAILS the exact consistency check.")
            print("             Full-summation and dense expectations should agree; inspect state reconstruction.")
    elif consistent and (not np.isfinite(r['r_hat']) or r['r_hat'] <= 1.10):
        print("  Diagnosis: tiny case passes the consistency check.")
        print("             Estimator and dense expectations agree within the tolerance.")
    else:
        print("  Diagnosis: tiny case FAILS the consistency check.")
        if not consistent:
            print("             Reported and dense expectations do not agree.")
            print("             This points to estimator / sampling issues, not the Hamiltonian path.")
        if np.isfinite(r['r_hat']) and r['r_hat'] > 1.10:
            print("             R_hat is above 1.10, so the Markov chains are not mixing well.")
        if np.isfinite(r['repeat_frac']) and r['repeat_frac'] > 0.95:
            print("             The sampled local energies are highly repetitive, suggesting stuck chains.")

    return {
        "descent_improved": descent["improved"],
        "energy_reported": r["energy"],
        "energy_dense": dense_e,
        "energy_ed": E_ed,
        "consistent": consistent,
        "r_hat": r["r_hat"],
        "repeat_frac": r["repeat_frac"],
        "sampler": r["sampler"],
        "optimizer": r.get("optimizer", "?"),
    }

# ═══════════════════════════════════════════════════════════════
# DEMONSTRATION 1: n_max CONVERGENCE  (§1.4 key check)
# ═══════════════════════════════════════════════════════════════


def nmax_convergence(N, U, t_hop, alpha=4, n_iter=300, n_samples=4096,
                     lr=0.001, diag_shift=0.10,
                     sampler_kind="auto", n_chains=32, sweep_multiplier=4,
                     n_discard_per_chain=None, use_complex_rbm=False,
                     sr_mode="auto"):
    """Sweep n_max = 1, 2, 3, 4 and show energy convergence."""
    import netket as nk

    print(f"  n_max convergence: N={N}, U/t={U/t_hop:.1f}")
    g = nk.graph.Chain(length=N, pbc=True)

    results = []
    for n_max in [1, 2, 3, 4]:
        local_dim = n_max + 1
        full_dim, sector_dim, reduction = hilbert_space_stats(N, n_max, N)
        print(f"\n    n_max={n_max}  "
              f"(local dim={local_dim})")
        print(f"      {format_hilbert_space_stats(N, n_max, N)}")

        if n_max == 1:
            print("      Trivial frozen sector at unit filling; skipping VMC.")
            full_dim, sector_dim, reduction = hilbert_space_stats(N, 1, N)
            results.append({
                "energy": 0.0, "std": float("nan"), "n_params": 0,
                "trace": np.asarray([0.0]), "elapsed_s": 0.0,
                "n_max": 1, "full_dim": full_dim, "sector_dim": sector_dim,
                "reduction_factor": reduction,
                "U": U, "t_hop": t_hop,
                "n_mean": 1.0, "delta_n2": 0.0, "E_ed": None,
                "n_samples": 0, "n_iter": 0, "sampler": "trivial",
                "optimizer": "none", "sr_enabled": False,
                "r_hat": float("nan"), "repeat_frac": float("nan"),
            })
            continue

        E_ed = ed_reference(g, n_max, U, t_hop, n_particles=N)
        if E_ed is not None:
            print(f"      ED  E0 = {E_ed:.8f}  (E0/N = {E_ed/N:.8f})")

        try:
            r = run_vmc(
                g, n_max, U, t_hop, n_particles=N,
                alpha=alpha, n_samples=n_samples, n_iter=n_iter,
                lr=lr, diag_shift=diag_shift,
                sampler_kind=sampler_kind, n_chains=n_chains,
                sweep_multiplier=sweep_multiplier,
                n_discard_per_chain=n_discard_per_chain,
                use_complex_rbm=use_complex_rbm,
                sr_mode=sr_mode,
            )
            r["E_ed"] = E_ed
            results.append(r)
            rel_err = _relative_error(r["energy"], E_ed)
            print(f"      NQS E0 = {r['energy']:.8f} ± {r['std']:.2e}  "
                  f"({r['n_params']} params, {r['elapsed_s']:.1f}s, {r['sampler']})")
            if E_ed is not None:
                print(f"      rel err = {rel_err:.2e}")
        except Exception as e:
            print(f"      VMC failed: {e}")

    return results

# ═══════════════════════════════════════════════════════════════
# DEMONSTRATION 2: U/t SWEEP (Mott–superfluid crossover)
# ═══════════════════════════════════════════════════════════════


def ut_sweep(N, n_max=3, alpha=4, n_iter=200, n_samples=4096,
             lr=0.001, diag_shift=0.10,
             sampler_kind="auto", n_chains=32, sweep_multiplier=4,
             n_discard_per_chain=None, use_complex_rbm=False,
             sr_mode="auto"):
    """Sweep U/t from superfluid to Mott regime and track density fluctuations."""
    import netket as nk

    g = nk.graph.Chain(length=N, pbc=True)
    t_hop = 1.0
    U_values = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]

    print(f"  U/t sweep: N={N}, n_max={n_max}, t={t_hop}")
    print(f"  {format_hilbert_space_stats(N, n_max, N)}")
    results = []
    for U in U_values:
        print(f"\n    U/t = {U/t_hop:.1f}:")

        E_ed = ed_reference(g, n_max, U, t_hop, n_particles=N)
        if E_ed is not None:
            print(f"      ED  E0/N = {E_ed/N:.6f}")

        try:
            r = run_vmc(
                g, n_max, U, t_hop, n_particles=N,
                alpha=alpha, n_samples=n_samples, n_iter=n_iter,
                lr=lr, diag_shift=diag_shift,
                sampler_kind=sampler_kind, n_chains=n_chains,
                sweep_multiplier=sweep_multiplier,
                n_discard_per_chain=n_discard_per_chain,
                use_complex_rbm=use_complex_rbm,
                sr_mode=sr_mode,
            )
            r["E_ed"] = E_ed
            r["U_over_t"] = U / t_hop
            results.append(r)
            print(f"      NQS E0/N = {r['energy']/N:.6f} ± "
                  f"{r['std']/N:.2e}  "
                  f"⟨δn²⟩ = {r['delta_n2']:.4f}  "
                  f"({r['elapsed_s']:.1f}s, {r['sampler']})")
        except Exception as e:
            print(f"      Failed: {e}")

    return results

# ═══════════════════════════════════════════════════════════════
# DEMONSTRATION 3: 2D LATTICE  (Project 01 target geometry)
# ═══════════════════════════════════════════════════════════════


def run_2d(Lx, Ly, U, t_hop, n_max=3, alpha=4, n_iter=600, n_samples=8192,
           lr=0.0005, diag_shift=0.10,
           sampler_kind="auto", n_chains=32, sweep_multiplier=4,
           n_discard_per_chain=None, use_complex_rbm=False,
           sr_mode="auto"):
    """VMC + SR on a 2D square lattice Bose–Hubbard at unit filling."""
    import netket as nk

    N_sites = Lx * Ly
    try:
        g = nk.graph.Grid(extent=[Lx, Ly], pbc=True)
    except TypeError:
        g = nk.graph.Grid([Lx, Ly], pbc=True)

    print(f"  2D lattice: {Lx}×{Ly} = {N_sites} sites, "
          f"n_max={n_max}, U/t={U/t_hop:.1f}, PBC")
    print(f"  {format_hilbert_space_stats(N_sites, n_max, N_sites)}")

    E_ed = ed_reference(g, n_max, U, t_hop, n_particles=N_sites)
    if E_ed is not None:
        print(f"  ED  E0 = {E_ed:.8f}  (E0/N = {E_ed/N_sites:.8f})")

    try:
        r = run_vmc(
            g, n_max, U, t_hop, n_particles=N_sites,
            alpha=alpha, n_samples=n_samples, n_iter=n_iter,
            lr=lr, diag_shift=diag_shift,
            sampler_kind=sampler_kind, n_chains=n_chains,
            sweep_multiplier=sweep_multiplier,
            n_discard_per_chain=n_discard_per_chain,
            use_complex_rbm=use_complex_rbm,
        )
        r["E_ed"] = E_ed
        r["Lx"] = Lx
        r["Ly"] = Ly
        rel_err = _relative_error(r["energy"], E_ed)
        print(f"  NQS E0 = {r['energy']:.8f} ± {r['std']:.2e}  "
              f"({r['n_params']} params, {r['elapsed_s']:.1f}s, {r['sampler']})")
        if E_ed is not None:
            print(f"  rel err = {rel_err:.2e}")
        return r
    except Exception as e:
        print(f"  Failed: {e}")
        return None

# ═══════════════════════════════════════════════════════════════
# FIGURES
# ═══════════════════════════════════════════════════════════════

def make_figure(nmax_results, sweep_results, result_2d, N):
    """
    Three-panel figure:
      (a) n_max convergence: E0 vs n_max  (ED + NQS)
      (b) U/t sweep: E0/N and ⟨δn²⟩ vs U/t
      (c) 2D lattice convergence trace
    """
    has_2d = result_2d is not None
    n_panels = 3 if has_2d else 2
    fig, axes = plt.subplots(1, n_panels,
                             figsize=(5.5 * n_panels, 5))

    # ── (a) n_max convergence ──
    ax = axes[0]
    if nmax_results:
        nmax_vals = [r["n_max"] for r in nmax_results]
        e_nqs = [r["energy"] for r in nmax_results]
        e_std = [r["std"] if not np.isnan(r["std"]) else 0
                 for r in nmax_results]
        e_ed = [r.get("E_ed") for r in nmax_results]

        ax.errorbar(nmax_vals, e_nqs, yerr=e_std, fmt="o-",
                     color="#2196F3", ms=8, lw=1.5, capsize=4,
                     label="NQS (VMC)")
        ed_x = [x for x, e in zip(nmax_vals, e_ed) if e is not None]
        ed_y = [e for e in e_ed if e is not None]
        if ed_y:
            ax.plot(ed_x, ed_y, "kx", ms=10, mew=2, label="ED exact")

        U_over_t = nmax_results[0]["U"] / nmax_results[0]["t_hop"]
        ax.set_xlabel("$n_{\\mathrm{max}}$ (Fock truncation)", fontsize=12)
        ax.set_ylabel("Ground-state energy $E_0$", fontsize=12)
        ax.set_title(f"(a) $n_{{\\max}}$ convergence\n"
                     f"N={N}, U/t={U_over_t:.0f}", fontsize=11)
        ax.set_xticks(nmax_vals)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # ── (b) U/t sweep ──
    ax = axes[1]
    if sweep_results:
        ut_vals = [r["U_over_t"] for r in sweep_results]
        e_per_site = [r["energy"] / N for r in sweep_results]
        dn2_vals = [r["delta_n2"] for r in sweep_results]

        # Energy on primary axis
        color1 = "#2196F3"
        ax.plot(ut_vals, e_per_site, "o-", color=color1, ms=7, lw=1.5)
        ax.set_xlabel("$U/t$", fontsize=12)
        ax.set_ylabel("$E_0 / N$", fontsize=12, color=color1)
        ax.tick_params(axis="y", labelcolor=color1)
        ax.set_xscale("log")

        # Density fluctuations on secondary axis
        ax2 = ax.twinx()
        color2 = "#E53935"
        ax2.plot(ut_vals, dn2_vals, "s--", color=color2, ms=7, lw=1.5)
        ax2.set_ylabel(
            r"$\langle\delta n^2\rangle$  (density fluct.)",
            fontsize=11, color=color2)
        ax2.tick_params(axis="y", labelcolor=color2)

        ax.set_title(f"(b) Mott–SF crossover\nN={N}", fontsize=11)
        ax.grid(True, alpha=0.3)

        # ED comparison points
        ed_pts = [(r["U_over_t"], r["E_ed"] / N)
                  for r in sweep_results if r.get("E_ed") is not None]
        if ed_pts:
            ax.plot([p[0] for p in ed_pts], [p[1] for p in ed_pts],
                    "kx", ms=9, mew=2, label="ED", zorder=5)
            ax.legend(fontsize=9, loc="lower right")

    # ── (c) 2D lattice convergence ──
    if has_2d:
        ax = axes[2]
        ax.plot(result_2d["trace"], color="#4CAF50", lw=1.2)
        if result_2d.get("E_ed") is not None:
            ax.axhline(result_2d["E_ed"], color="k", ls="--", lw=1,
                        alpha=0.5,
                        label=f"ED = {result_2d['E_ed']:.4f}")
        Lx, Ly = result_2d.get("Lx", "?"), result_2d.get("Ly", "?")
        U_over_t = result_2d["U"] / result_2d["t_hop"]
        ax.set_xlabel("VMC iteration", fontsize=12)
        ax.set_ylabel("Energy", fontsize=12)
        ax.set_title(f"(c) 2D lattice {Lx}×{Ly}\n"
                     f"U/t={U_over_t:.0f}, "
                     f"$n_{{\\max}}$={result_2d['n_max']}",
                     fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Bose–Hubbard ground state with NQS (§1.4)",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig("bose_hubbard_nqs.png", dpi=150, bbox_inches="tight")
    plt.savefig("bose_hubbard_nqs.pdf", bbox_inches="tight")
    print("\nSaved: bose_hubbard_nqs.png / .pdf")


# ═══════════════════════════════════════════════════════════════
# CSV LOGGING
# ═══════════════════════════════════════════════════════════════

def _relative_error(energy, e_ref):
    """Return relative error |E - E_ref| / |E_ref| when reference is available."""
    if e_ref is None:
        return float("nan")
    e_ref = float(e_ref)
    if e_ref == 0.0:
        return float("nan")
    return abs(float(energy) - e_ref) / abs(e_ref)


def save_results_csv(nmax_results, sweep_results, result_2d, N, Lx, Ly,
                     csv_path=None):
    """Save row-wise results from all executed demos."""
    if csv_path is None:
        csv_path = f"bose_hubbard_N{N}_results.csv"

    rows = []

    for r in nmax_results:
        rows.append({
            "demo": "nmax_convergence",
            "N": N,
            "Lx": "",
            "Ly": "",
            "n_max": r.get("n_max", ""),
            "full_dim": r.get("full_dim", ""),
            "sector_dim": r.get("sector_dim", ""),
            "reduction_factor": r.get("reduction_factor", ""),
            "U": r.get("U", ""),
            "t_hop": r.get("t_hop", ""),
            "U_over_t": (r.get("U", float("nan")) / r.get("t_hop", 1.0)),
            "energy": r.get("energy", float("nan")),
            "std": r.get("std", float("nan")),
            "E_ed": r.get("E_ed", float("nan")),
            "rel_err": _relative_error(r.get("energy", float("nan")), r.get("E_ed", None)),
            "n_params": r.get("n_params", ""),
            "elapsed_s": r.get("elapsed_s", ""),
            "n_mean": r.get("n_mean", float("nan")),
            "delta_n2": r.get("delta_n2", float("nan")),
            "sampler": r.get("sampler", ""),
            "optimizer": r.get("optimizer", ""),
            "sr_enabled": r.get("sr_enabled", ""),
            "r_hat": r.get("r_hat", float("nan")),
            "repeat_frac": r.get("repeat_frac", float("nan")),
        })

    for r in sweep_results:
        rows.append({
            "demo": "ut_sweep",
            "N": N,
            "Lx": "",
            "Ly": "",
            "n_max": r.get("n_max", ""),
            "full_dim": r.get("full_dim", ""),
            "sector_dim": r.get("sector_dim", ""),
            "reduction_factor": r.get("reduction_factor", ""),
            "U": r.get("U", ""),
            "t_hop": r.get("t_hop", ""),
            "U_over_t": r.get("U_over_t", float("nan")),
            "energy": r.get("energy", float("nan")),
            "std": r.get("std", float("nan")),
            "E_ed": r.get("E_ed", float("nan")),
            "rel_err": _relative_error(r.get("energy", float("nan")), r.get("E_ed", None)),
            "n_params": r.get("n_params", ""),
            "elapsed_s": r.get("elapsed_s", ""),
            "n_mean": r.get("n_mean", float("nan")),
            "delta_n2": r.get("delta_n2", float("nan")),
            "sampler": r.get("sampler", ""),
            "optimizer": r.get("optimizer", ""),
            "sr_enabled": r.get("sr_enabled", ""),
            "r_hat": r.get("r_hat", float("nan")),
            "repeat_frac": r.get("repeat_frac", float("nan")),
        })

    if result_2d is not None:
        rows.append({
            "demo": "2d",
            "N": int(Lx) * int(Ly),
            "Lx": Lx,
            "Ly": Ly,
            "n_max": result_2d.get("n_max", ""),
            "full_dim": result_2d.get("full_dim", ""),
            "sector_dim": result_2d.get("sector_dim", ""),
            "reduction_factor": result_2d.get("reduction_factor", ""),
            "U": result_2d.get("U", ""),
            "t_hop": result_2d.get("t_hop", ""),
            "U_over_t": (result_2d.get("U", float("nan")) / result_2d.get("t_hop", 1.0)),
            "energy": result_2d.get("energy", float("nan")),
            "std": result_2d.get("std", float("nan")),
            "E_ed": result_2d.get("E_ed", float("nan")),
            "rel_err": _relative_error(result_2d.get("energy", float("nan")), result_2d.get("E_ed", None)),
            "n_params": result_2d.get("n_params", ""),
            "elapsed_s": result_2d.get("elapsed_s", ""),
            "n_mean": result_2d.get("n_mean", float("nan")),
            "delta_n2": result_2d.get("delta_n2", float("nan")),
            "sampler": result_2d.get("sampler", ""),
            "optimizer": result_2d.get("optimizer", ""),
            "sr_enabled": result_2d.get("sr_enabled", ""),
            "r_hat": result_2d.get("r_hat", float("nan")),
            "repeat_frac": result_2d.get("repeat_frac", float("nan")),
        })

    fieldnames = [
        "demo", "N", "Lx", "Ly", "n_max", "full_dim", "sector_dim",
        "reduction_factor", "U", "t_hop", "U_over_t",
        "energy", "std", "E_ed", "rel_err", "n_params", "elapsed_s",
        "n_mean", "delta_n2", "sampler", "optimizer", "sr_enabled", "r_hat", "repeat_frac",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return csv_path


def save_summary_csv(nmax_results, sweep_results, result_2d, N, U, t_hop,
                     summary_csv_path=None):
    """Save compact summary metrics to CSV."""
    if summary_csv_path is None:
        summary_csv_path = f"bose_hubbard_N{N}_summary.csv"

    rows = [
        {"metric": "N_1d", "value": N},
        {"metric": "U_over_t_input", "value": U / t_hop},
        {"metric": "nmax_points", "value": len(nmax_results)},
        {"metric": "ut_sweep_points", "value": len(sweep_results)},
        {"metric": "has_2d_result", "value": int(result_2d is not None)},
    ]

    if nmax_results:
        best_nmax = min(nmax_results, key=lambda r: r["energy"])
        rows.extend([
            {"metric": "nmax_best_n_max", "value": best_nmax["n_max"]},
            {"metric": "nmax_best_energy", "value": best_nmax["energy"]},
            {"metric": "nmax_best_rel_err", "value": _relative_error(best_nmax["energy"], best_nmax.get("E_ed", None))},
            {"metric": "nmax_best_sector_dim", "value": best_nmax.get("sector_dim", float("nan"))},
            {"metric": "nmax_best_reduction_factor", "value": best_nmax.get("reduction_factor", float("nan"))},
        ])

    if sweep_results:
        min_dn2 = min(sweep_results, key=lambda r: r["delta_n2"])
        max_dn2 = max(sweep_results, key=lambda r: r["delta_n2"])
        rows.extend([
            {"metric": "sweep_min_delta_n2", "value": min_dn2["delta_n2"]},
            {"metric": "sweep_min_delta_n2_U_over_t", "value": min_dn2["U_over_t"]},
            {"metric": "sweep_max_delta_n2", "value": max_dn2["delta_n2"]},
            {"metric": "sweep_max_delta_n2_U_over_t", "value": max_dn2["U_over_t"]},
        ])

    if result_2d:
        rows.extend([
            {"metric": "result_2d_energy", "value": result_2d["energy"]},
            {"metric": "result_2d_rel_err", "value": _relative_error(result_2d["energy"], result_2d.get("E_ed", None))},
            {"metric": "result_2d_delta_n2", "value": result_2d.get("delta_n2", float("nan"))},
        ])

    with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows(rows)

    return summary_csv_path


# ═══════════════════════════════════════════════════════════════
# ROBUST OVERRIDES FOR PORTFOLIO-QUALITY 1D/2D BENCHMARKS
# ═══════════════════════════════════════════════════════════════


def _tree_copy(tree):
    """Best-effort deep copy for parameter pytrees."""
    try:
        import copy
        return copy.deepcopy(tree)
    except Exception:
        return tree


def _set_global_seed(seed):
    """Best-effort process-local seed for reproducibility."""
    if seed is None:
        return
    try:
        seed = int(seed)
    except Exception:
        return
    try:
        np.random.seed(seed % (2**32 - 1))
    except Exception:
        pass
    try:
        import random
        random.seed(seed)
    except Exception:
        pass


def _aggregate_run_group(runs):
    """Aggregate a list of compatible run dictionaries.

    The returned dict keeps the *best* run (lowest energy) while also attaching
    mean/std summaries across the supplied runs. This is used both for
    restart-selection and for multi-seed reporting.
    """
    if not runs:
        raise ValueError('Cannot aggregate an empty run list')

    best = min(runs, key=lambda r: float(r.get('energy', np.inf)))
    out = dict(best)

    energies = np.asarray([float(r.get('energy', np.nan)) for r in runs], dtype=float)
    finite = np.isfinite(energies)
    finite_energies = energies[finite]
    if finite_energies.size:
        out['energy_mean'] = float(np.mean(finite_energies))
        out['energy_std'] = float(np.std(finite_energies, ddof=1)) if finite_energies.size > 1 else 0.0
    else:
        out['energy_mean'] = float('nan')
        out['energy_std'] = float('nan')
    out['energy_yerr'] = out['energy_std']
    out['seed_count'] = int(len(runs))

    dn2_vals = np.asarray([float(r.get('delta_n2', np.nan)) for r in runs], dtype=float)
    finite_dn2 = dn2_vals[np.isfinite(dn2_vals)]
    if finite_dn2.size:
        out['delta_n2_mean'] = float(np.mean(finite_dn2))
        out['delta_n2_std'] = float(np.std(finite_dn2, ddof=1)) if finite_dn2.size > 1 else 0.0
    else:
        out['delta_n2_mean'] = float('nan')
        out['delta_n2_std'] = float('nan')

    elapsed_vals = np.asarray([float(r.get('elapsed_s', np.nan)) for r in runs], dtype=float)
    finite_elapsed = elapsed_vals[np.isfinite(elapsed_vals)]
    if finite_elapsed.size:
        out['elapsed_mean_s'] = float(np.mean(finite_elapsed))
        out['elapsed_std_s'] = float(np.std(finite_elapsed, ddof=1)) if finite_elapsed.size > 1 else 0.0

    traces = []
    for r in runs:
        tr = np.asarray(r.get('trace', []), dtype=float)
        if tr.size:
            traces.append(tr)
    if traces:
        min_len = min(len(tr) for tr in traces)
        arr = np.vstack([tr[:min_len] for tr in traces])
        out['trace_mean'] = np.mean(arr, axis=0)
        out['trace_std'] = np.std(arr, axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros(min_len)

    out['all_runs'] = runs
    return out


def run_vmc(graph, n_max, U, t_hop, mu=0.0, n_particles=None,
            alpha=4, n_samples=4096, n_iter=600, lr=0.001,
            diag_shift=0.10, sampler_kind='auto', n_chains=32,
            sweep_multiplier=4, n_discard_per_chain=None,
            use_complex_rbm=False, sr_mode='auto', optimizer='adam',
            return_internal=False, fullsum_mode='auto',
            fullsum_threshold=50000, init_parameters=None):
    """Robust VMC ground-state search for Bose–Hubbard.

    Compared with the legacy baseline, this override supports:
      - Adam or SGD when SR is disabled
      - FullSumState on small enumerable sectors
      - warm-starting via ``init_parameters``
    """
    import netket as nk
    import optax

    hi, H = build_bh_netket(graph, n_max, U, t_hop, mu, n_particles)
    N_sites = graph.n_nodes
    full_dim, sector_dim, reduction_factor = hilbert_space_stats(N_sites, n_max, n_particles)

    param_dtype = complex if use_complex_rbm else float
    model = nk.models.RBM(alpha=alpha, param_dtype=param_dtype)

    try:
        n_states = int(getattr(hi, 'n_states'))
    except Exception:
        n_states = None

    allow_fullsum = False
    if fullsum_mode not in {'off', 'auto', 'on'}:
        raise ValueError(f'Unsupported fullsum_mode={fullsum_mode!r}')
    if fullsum_mode == 'on':
        allow_fullsum = True
    elif fullsum_mode == 'auto' and n_states is not None and n_states <= int(fullsum_threshold):
        allow_fullsum = True

    n_chains = int(max(1, n_chains))
    n_samples = _round_up_to_multiple(max(n_samples, n_chains), n_chains)
    vstate, sampler_used, sweep_size, n_discard = make_variational_state(
        hi, H, graph, model,
        n_particles=n_particles, n_samples=n_samples,
        sampler_kind=sampler_kind, n_chains=n_chains,
        sweep_multiplier=sweep_multiplier,
        n_discard_per_chain=n_discard_per_chain,
        fullsum_threshold=fullsum_threshold,
        allow_fullsum=allow_fullsum,
    )

    if init_parameters is not None:
        try:
            vstate.parameters = _tree_copy(init_parameters)
        except Exception:
            pass

    if sr_mode not in {'auto', 'on', 'off'}:
        raise ValueError(f'Unsupported sr_mode={sr_mode!r}')
    use_sr = (sr_mode == 'on') or (sr_mode == 'auto' and sampler_used not in ('exact', 'fullsum'))

    opt_name = str(optimizer).lower()
    if use_sr:
        opt = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(learning_rate=lr),
        )
        preconditioner = nk.optimizer.SR(diag_shift=diag_shift)
        optimizer_name = 'adam+sr'
    else:
        if opt_name == 'sgd':
            opt = optax.sgd(learning_rate=lr)
            optimizer_name = 'sgd'
        else:
            opt = optax.adam(learning_rate=lr)
            optimizer_name = 'adam'
        preconditioner = None

    driver_kwargs = {'variational_state': vstate}
    if preconditioner is not None:
        driver_kwargs['preconditioner'] = preconditioner
    driver = nk.driver.VMC(H, opt, **driver_kwargs)

    log = nk.logging.RuntimeLog()
    t0 = time.time()
    driver.run(n_iter=int(n_iter), out=log)
    elapsed = time.time() - t0

    energy_data = log.data.get('Energy', {})
    if isinstance(energy_data, dict) and 'Mean' in energy_data:
        raw_trace = energy_data['Mean']
    else:
        raw_trace = log['Energy']['Mean']
    energy_trace = np.real(np.asarray(raw_trace))

    E_stats = vstate.expect(H)
    final_e = float(np.real(E_stats.mean))
    err_attr = getattr(E_stats, 'error_of_mean', None)
    if err_attr is None:
        final_std = 0.0
    else:
        try:
            final_std = float(np.real(err_attr))
        except Exception:
            final_std = float(np.real(err_attr()))
    r_hat = _extract_r_hat(E_stats) if sampler_used not in ('fullsum', 'exact') else float('nan')

    samples = np.empty((0, N_sites))
    if sampler_used == 'fullsum':
        exact_e, _, psi_exact = _exact_energy_from_vstate(vstate, H)
        probs = np.abs(psi_exact) ** 2
        states = np.asarray(hi.all_states())
        occ_mean = np.mean(states, axis=1)
        occ2_mean = np.mean(states ** 2, axis=1)
        n_mean = float(np.sum(probs * occ_mean))
        n2_mean = float(np.sum(probs * occ2_mean))
        delta_n2 = float(n2_mean - n_mean ** 2)
        final_e = exact_e
        final_std = 0.0
    else:
        try:
            samples = np.asarray(vstate.samples).reshape(-1, N_sites)
            n_mean = float(np.mean(samples))
            n2_mean = float(np.mean(samples ** 2))
            delta_n2 = float(n2_mean - n_mean ** 2)
        except Exception:
            n_mean = float('nan')
            delta_n2 = float('nan')

    out = {
        'energy': final_e,
        'std': final_std,
        'n_params': int(vstate.n_parameters),
        'trace': energy_trace,
        'elapsed_s': elapsed,
        'n_max': n_max,
        'full_dim': int(full_dim),
        'sector_dim': int(sector_dim if sector_dim is not None else full_dim),
        'reduction_factor': float(reduction_factor),
        'U': U,
        't_hop': t_hop,
        'n_mean': n_mean,
        'delta_n2': delta_n2,
        'n_samples': int(n_samples if sampler_used != 'fullsum' else 0),
        'n_iter': int(n_iter),
        'sampler': sampler_used,
        'optimizer': optimizer_name,
        'sr_enabled': bool(use_sr),
        'n_chains': int(n_chains),
        'sweep_size': int(sweep_size),
        'n_discard_per_chain': int(n_discard),
        'r_hat': r_hat,
        'repeat_frac': float('nan'),
        'parameters': _tree_copy(vstate.parameters),
    }
    if sampler_used != 'fullsum' and samples.size:
        try:
            local_vals = np.asarray(vstate.local_estimators(H)).reshape(-1)
            out['repeat_frac'] = _repeat_fraction(local_vals)
        except Exception:
            pass

    if return_internal:
        out.update({
            'hilbert': hi,
            'hamiltonian': H,
            'vstate': vstate,
            'model': model,
        })

    return out


def _run_with_restarts(*, graph, n_max, U, t_hop, mu=0.0, n_particles=None,
                       alpha=4, n_samples=4096, n_iter=600, lr=0.001,
                       diag_shift=0.10, sampler_kind='auto', n_chains=32,
                       sweep_multiplier=4, n_discard_per_chain=None,
                       use_complex_rbm=False, sr_mode='auto', optimizer='adam',
                       fullsum_mode='auto', fullsum_threshold=50000,
                       n_restarts=1, seed_base=None, warm_start_parameters=None):
    runs = []
    for r_idx in range(int(max(1, n_restarts))):
        run_seed = None if seed_base is None else int(seed_base + r_idx)
        _set_global_seed(run_seed)
        init_params = warm_start_parameters if (r_idx == 0 and warm_start_parameters is not None) else None
        r = run_vmc(
            graph, n_max, U, t_hop, mu=mu, n_particles=n_particles,
            alpha=alpha, n_samples=n_samples, n_iter=n_iter, lr=lr,
            diag_shift=diag_shift, sampler_kind=sampler_kind, n_chains=n_chains,
            sweep_multiplier=sweep_multiplier, n_discard_per_chain=n_discard_per_chain,
            use_complex_rbm=use_complex_rbm, sr_mode=sr_mode, optimizer=optimizer,
            fullsum_mode=fullsum_mode, fullsum_threshold=fullsum_threshold,
            init_parameters=init_params,
        )
        r['restart_index'] = int(r_idx + 1)
        r['restart_count'] = int(max(1, n_restarts))
        r['warm_start_used'] = bool(init_params is not None)
        r['run_seed'] = run_seed
        runs.append(r)
    return _aggregate_run_group(runs)


def _format_seed_stats(r):
    if int(r.get('seed_count', 1)) <= 1:
        return ''
    return f", seed mean={r.get('energy_mean', float('nan')):.6f} ± {r.get('energy_std', float('nan')):.2e}"


def nmax_convergence(N, U, t_hop, alpha=4, n_iter=300, n_samples=4096,
                     lr=0.001, diag_shift=0.10, sampler_kind='auto',
                     n_chains=32, sweep_multiplier=4, n_discard_per_chain=None,
                     use_complex_rbm=False, sr_mode='auto', optimizer='adam',
                     fullsum_mode='auto', fullsum_threshold=50000,
                     n_restarts=5, n_seeds=1, seed=1234, seed_stride=1000,
                     alpha_nmax=None, n_iter_nmax=None, lr_nmax=None):
    """Sweep n_max = 1,2,3,4 using stronger defaults and warm-starts.

    Publication-oriented behavior:
      - warm-start from n_max -> n_max+1 within each seed
      - multiple restarts per point, keep the lowest energy
      - optional multiple seeds for error bars
    """
    import netket as nk

    alpha_eff = int(alpha_nmax if alpha_nmax is not None else max(alpha, 8))
    n_iter_eff = int(n_iter_nmax if n_iter_nmax is not None else max(int(n_iter), 1800))
    lr_eff = float(lr_nmax if lr_nmax is not None else max(float(lr), 3e-3))

    print(f"  n_max convergence: N={N}, U/t={U/t_hop:.1f}")
    print(f"  Robust settings: alpha={alpha_eff}, n_iter={n_iter_eff}, lr={lr_eff:.3g}, restarts={int(max(1, n_restarts))}, seeds={int(max(1, n_seeds))}")
    g = nk.graph.Chain(length=N, pbc=True)

    trivial = {
        'energy': 0.0, 'std': float('nan'), 'n_params': 0,
        'trace': np.asarray([0.0]), 'elapsed_s': 0.0,
        'n_max': 1, 'full_dim': hilbert_space_stats(N, 1, N)[0],
        'sector_dim': hilbert_space_stats(N, 1, N)[1],
        'reduction_factor': hilbert_space_stats(N, 1, N)[2],
        'U': U, 't_hop': t_hop, 'n_mean': 1.0, 'delta_n2': 0.0,
        'E_ed': None, 'n_samples': 0, 'n_iter': 0, 'sampler': 'trivial',
        'optimizer': 'none', 'sr_enabled': False, 'r_hat': float('nan'),
        'repeat_frac': float('nan'), 'seed_count': int(max(1, n_seeds)),
        'energy_mean': 0.0, 'energy_std': 0.0, 'energy_yerr': 0.0,
    }

    per_nmax = {1: [trivial]}
    prev_by_seed = {}

    for seed_idx in range(int(max(1, n_seeds))):
        seed_offset = int(seed) + seed_idx * int(seed_stride)
        prev_params = None
        for n_max in [2, 3, 4]:
            E_ed = ed_reference(g, n_max, U, t_hop, n_particles=N)
            result = _run_with_restarts(
                graph=g, n_max=n_max, U=U, t_hop=t_hop, n_particles=N,
                alpha=alpha_eff, n_samples=n_samples, n_iter=n_iter_eff, lr=lr_eff,
                diag_shift=diag_shift, sampler_kind=sampler_kind, n_chains=n_chains,
                sweep_multiplier=sweep_multiplier, n_discard_per_chain=n_discard_per_chain,
                use_complex_rbm=use_complex_rbm, sr_mode=sr_mode, optimizer=optimizer,
                fullsum_mode=fullsum_mode, fullsum_threshold=fullsum_threshold,
                n_restarts=n_restarts, seed_base=seed_offset + 100 * n_max,
                warm_start_parameters=prev_params,
            )
            result['E_ed'] = E_ed
            result['n_max'] = n_max
            result['seed_index'] = int(seed_idx + 1)
            prev_params = _tree_copy(result.get('parameters'))
            prev_by_seed[(seed_idx, n_max)] = prev_params
            per_nmax.setdefault(n_max, []).append(result)

    results = [trivial]
    for n_max in [2, 3, 4]:
        seed_runs = per_nmax.get(n_max, [])
        agg = _aggregate_run_group(seed_runs)
        agg['n_max'] = n_max
        agg['U'] = U
        agg['t_hop'] = t_hop
        agg['E_ed'] = ed_reference(g, n_max, U, t_hop, n_particles=N)
        results.append(agg)

        local_dim = n_max + 1
        print(f"\n    n_max={n_max}  (local dim={local_dim})")
        print(f"      {format_hilbert_space_stats(N, n_max, N)}")
        if agg['E_ed'] is not None:
            print(f"      ED  E0 = {agg['E_ed']:.8f}  (E0/N = {agg['E_ed']/N:.8f})")
        rel_err = _relative_error(agg['energy'], agg.get('E_ed'))
        print(
            f"      NQS E0 = {agg['energy']:.8f} ± {agg['std']:.2e}  "
            f"(best seed {agg.get('seed_index', 1)}/{agg.get('seed_count', 1)}, "
            f"restart {agg.get('restart_index', 1)}/{agg.get('restart_count', 1)}, "
            f"{agg['n_params']} params, {agg['elapsed_s']:.1f}s, {agg['sampler']})"
        )
        if int(agg.get('seed_count', 1)) > 1:
            print(f"      seed mean E0 = {agg.get('energy_mean', float('nan')):.8f} ± {agg.get('energy_std', float('nan')):.2e}")
        if agg['E_ed'] is not None:
            print(f"      rel err = {rel_err:.2e}")

    return results


def ut_sweep(N, n_max=3, alpha=4, n_iter=200, n_samples=4096,
             lr=0.001, diag_shift=0.10, sampler_kind='auto',
             n_chains=32, sweep_multiplier=4, n_discard_per_chain=None,
             use_complex_rbm=False, sr_mode='auto', optimizer='adam',
             fullsum_mode='auto', fullsum_threshold=50000,
             n_restarts=5, n_seeds=1, seed=1234, seed_stride=1000):
    """Sweep U/t with warm-starting across the ordered interaction points."""
    import netket as nk

    alpha_eff = int(max(alpha, 8))
    n_iter_eff = int(max(int(n_iter), 1200))
    lr_eff = float(max(float(lr), 1e-2))

    g = nk.graph.Chain(length=N, pbc=True)
    t_hop = 1.0
    U_values = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]

    print(f"  U/t sweep: N={N}, n_max={n_max}, t={t_hop}")
    print(f"  {format_hilbert_space_stats(N, n_max, N)}")
    print(f"  Robust settings: alpha={alpha_eff}, n_iter={n_iter_eff}, lr={lr_eff:.3g}, restarts={int(max(1, n_restarts))}, seeds={int(max(1, n_seeds))}")

    per_u = {float(U): [] for U in U_values}
    for seed_idx in range(int(max(1, n_seeds))):
        seed_offset = int(seed) + seed_idx * int(seed_stride)
        prev_params = None
        for U_now in U_values:
            result = _run_with_restarts(
                graph=g, n_max=n_max, U=U_now, t_hop=t_hop, n_particles=N,
                alpha=alpha_eff, n_samples=n_samples, n_iter=n_iter_eff, lr=lr_eff,
                diag_shift=diag_shift, sampler_kind=sampler_kind, n_chains=n_chains,
                sweep_multiplier=sweep_multiplier, n_discard_per_chain=n_discard_per_chain,
                use_complex_rbm=use_complex_rbm, sr_mode=sr_mode, optimizer=optimizer,
                fullsum_mode=fullsum_mode, fullsum_threshold=fullsum_threshold,
                n_restarts=n_restarts, seed_base=seed_offset + int(round(U_now * 100)),
                warm_start_parameters=prev_params,
            )
            result['E_ed'] = ed_reference(g, n_max, U_now, t_hop, n_particles=N)
            result['U_over_t'] = U_now / t_hop
            result['seed_index'] = int(seed_idx + 1)
            per_u[float(U_now)].append(result)
            prev_params = _tree_copy(result.get('parameters'))

    results = []
    for U_now in U_values:
        print(f"\n    U/t = {U_now/t_hop:.1f}:")
        agg = _aggregate_run_group(per_u[float(U_now)])
        agg['E_ed'] = ed_reference(g, n_max, U_now, t_hop, n_particles=N)
        agg['U_over_t'] = U_now / t_hop
        results.append(agg)
        if agg['E_ed'] is not None:
            print(f"      ED  E0/N = {agg['E_ed']/N:.6f}")
        print(
            f"      NQS E0/N = {agg['energy']/N:.6f} ± {agg['std']/N:.2e}  "
            f"⟨δn²⟩ = {agg['delta_n2']:.4f}  "
            f"(best seed {agg.get('seed_index', 1)}/{agg.get('seed_count', 1)}, "
            f"restart {agg.get('restart_index', 1)}/{agg.get('restart_count', 1)}, {agg['sampler']})"
        )
        if int(agg.get('seed_count', 1)) > 1:
            print(f"      seed mean E0/N = {agg.get('energy_mean', float('nan'))/N:.6f} ± {agg.get('energy_std', float('nan'))/N:.2e}")

    return results


def run_2d(Lx, Ly, U, t_hop, n_max=3, alpha=4, n_iter=600, n_samples=8192,
           lr=0.0005, diag_shift=0.10, sampler_kind='auto', n_chains=32,
           sweep_multiplier=4, n_discard_per_chain=None,
           use_complex_rbm=False, sr_mode='auto', optimizer='adam',
           fullsum_mode='auto', fullsum_threshold=50000,
           n_restarts=5, n_seeds=5, seed=1234, seed_stride=1000):
    """Robust 2D square-lattice run with best-of-restarts and optional seed bars."""
    import netket as nk

    N_sites = Lx * Ly
    try:
        g = nk.graph.Grid(extent=[Lx, Ly], pbc=True)
    except TypeError:
        g = nk.graph.Grid([Lx, Ly], pbc=True)

    alpha_eff = int(max(alpha, 8))
    n_iter_eff = int(max(int(n_iter), 2500))
    lr_eff = float(max(float(lr), 3e-3))

    print(f"  2D lattice: {Lx}×{Ly} = {N_sites} sites, n_max={n_max}, U/t={U/t_hop:.1f}, PBC")
    print(f"  {format_hilbert_space_stats(N_sites, n_max, N_sites)}")
    print(f"  Robust settings: alpha={alpha_eff}, n_iter={n_iter_eff}, lr={lr_eff:.3g}, restarts={int(max(1, n_restarts))}, seeds={int(max(1, n_seeds))}")

    E_ed = ed_reference(g, n_max, U, t_hop, n_particles=N_sites)
    if E_ed is not None:
        print(f"  ED  E0 = {E_ed:.8f}  (E0/N = {E_ed/N_sites:.8f})")

    seed_runs = []
    for seed_idx in range(int(max(1, n_seeds))):
        seed_offset = int(seed) + seed_idx * int(seed_stride)
        r = _run_with_restarts(
            graph=g, n_max=n_max, U=U, t_hop=t_hop, n_particles=N_sites,
            alpha=alpha_eff, n_samples=n_samples, n_iter=n_iter_eff, lr=lr_eff,
            diag_shift=diag_shift, sampler_kind=sampler_kind, n_chains=n_chains,
            sweep_multiplier=sweep_multiplier, n_discard_per_chain=n_discard_per_chain,
            use_complex_rbm=use_complex_rbm, sr_mode=sr_mode, optimizer=optimizer,
            fullsum_mode=fullsum_mode, fullsum_threshold=fullsum_threshold,
            n_restarts=n_restarts, seed_base=seed_offset + 777,
        )
        r['seed_index'] = int(seed_idx + 1)
        seed_runs.append(r)

    agg = _aggregate_run_group(seed_runs)
    agg['E_ed'] = E_ed
    agg['Lx'] = Lx
    agg['Ly'] = Ly
    rel_err = _relative_error(agg['energy'], E_ed)
    print(
        f"  NQS E0 = {agg['energy']:.8f} ± {agg['std']:.2e}  "
        f"(best seed {agg.get('seed_index', 1)}/{agg.get('seed_count', 1)}, "
        f"restart {agg.get('restart_index', 1)}/{agg.get('restart_count', 1)}, "
        f"{agg['n_params']} params, {agg['elapsed_s']:.1f}s, {agg['sampler']})"
    )
    if int(agg.get('seed_count', 1)) > 1:
        print(f"  seed mean E0 = {agg.get('energy_mean', float('nan')):.8f} ± {agg.get('energy_std', float('nan')):.2e}")
    if E_ed is not None:
        print(f"  rel err = {rel_err:.2e}")
    return agg


def make_figure(nmax_results, sweep_results, result_2d, N):
    """Create only the panels that were executed, with seed-derived error bars."""
    panel_kinds = []
    if nmax_results:
        panel_kinds.append('nmax')
    if sweep_results:
        panel_kinds.append('sweep')
    if result_2d is not None:
        panel_kinds.append('2d')
    if not panel_kinds:
        return

    fig, axes = plt.subplots(1, len(panel_kinds), figsize=(6.0 * len(panel_kinds), 5.2))
    if len(panel_kinds) == 1:
        axes = [axes]

    for idx, (ax, kind) in enumerate(zip(axes, panel_kinds)):
        panel_label = f"({chr(ord('a') + idx)})"

        if kind == 'nmax':
            nmax_vals = [r['n_max'] for r in nmax_results]
            e_nqs = [r['energy'] for r in nmax_results]
            e_yerr = [0.0 if r['n_max'] == 1 else float(r.get('energy_yerr', 0.0)) for r in nmax_results]
            e_ed = [r.get('E_ed') for r in nmax_results]

            ax.errorbar(nmax_vals, e_nqs, yerr=e_yerr, fmt='o-', color='#2196F3', ms=8, lw=1.5, capsize=4, label='NQS (best)')
            ed_x = [x for x, e in zip(nmax_vals, e_ed) if e is not None]
            ed_y = [e for e in e_ed if e is not None]
            if ed_y:
                ax.plot(ed_x, ed_y, 'kx', ms=10, mew=2, label='ED exact')

            U_over_t = nmax_results[0]['U'] / nmax_results[0]['t_hop']
            ax.set_xlabel('$n_{\mathrm{max}}$ (Fock truncation)', fontsize=12)
            ax.set_ylabel('Ground-state energy $E_0$', fontsize=12)
            ax.set_title(f"{panel_label} $n_{{\max}}$ convergence\nN={N}, U/t={U_over_t:.0f}", fontsize=11)
            ax.set_xticks(nmax_vals)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        elif kind == 'sweep':
            ut_vals = [r['U_over_t'] for r in sweep_results]
            e_per_site = [r['energy'] / N for r in sweep_results]
            e_per_site_err = [float(r.get('energy_yerr', 0.0)) / N for r in sweep_results]
            dn2_vals = [r['delta_n2'] for r in sweep_results]
            dn2_err = [float(r.get('delta_n2_std', 0.0)) for r in sweep_results]

            color1 = '#2196F3'
            ax.errorbar(ut_vals, e_per_site, yerr=e_per_site_err, fmt='o-', color=color1, ms=7, lw=1.5, capsize=3)
            ax.set_xlabel('$U/t$', fontsize=12)
            ax.set_ylabel('$E_0 / N$', fontsize=12, color=color1)
            ax.tick_params(axis='y', labelcolor=color1)
            ax.set_xscale('log')

            ax2 = ax.twinx()
            color2 = '#E53935'
            ax2.errorbar(ut_vals, dn2_vals, yerr=dn2_err, fmt='s--', color=color2, ms=7, lw=1.5, capsize=3)
            ax2.set_ylabel(r'$\langle\delta n^2\rangle$  (density fluct.)', fontsize=11, color=color2)
            ax2.tick_params(axis='y', labelcolor=color2)

            ax.set_title(f"{panel_label} Mott–SF crossover\nN={N}", fontsize=11)
            ax.grid(True, alpha=0.3)

            ed_pts = [(r['U_over_t'], r['E_ed'] / N) for r in sweep_results if r.get('E_ed') is not None]
            if ed_pts:
                ax.plot([p[0] for p in ed_pts], [p[1] for p in ed_pts], 'kx', ms=9, mew=2, label='ED', zorder=5)
                ax.legend(fontsize=9, loc='lower right')

        elif kind == '2d':
            trace = np.asarray(result_2d.get('trace_mean', result_2d.get('trace', [])), dtype=float)
            trace_std = np.asarray(result_2d.get('trace_std', np.zeros_like(trace)), dtype=float)
            x = np.arange(trace.size)
            ax.plot(x, trace, color='#4CAF50', lw=1.4, label='Seed mean' if result_2d.get('seed_count', 1) > 1 else 'VMC')
            if trace.size and trace_std.size == trace.size and result_2d.get('seed_count', 1) > 1:
                ax.fill_between(x, trace - trace_std, trace + trace_std, color='#4CAF50', alpha=0.20, lw=0)
            if result_2d.get('E_ed') is not None:
                ax.axhline(result_2d['E_ed'], color='k', ls='--', lw=1, alpha=0.5, label=f"ED = {result_2d['E_ed']:.4f}")
            if result_2d.get('seed_count', 1) > 1 and np.isfinite(result_2d.get('energy_mean', np.nan)):
                emean = result_2d.get('energy_mean', np.nan)
                estd = result_2d.get('energy_std', 0.0)
                ax.axhspan(emean - estd, emean + estd, color='#4CAF50', alpha=0.10, label=f"seed final ±σ = {emean:.4f} ± {estd:.2e}")
            Lx, Ly = result_2d.get('Lx', '?'), result_2d.get('Ly', '?')
            U_over_t = result_2d['U'] / result_2d['t_hop']
            ax.set_xlabel('VMC iteration', fontsize=12)
            ax.set_ylabel('Energy', fontsize=12)
            ax.set_title(f"{panel_label} 2D lattice {Lx}×{Ly}\nU/t={U_over_t:.0f}, $n_{{\max}}$={result_2d['n_max']}", fontsize=11)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

    plt.suptitle('Bose–Hubbard ground state with NQS (§1.4)', fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig('bose_hubbard_nqs.png', dpi=150, bbox_inches='tight')
    plt.savefig('bose_hubbard_nqs.pdf', bbox_inches='tight')
    print('\nSaved: bose_hubbard_nqs.png / .pdf')


def save_results_csv(nmax_results, sweep_results, result_2d, N, Lx, Ly, csv_path=None):
    """Save row-wise results, including seed-derived summary stats."""
    if csv_path is None:
        csv_path = f'bose_hubbard_N{N}_results.csv'

    rows = []

    def _row_from_result(demo, r, N_value, Lx_value='', Ly_value=''):
        return {
            'demo': demo,
            'N': N_value,
            'Lx': Lx_value,
            'Ly': Ly_value,
            'n_max': r.get('n_max', ''),
            'full_dim': r.get('full_dim', ''),
            'sector_dim': r.get('sector_dim', ''),
            'reduction_factor': r.get('reduction_factor', ''),
            'U': r.get('U', ''),
            't_hop': r.get('t_hop', ''),
            'U_over_t': r.get('U_over_t', (r.get('U', float('nan')) / r.get('t_hop', 1.0))),
            'energy': r.get('energy', float('nan')),
            'std': r.get('std', float('nan')),
            'energy_mean': r.get('energy_mean', float('nan')),
            'energy_std': r.get('energy_std', float('nan')),
            'seed_count': r.get('seed_count', 1),
            'E_ed': r.get('E_ed', float('nan')),
            'rel_err': _relative_error(r.get('energy', float('nan')), r.get('E_ed', None)),
            'n_params': r.get('n_params', ''),
            'elapsed_s': r.get('elapsed_s', ''),
            'elapsed_mean_s': r.get('elapsed_mean_s', float('nan')),
            'n_mean': r.get('n_mean', float('nan')),
            'delta_n2': r.get('delta_n2', float('nan')),
            'delta_n2_mean': r.get('delta_n2_mean', float('nan')),
            'delta_n2_std': r.get('delta_n2_std', float('nan')),
            'sampler': r.get('sampler', ''),
            'optimizer': r.get('optimizer', ''),
            'sr_enabled': r.get('sr_enabled', ''),
            'r_hat': r.get('r_hat', float('nan')),
            'repeat_frac': r.get('repeat_frac', float('nan')),
            'best_seed': r.get('seed_index', ''),
            'best_restart': r.get('restart_index', ''),
        }

    for r in nmax_results:
        rows.append(_row_from_result('nmax_convergence', r, N))
    for r in sweep_results:
        rows.append(_row_from_result('ut_sweep', r, N))
    if result_2d is not None:
        rows.append(_row_from_result('2d', result_2d, int(Lx) * int(Ly), Lx, Ly))

    fieldnames = [
        'demo', 'N', 'Lx', 'Ly', 'n_max', 'full_dim', 'sector_dim', 'reduction_factor',
        'U', 't_hop', 'U_over_t', 'energy', 'std', 'energy_mean', 'energy_std', 'seed_count',
        'E_ed', 'rel_err', 'n_params', 'elapsed_s', 'elapsed_mean_s', 'n_mean', 'delta_n2',
        'delta_n2_mean', 'delta_n2_std', 'sampler', 'optimizer', 'sr_enabled', 'r_hat',
        'repeat_frac', 'best_seed', 'best_restart',
    ]
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def save_summary_csv(nmax_results, sweep_results, result_2d, N, U, t_hop, summary_csv_path=None):
    """Save compact summary metrics, now including seed statistics."""
    if summary_csv_path is None:
        summary_csv_path = f'bose_hubbard_N{N}_summary.csv'

    rows = [
        {'metric': 'N_1d', 'value': N},
        {'metric': 'U_over_t_input', 'value': U / t_hop},
        {'metric': 'nmax_points', 'value': len(nmax_results)},
        {'metric': 'ut_sweep_points', 'value': len(sweep_results)},
        {'metric': 'has_2d_result', 'value': int(result_2d is not None)},
    ]

    if nmax_results:
        valid = [r for r in nmax_results if r.get('n_max', 0) > 1]
        if valid:
            best_nmax = min(valid, key=lambda r: r['energy'])
            worst_rel = max((_relative_error(r['energy'], r.get('E_ed')) for r in valid if r.get('E_ed') is not None), default=float('nan'))
            rows.extend([
                {'metric': 'nmax_best_n_max', 'value': best_nmax['n_max']},
                {'metric': 'nmax_best_energy', 'value': best_nmax['energy']},
                {'metric': 'nmax_best_rel_err', 'value': _relative_error(best_nmax['energy'], best_nmax.get('E_ed'))},
                {'metric': 'nmax_worst_rel_err', 'value': worst_rel},
            ])

    if sweep_results:
        min_dn2 = min(sweep_results, key=lambda r: r['delta_n2'])
        max_dn2 = max(sweep_results, key=lambda r: r['delta_n2'])
        rows.extend([
            {'metric': 'sweep_min_delta_n2', 'value': min_dn2['delta_n2']},
            {'metric': 'sweep_min_delta_n2_U_over_t', 'value': min_dn2['U_over_t']},
            {'metric': 'sweep_max_delta_n2', 'value': max_dn2['delta_n2']},
            {'metric': 'sweep_max_delta_n2_U_over_t', 'value': max_dn2['U_over_t']},
        ])

    if result_2d:
        rows.extend([
            {'metric': 'result_2d_energy', 'value': result_2d['energy']},
            {'metric': 'result_2d_seed_mean', 'value': result_2d.get('energy_mean', float('nan'))},
            {'metric': 'result_2d_seed_std', 'value': result_2d.get('energy_std', float('nan'))},
            {'metric': 'result_2d_rel_err', 'value': _relative_error(result_2d['energy'], result_2d.get('E_ed'))},
            {'metric': 'result_2d_delta_n2', 'value': result_2d.get('delta_n2', float('nan'))},
        ])

    with open(summary_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['metric', 'value'])
        writer.writeheader()
        writer.writerows(rows)
    return summary_csv_path


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description='Bose–Hubbard ground state with NQS (§1.4)')
    parser.add_argument('--N', type=int, default=6, help='1D chain length (default: 6)')
    parser.add_argument('--U', type=float, default=10.0, help='On-site interaction for n_max study / 2D')
    parser.add_argument('--t-hop', type=float, default=1.0, help='Hopping amplitude')
    parser.add_argument('--alpha', type=int, default=8, help='RBM hidden density (robust default: 8)')
    parser.add_argument('--alpha-nmax', type=int, default=None, help='Optional stronger alpha for panel (a)')
    parser.add_argument('--lr', type=float, default=3e-3, help='Learning rate for demo 1 / validation (robust default: 3e-3)')
    parser.add_argument('--lr-sweep', type=float, default=1e-2, help='Learning rate for demo 2 (robust default: 1e-2)')
    parser.add_argument('--lr-2d', type=float, default=3e-3, help='Learning rate for demo 3 (robust default: 3e-3)')
    parser.add_argument('--lr-nmax', type=float, default=None, help='Optional stronger learning rate for panel (a)')
    parser.add_argument('--diag-shift', type=float, default=0.10, help='SR diagonal shift')
    parser.add_argument('--n-iter', type=int, default=1800, help='VMC iterations for demo 1 (robust default: 1800)')
    parser.add_argument('--n-iter-sweep', type=int, default=1200, help='VMC iterations for demo 2 (robust default: 1200)')
    parser.add_argument('--n-iter-2d', type=int, default=2500, help='VMC iterations for demo 3 (robust default: 2500)')
    parser.add_argument('--n-iter-nmax', type=int, default=None, help='Optional stronger iteration budget for panel (a)')
    parser.add_argument('--n-samples', type=int, default=4096, help='MC samples for demo 1')
    parser.add_argument('--n-samples-sweep', type=int, default=4096, help='MC samples for demo 2')
    parser.add_argument('--n-samples-2d', type=int, default=8192, help='MC samples for demo 3')
    parser.add_argument('--Lx', type=int, default=3, help='2D lattice width')
    parser.add_argument('--Ly', type=int, default=3, help='2D lattice height')
    parser.add_argument('--mode', choices=['full', '2d', 'sweep-only', 'validate'], default='full', help='Run mode')
    parser.add_argument('--sampler', choices=['auto', 'exact', 'hamiltonian', 'exchange', 'local'], default='auto', help='Sampler choice for MCState')
    parser.add_argument('--optimizer', choices=['adam', 'sgd'], default='adam', help='Optimizer when SR is disabled')
    parser.add_argument('--restarts', type=int, default=5, help='Independent restarts per point (robust default: 5)')
    parser.add_argument('--n-seeds', type=int, default=1, help='Independent seeds for seed mean/std reporting')
    parser.add_argument('--seed', type=int, default=1234, help='Base random seed')
    parser.add_argument('--seed-stride', type=int, default=1000, help='Seed increment between independent seeds')
    parser.add_argument('--n-chains', type=int, default=32, help='Number of Markov chains')
    parser.add_argument('--sweep-multiplier', type=int, default=4, help='Sampler sweep size = multiplier × N_sites')
    parser.add_argument('--n-discard-per-chain', type=int, default=None, help='Thermalization discards per chain (default: auto)')
    parser.add_argument('--complex-rbm', action='store_true', help='Force complex RBM parameters')
    parser.add_argument('--real-rbm', action='store_true', help='Force real RBM parameters (robust defaults use complex)')
    parser.add_argument('--sr-mode', choices=['auto', 'on', 'off'], default='auto', help='Stochastic reconfiguration mode (auto disables SR for exact/fullsum states)')
    parser.add_argument('--fullsum-mode', choices=['off', 'auto', 'on'], default='auto', help='Use FullSumState on small enumerable sectors')
    parser.add_argument('--fullsum-threshold', type=int, default=50000, help='Maximum basis size for auto FullSumState / exact precompute')
    parser.add_argument('--quick', action='store_true', help='Quick test with fewer iterations and samples')
    parser.add_argument('--results-csv-path', type=str, default=None, help='Optional path for detailed results CSV')
    parser.add_argument('--summary-csv-path', type=str, default=None, help='Optional path for summary CSV')
    args = parser.parse_args()

    if args.quick:
        n_iter = min(args.n_iter, 600)
        n_iter_sweep = min(args.n_iter_sweep, 400)
        n_iter_2d = min(args.n_iter_2d, 800)
        n_samples = min(args.n_samples, 1024)
        n_samples_sweep = min(args.n_samples_sweep, 1024)
        n_samples_2d = min(args.n_samples_2d, 2048)
    else:
        n_iter = args.n_iter
        n_iter_sweep = args.n_iter_sweep
        n_iter_2d = args.n_iter_2d
        n_samples = args.n_samples
        n_samples_sweep = args.n_samples_sweep
        n_samples_2d = args.n_samples_2d

    use_complex = bool(args.complex_rbm or (not args.real_rbm and args.mode in ('full', '2d', 'sweep-only')))

    if args.mode == 'validate':
        tiny_deterministic_validation(
            alpha=max(args.alpha, 4),
            n_iter=min(n_iter, 200),
            n_samples=min(max(n_samples, 2048), 8192),
            lr=args.lr,
            diag_shift=args.diag_shift,
            sampler_kind=args.sampler,
            n_chains=args.n_chains,
            sweep_multiplier=max(args.sweep_multiplier, 8),
            n_discard_per_chain=args.n_discard_per_chain,
            use_complex_rbm=use_complex,
            sr_mode=args.sr_mode,
        )
        return

    N = args.N
    print(f'Bose–Hubbard Ground State: N={N}, U/t={args.U/args.t_hop:.1f}')
    print('Canonical ensemble (unit filling: n_particles = N_sites)')
    print('U(1) number-sector projection is enabled throughout the demos.')
    print()

    nmax_results = []
    sweep_results = []
    result_2d = None

    if args.mode == 'full':
        print('=' * 55)
        print('DEMO 1: Fock-space truncation convergence (§1.4)')
        print('=' * 55)
        print(f'  Fixed: N={N}, U={args.U}, t={args.t_hop}, unit filling')
        print('  Sweep: n_max = 1, 2, 3, 4')
        print()
        nmax_results = nmax_convergence(
            N, args.U, args.t_hop,
            alpha=args.alpha, alpha_nmax=args.alpha_nmax,
            n_iter=n_iter, n_iter_nmax=args.n_iter_nmax,
            n_samples=n_samples,
            lr=args.lr, lr_nmax=args.lr_nmax,
            diag_shift=args.diag_shift,
            sampler_kind=args.sampler, n_chains=args.n_chains,
            sweep_multiplier=args.sweep_multiplier,
            n_discard_per_chain=args.n_discard_per_chain,
            use_complex_rbm=use_complex,
            sr_mode=args.sr_mode, optimizer=args.optimizer,
            fullsum_mode=args.fullsum_mode,
            fullsum_threshold=args.fullsum_threshold,
            n_restarts=args.restarts, n_seeds=args.n_seeds,
            seed=args.seed, seed_stride=args.seed_stride,
        )
        print()

    if args.mode in ('full', 'sweep-only'):
        print('=' * 55)
        print('DEMO 2: Mott–superfluid crossover')
        print('=' * 55)
        sweep_results = ut_sweep(
            N, n_max=3,
            alpha=args.alpha, n_iter=n_iter_sweep, n_samples=n_samples_sweep,
            lr=args.lr_sweep, diag_shift=args.diag_shift,
            sampler_kind=args.sampler, n_chains=args.n_chains,
            sweep_multiplier=args.sweep_multiplier,
            n_discard_per_chain=args.n_discard_per_chain,
            use_complex_rbm=use_complex,
            sr_mode=args.sr_mode, optimizer=args.optimizer,
            fullsum_mode=args.fullsum_mode,
            fullsum_threshold=args.fullsum_threshold,
            n_restarts=args.restarts, n_seeds=args.n_seeds,
            seed=args.seed + 50_000, seed_stride=args.seed_stride,
        )
        print()

    if args.mode in ('full', '2d'):
        print('=' * 55)
        print('DEMO 3: 2D square lattice (Project 01 geometry)')
        print('=' * 55)
        # For the 3×3 publication target, 5 seeds is a sensible default once the user asks for stronger bars.
        seeds_2d = max(args.n_seeds, 5 if (args.Lx, args.Ly) == (3, 3) and args.mode in ('full', '2d') else 1)
        result_2d = run_2d(
            args.Lx, args.Ly, args.U, args.t_hop,
            n_max=3, alpha=args.alpha,
            n_iter=n_iter_2d, n_samples=n_samples_2d,
            lr=args.lr_2d, diag_shift=args.diag_shift,
            sampler_kind=args.sampler, n_chains=args.n_chains,
            sweep_multiplier=args.sweep_multiplier,
            n_discard_per_chain=args.n_discard_per_chain,
            use_complex_rbm=use_complex,
            sr_mode=args.sr_mode, optimizer=args.optimizer,
            fullsum_mode=args.fullsum_mode,
            fullsum_threshold=args.fullsum_threshold,
            n_restarts=args.restarts, n_seeds=seeds_2d,
            seed=args.seed + 100_000, seed_stride=args.seed_stride,
        )
        print()

    print('=' * 55)
    print('SUMMARY')
    print('=' * 55)

    if nmax_results:
        print(f"\n  n_max convergence (1D, N={N}, U/t={args.U/args.t_hop:.0f}):")
        for r in nmax_results:
            if r['n_max'] == 1:
                print(f"    n_max=1: E0=0.000000 ± nan  (no ED, trivial; sector={r.get('sector_dim', 0):,}, reduction={r.get('reduction_factor', float('nan')):.1f}×)")
                continue
            extra = _format_seed_stats(r)
            print(
                f"    n_max={r['n_max']}: E0={r['energy']:.6f} ± {r['std']:.2e}  "
                f"(ED={r.get('E_ed', float('nan')):.6f}, rel err={_relative_error(r['energy'], r.get('E_ed')):.2e}, "
                f"sector={r.get('sector_dim', float('nan')):,}, red={r.get('reduction_factor', float('nan')):.1f}×, "
                f"sampler={r.get('sampler','?')}, opt={r.get('optimizer','?')}, "
                f"restart={r.get('restart_index', 1)}/{r.get('restart_count', 1)}, R̂={r.get('r_hat', float('nan')):.3f}{extra})"
            )

    if sweep_results:
        print(f"\n  U/t sweep (1D, N={N}, n_max=3):")
        for r in sweep_results:
            extra = _format_seed_stats(r)
            print(
                f"    U/t={r['U_over_t']:5.1f}: E0/N={r['energy']/N:.6f}  "
                f"⟨δn²⟩={r['delta_n2']:.4f}  (sector={r.get('sector_dim', float('nan')):,}, "
                f"red={r.get('reduction_factor', float('nan')):.1f}×, sampler={r.get('sampler','?')}, "
                f"opt={r.get('optimizer','?')}, restart={r.get('restart_index', 1)}/{r.get('restart_count', 1)}, "
                f"R̂={r.get('r_hat', float('nan')):.3f}{extra})"
            )

    if result_2d is not None:
        print(f"\n  2D lattice ({args.Lx}×{args.Ly}):")
        extra = _format_seed_stats(result_2d)
        print(
            f"    E0 = {result_2d['energy']:.6f} ± {result_2d['std']:.2e}  "
            f"(sector={result_2d.get('sector_dim', float('nan')):,}, red={result_2d.get('reduction_factor', float('nan')):.1f}×, "
            f"sampler={result_2d.get('sampler','?')}, opt={result_2d.get('optimizer','?')}, "
            f"restart={result_2d.get('restart_index', 1)}/{result_2d.get('restart_count', 1)}, R̂={result_2d.get('r_hat', float('nan')):.3f}{extra})"
        )
        if result_2d.get('E_ed') is not None:
            rel_err = _relative_error(result_2d['energy'], result_2d['E_ed'])
            print(f"    ED = {result_2d['E_ed']:.6f}  (rel err = {rel_err:.2e})")

    results_csv = save_results_csv(
        nmax_results, sweep_results, result_2d, N, args.Lx, args.Ly,
        csv_path=args.results_csv_path,
    )
    summary_csv = save_summary_csv(
        nmax_results, sweep_results, result_2d, N, args.U, args.t_hop,
        summary_csv_path=args.summary_csv_path,
    )
    print(f"\nSaved: {results_csv}")
    print(f"Saved: {summary_csv}")

    if nmax_results or sweep_results or result_2d:
        make_figure(nmax_results, sweep_results, result_2d, N)


if __name__ == '__main__':
    main()
