"""
Triangle Comparison: Equivariant VQC vs NQS vs DMRG vs ED
==========================================================
1D Heisenberg model, N=10, PBC.

Produces a "signature figure" comparing:
  1. Equivariant VQC      — existing PennyLane results (2 params/layer)
  2. Non-equivariant VQC  — existing results (7 params/layer)
  3. NQS / RBM            — NetKet VMC results (sweep alpha)
  4. DMRG                 — TeNPy finite DMRG (sweep bond dim chi)
  5. ED exact             — ground truth

Figure design:
  - three panels sharing the x-axis (real variational parameter count):
      Left:   |E_var - E_ED| (accuracy vs cost)
      Centre: absolute energy with error bars
      Right:  half-chain entanglement error ΔS_{N/2} = S_{N/2} - S_{N/2}^{exact}
  - the third panel is the "bridge" diagnostic: it shows *why* the
    accuracy curves in the left panel look the way they do.
    DMRG captures full entanglement even at modest chi because 1D
    Heisenberg ground states obey an area law.  NQS converges to the
    correct entanglement with increasing alpha.  This connects the
    methods conceptually, not just numerically.

Convention note:
  All methods use the Pauli convention: H = J * sum sigma_i . sigma_{i+1}.
  NetKet: H = J * (XX + YY + ZZ) with Pauli matrices — J=1 directly.
  TeNPy: We use a custom model with SpinHalfSite Pauli operators
         (Sp, Sm, Sigmaz) so J=1 matches the Pauli convention directly,
         avoiding the spin-1/2 <-> Pauli factor-of-4 pitfall.

  Why NOT XXZChain:
    1. XXZChain silently ignores `bc_x` → always runs OBC.
    2. XXZChain uses spin-1/2 operators (S = sigma/2), requiring
       Jxx=Jz=4 to match Pauli convention — easy to get wrong.

Requirements:
  - numpy, matplotlib (required)
  - netket >= 3.0, Python >= 3.11 (for NQS runs)
  - tenpy (pip install physics-tenpy) (for DMRG)
"""

from math import comb

import matplotlib.pyplot as plt
import numpy as np

# ═══════════════════════════════════════════════════════════════
# YOUR EXISTING VQE DATA — paste from heisenberg-model.ipynb
# ═══════════════════════════════════════════════════════════════

N_SITES = 10
E_EXACT = -18.06178542  # ED ground state for N=10 Heisenberg PBC

P_VALUES = [2, 4, 6, 8, 10, 12]

# Equivariant ansatz: SU(2)-symmetric, 2 params/layer (beta, gamma)
EQ_ENERGIES = [
    -17.097170,
    -17.717170,
    -17.746401,
    -18.009478,
    -18.040247,
    -18.058708,
]  # Replace with exact data_heis_eq["energy"] if available
EQ_STDS = [
    0.033846,
    0.218462,
    0.429231,
    0.081538,
    0.036923,
    0.012308,
]  # Replace with exact data_heis_eq["std"]

# Non-equivariant: 7 params/layer (alpha + 3*beta + 3*gamma)
NEQ_ENERGIES = [
    -17.040247,
    -17.883324,
    -18.037170,
    -18.052555,
    -18.054093,
    -18.024862,
]  # Replace with exact data_heis_neq["energy"]
NEQ_STDS = [
    0.069231,
    0.173846,
    0.016923,
    0.012308,
    0.012308,
    0.067692,
]  # Replace with exact data_heis_neq["std"]

# Edit this to match your actual circuit implementation.
VQC_SYMMETRY_NOTE = "VQC symmetry constraint depends on ansatz implementation."

# VQC entanglement entropies — fill these in from your PennyLane notebook
# by extracting the statevector at each layer depth and computing S_{N/2}.
# Example:
#   state = circuit(params)          # shape (2^N,) or constrained
#   S = compute_half_chain_entropy(state, all_states, N_SITES)
#
# Until then, set to None to omit VQCs from the entanglement panel.
EQ_ENTROPIES = [0.976538, 1.130749, 1.125736, 1.124615, 1.124609, 1.124748]
NEQ_ENTROPIES = [0.951662, 1.13564, 1.127042, 1.124963, 1.125476, 1.124584]


# ═══════════════════════════════════════════════════════════════
# ENTANGLEMENT ENTROPY UTILITIES
# ═══════════════════════════════════════════════════════════════


def compute_half_chain_entropy(amplitudes, all_states, n_sites):
    """
    Half-chain von Neumann entanglement entropy via Schmidt decomposition.

    Given a state |ψ⟩ = Σ_σ ψ(σ)|σ⟩ (possibly in a constrained sector),
    bipartitions into sites {0,...,N/2-1} vs {N/2,...,N-1} and computes
    S = -Tr(ρ_L log ρ_L) from the singular values of the bipartition matrix.

    Parameters
    ----------
    amplitudes : array, shape (n_basis,)
        Wavefunction amplitudes, ordered consistently with `all_states`.
    all_states : array, shape (n_basis, n_sites)
        Spin configurations (entries ±1 or ±0.5).  Each row is one basis state.
    n_sites : int
        Total number of sites.

    Returns
    -------
    float
        Von Neumann entropy S_{N/2}.
    """
    n_left = n_sites // 2
    dim_L = 2**n_left
    dim_R = 2**(n_sites - n_left)

    amps = np.asarray(amplitudes, dtype=complex).ravel()
    amps = amps / np.linalg.norm(amps)

    states = np.asarray(all_states)
    # Map spin eigenvalues → binary: anything > 0 becomes 1, else 0.
    # Works for both {-1, +1} and {-0.5, +0.5} conventions.
    bits = (states > 0).astype(int)

    # Binary-encode each half into an integer index
    pow_L = 2 ** np.arange(n_left - 1, -1, -1)
    pow_R = 2 ** np.arange(n_sites - n_left - 1, -1, -1)
    idx_L = (bits[:, :n_left] @ pow_L).astype(int)
    idx_R = (bits[:, n_left:] @ pow_R).astype(int)

    # Build bipartition matrix  M[i_L, i_R] = ψ(σ_L = i_L, σ_R = i_R)
    # For a constrained sector (e.g. Sz=0) most entries stay zero — correct.
    M = np.zeros((dim_L, dim_R), dtype=complex)
    for k in range(len(amps)):
        M[idx_L[k], idx_R[k]] += amps[k]

    # Schmidt decomposition via SVD
    sv = np.linalg.svd(M, compute_uv=False)
    sv = sv[sv > 1e-15]
    p = sv**2
    p = p / p.sum()
    return float(-np.sum(p * np.log(p)))


# ═══════════════════════════════════════════════════════════════
# ED — exact ground state + entanglement entropy reference
# ═══════════════════════════════════════════════════════════════


def run_ed_entropy(n_sites):
    """
    Exact diagonalization: ground-state energy and half-chain entropy.

    Uses NetKet to build the sparse Hamiltonian in the Sz=0 sector,
    then scipy.sparse.linalg.eigsh for the lowest eigenvalue/vector.
    The entanglement entropy serves as the exact reference in the
    third panel — the diagnostic target that DMRG and NQS must match.

    Returns
    -------
    dict with keys: energy (float), entropy (float)
    """
    import netket as nk
    from scipy.sparse.linalg import eigsh

    g = nk.graph.Chain(length=n_sites, pbc=True)
    hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes, total_sz=0)
    H = nk.operator.Heisenberg(hilbert=hi, graph=g, J=1.0)

    H_sparse = H.to_sparse()
    evals, evecs = eigsh(H_sparse, k=1, which="SA")

    psi0 = evecs[:, 0]
    all_states = np.array(hi.all_states())
    S_half = compute_half_chain_entropy(psi0, all_states, n_sites)

    return {"energy": float(evals[0]), "entropy": S_half}


# ═══════════════════════════════════════════════════════════════
# DMRG GROUND STATE (TeNPy) — custom model for PBC + Pauli convention
# ═══════════════════════════════════════════════════════════════


def run_dmrg_heisenberg(n_sites, chi_max=32, svd_min=1e-10, n_sweeps=20):
    """
    Run finite two-site DMRG for 1D Heisenberg (PBC) using TeNPy.

    Uses a custom CouplingMPOModel to:
      - Set bc='periodic' explicitly via init_lattice() override
        (XXZChain ignores bc_x, always giving OBC).
      - Use Pauli operators (Sp, Sm, Sigmaz) with J=1 directly,
        since: sigma.sigma = 2*(Sp Sm + Sm Sp) + Sigmaz Sigmaz.
        This avoids the S=sigma/2 factor-of-4 pitfall.

    Parameter counting (real DOF):
      MPS tangent-space dimension (gauge-corrected) computed from
      actual bond dimensions of the converged state.

    Entanglement entropy:
      Extracted directly from the Schmidt values at the half-chain
      bond (index N//2 - 1).  For MPS this is essentially free.

    Returns:
        dict with keys: energy, n_real_params, chi_max, chis, entropy
    """
    from tenpy.algorithms import dmrg
    from tenpy.models.lattice import Chain
    from tenpy.models.model import CouplingMPOModel
    from tenpy.networks.mps import MPS
    from tenpy.networks.site import SpinHalfSite

    class HeisenbergPBC(CouplingMPOModel):
        """Heisenberg chain with explicit PBC, Pauli convention."""

        def init_lattice(self, model_params):
            conserve = model_params.get("conserve", "Sz")
            site = SpinHalfSite(conserve=conserve)
            L = model_params["L"]
            bc_MPS = model_params.get("bc_MPS", "finite")
            # bc='periodic' on the Chain lattice ensures the wrap-around
            # bond (N-1, 0) is included in nearest_neighbors pairs.
            return Chain(L, site, bc="periodic", bc_MPS=bc_MPS)

        def init_terms(self, model_params):
            J = model_params.get("J", 1.0)
            # H = J * sum_<ij> sigma_i . sigma_j  (Pauli convention)
            #   = J * sum_<ij> [2*(Sp_i Sm_j + Sm_i Sp_j) + Sigmaz_i Sigmaz_j]
            #
            # Derivation: sigma_x sigma_x + sigma_y sigma_y = 2*(Sp Sm + Sm Sp)
            # where Sp = |up><down|, Sm = |down><up| (= sigma_+, sigma_-)
            for u1, u2, dx in self.lat.pairs["nearest_neighbors"]:
                self.add_coupling(2.0 * J, u1, "Sp", u2, "Sm", dx)
                self.add_coupling(2.0 * J, u1, "Sm", u2, "Sp", dx)
                self.add_coupling(J, u1, "Sigmaz", u2, "Sigmaz", dx)

    model_params = {
        "L": n_sites,
        "J": 1.0,
        "bc_MPS": "finite",
        "conserve": "Sz",
    }
    model = HeisenbergPBC(model_params)

    # Verify PBC correctly: for nearest-neighbor dx=[1], periodic finite chain
    # must include the wrap-around bond (N-1 -> 0), giving exactly N bonds.
    bc = model.lat.boundary_conditions[0]
    mps_i, mps_j, _, _ = model.lat.possible_couplings(0, 0, [1])
    n_bonds = len(mps_i)
    has_wrap = any((int(i) == n_sites - 1 and int(j) == 0) for i, j in zip(mps_i, mps_j))
    assert bc == "periodic" and n_bonds == n_sites and has_wrap, (
        f"Expected periodic NN couplings with {n_sites} bonds including ({n_sites-1},0), "
        f"got bc={bc!r}, n_bonds={n_bonds}, has_wrap={has_wrap}."
    )

    # Néel initial state (Sz=0 sector for even N)
    psi = MPS.from_lat_product_state(model.lat, [["up"], ["down"]])

    dmrg_params = {
        "trunc_params": {
            "chi_max": chi_max,
            "svd_min": svd_min,
        },
        # Finite-MPS + periodic couplings can have larger intermediate
        # truncation errors; keep optimization running and assess final energy.
        "max_trunc_err": 1.0,
        "max_sweeps": n_sweeps,
        "mixer": True,
        "mixer_params": {
            "amplitude": 1e-3,
            "decay": 1.5,
            "disable_after": max(1, n_sweeps - 4),
        },
    }

    eng = dmrg.TwoSiteDMRGEngine(psi, model, dmrg_params)
    E0, psi_opt = eng.run()

    # Count real variational parameters (tangent-space dimension)
    # d_T = sum_i (chi_{i-1} * d * chi_i) - sum_i chi_i^2  [gauge correction]
    # Factor of 2 for complex-valued MPS tensors.
    chis = psi_opt.chi
    d = 2
    raw_params = 0
    gauge_params = 0
    for i in range(n_sites):
        chi_l = chis[i - 1] if i > 0 else chis[-1]
        chi_r = chis[i] if i < len(chis) else chis[0]
        raw_params += chi_l * d * chi_r
        gauge_params += chi_r * chi_r
    n_real_params = max(2 * raw_params - 2 * gauge_params, 1)

    # ── Entanglement entropy at the half-chain cut ──
    # TeNPy's entanglement_entropy() returns S(bond_i) for each bond.
    # Bond i sits between MPS sites i and i+1.
    # Half-chain bipartition {0,..,N/2-1} | {N/2,..,N-1} → bond index N//2 - 1.
    ee = psi_opt.entanglement_entropy()
    half_bond = n_sites // 2 - 1
    S_half = float(ee[half_bond])

    return {
        "energy": float(np.real(E0)),
        "n_real_params": int(n_real_params),
        "chi_max": chi_max,
        "chis": [int(c) for c in chis],
        "entropy": S_half,
    }


# ═══════════════════════════════════════════════════════════════
# NQS GROUND STATE (NetKet)
# ═══════════════════════════════════════════════════════════════


def run_nqs_heisenberg(n_sites, alpha=2, n_samples=4096, n_iter=30, lr=1e-2):
    """
    Run NetKet VMC for 1D Heisenberg with RBM ansatz + SR.

    Key choices:
      - total_sz=0 restricts to the correct symmetry sector
      - MetropolisExchange sampler respects magnetization conservation
      - Complex RBM parameters capture sign structure
      - Stochastic Reconfiguration (SR) is the standard NQS optimizer

    Entanglement entropy:
      After VMC, evaluates the log-amplitude on every Sz=0 basis state
      (dim = C(N, N/2) = 252 for N=10) and computes S_{N/2} from the
      resulting statevector.  Feasible because N=10 is small.

    Returns:
        dict with keys: energy, std, n_params, trace, entropy
    """
    import netket as nk
    import optax

    g = nk.graph.Chain(length=n_sites, pbc=True)
    hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes, total_sz=0)
    H = nk.operator.Heisenberg(hilbert=hi, graph=g, J=1.0)

    model = nk.models.RBM(alpha=alpha, param_dtype=complex)
    sampler = nk.sampler.MetropolisExchange(hi, graph=g, n_chains=16)
    vstate = nk.vqs.MCState(sampler, model, n_samples=n_samples)

    opt = optax.sgd(learning_rate=lr)
    sr = nk.optimizer.SR(diag_shift=0.01, holomorphic=True)
    driver = nk.driver.VMC(H, opt, variational_state=vstate, preconditioner=sr)

    log = nk.logging.RuntimeLog()
    driver.run(n_iter=n_iter, out=log)

    energy_data = log.data.get("Energy", {})
    if isinstance(energy_data, dict) and "Mean" in energy_data:
        raw_trace = energy_data["Mean"]
    else:
        raw_trace = log["Energy"]["Mean"]

    energy_trace = np.real(np.asarray(raw_trace))

    try:
        estats = vstate.expect(H)
        final_e = float(np.real(estats.mean))
        final_std = float(np.real(estats.error_of_mean))
    except Exception:
        final_e = float(energy_trace[-1])
        try:
            raw_sigma = (
                energy_data["Sigma"]
                if (isinstance(energy_data, dict) and "Sigma" in energy_data)
                else log["Energy"]["Sigma"]
            )
            final_std = float(np.real(np.asarray(raw_sigma)[-1]))
        except Exception:
            final_std = float("nan")

    # ── Entanglement entropy from the converged NQS state ──
    # Evaluate log ψ(σ) on every basis state in the Sz=0 sector.
    # For N=10 this is C(10,5) = 252 amplitudes — trivially cheap.
    S_half = float("nan")
    try:
        all_states = np.array(hi.all_states())
        log_psi = np.array(vstate.log_value(all_states))
        # Shift for numerical stability before exponentiating
        log_psi = log_psi - np.max(np.real(log_psi))
        psi_array = np.exp(log_psi)
        psi_array = psi_array / np.linalg.norm(psi_array)
        S_half = compute_half_chain_entropy(psi_array, all_states, n_sites)
    except Exception as ex:
        print(f"    (entanglement computation failed: {ex})")

    return {
        "alpha": alpha,
        "energy": final_e,
        "std": final_std,
        "n_params": int(vstate.n_parameters),
        "trace": energy_trace,
        "entropy": S_half,
    }


# ═══════════════════════════════════════════════════════════════
# FIGURE
# ═══════════════════════════════════════════════════════════════


def make_figure():
    print(f"Heisenberg 1D, N={N_SITES}, PBC")
    print(f"Exact E0 = {E_EXACT:.8f}\n")

    # --- ED entropy reference ---
    ed_entropy = float("nan")
    print("Running ED (exact ground state + entanglement)...")
    try:
        ed_result = run_ed_entropy(N_SITES)
        ed_entropy = ed_result["entropy"]
        print(f"  E0  = {ed_result['energy']:.8f}")
        print(f"  S_{{N/2}} = {ed_entropy:.6f}")
    except Exception as ex:
        print(f"  Failed: {ex}")

    # --- NQS runs ---
    alphas = [1, 2, 4]
    nqs_results = []

    for alpha in alphas:
        print(f"Running NQS (alpha={alpha})...")
        try:
            result = run_nqs_heisenberg(N_SITES, alpha=alpha)
            nqs_results.append(result)
            print(
                f"  E = {result['energy']:.8f}, "
                f"error = {abs(result['energy'] - E_EXACT):.2e}, "
                f"params = {result['n_params']}, "
                f"S_{{N/2}} = {result['entropy']:.4f}"
            )
        except Exception as ex:
            print(f"  Failed: {ex}")

    if not nqs_results:
        print("\nNo NQS results — install netket (pip install netket, Python >= 3.11)")
        return

    # --- DMRG runs (sweep bond dimension) ---
    dmrg_results = []
    dmrg_chis = [8, 16, 32, 64]

    for chi in dmrg_chis:
        print(f"Running DMRG (chi_max={chi})...")
        try:
            result = run_dmrg_heisenberg(N_SITES, chi_max=chi)
            dmrg_results.append(result)
            print(
                f"  E = {result['energy']:.8f}, "
                f"error = {abs(result['energy'] - E_EXACT):.2e}, "
                f"real params ≈ {result['n_real_params']}, "
                f"S_{{N/2}} = {result['entropy']:.4f}"
            )
        except Exception as ex:
            print(f"  Failed: {ex}")

    if not dmrg_results:
        print("\nNo DMRG results — install tenpy (pip install physics-tenpy)")

    # --- Assemble plot data ---
    has_vqe = (
        len(EQ_ENERGIES) == len(P_VALUES)
        and len(EQ_STDS) == len(P_VALUES)
        and len(NEQ_ENERGIES) == len(P_VALUES)
        and len(NEQ_STDS) == len(P_VALUES)
    )

    if not has_vqe:
        print("\n  Note: VQE data missing/incomplete — only showing NQS.\n")

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(19.5, 5.8))

    # ── Shared axis data ──
    if has_vqe:
        eq_x = np.asarray([2 * p for p in P_VALUES], dtype=float)
        neq_x = np.asarray([7 * p for p in P_VALUES], dtype=float)
        eq_e = np.asarray(EQ_ENERGIES, dtype=float)
        neq_e = np.asarray(NEQ_ENERGIES, dtype=float)
        eq_std = np.asarray(EQ_STDS, dtype=float)
        neq_std = np.asarray(NEQ_STDS, dtype=float)
        eq_err = np.abs(eq_e - E_EXACT)
        neq_err = np.abs(neq_e - E_EXACT)
    else:
        eq_x = neq_x = np.asarray([], dtype=float)
        eq_e = neq_e = eq_std = neq_std = np.asarray([], dtype=float)
        eq_err = neq_err = np.asarray([], dtype=float)

    nqs_x = 2.0 * np.asarray([r["n_params"] for r in nqs_results], dtype=float)
    nqs_e = np.asarray([r["energy"] for r in nqs_results], dtype=float)
    nqs_std = np.asarray([r["std"] for r in nqs_results], dtype=float)
    nqs_err = np.abs(nqs_e - E_EXACT)
    nqs_S = np.asarray([r["entropy"] for r in nqs_results], dtype=float)

    has_dmrg = len(dmrg_results) > 0
    if has_dmrg:
        dmrg_x = np.asarray([r["n_real_params"] for r in dmrg_results], dtype=float)
        dmrg_e = np.asarray([r["energy"] for r in dmrg_results], dtype=float)
        dmrg_err = np.abs(dmrg_e - E_EXACT)
        dmrg_err = np.clip(dmrg_err, 1e-14, None)  # floor for log scale
        dmrg_S = np.asarray([r["entropy"] for r in dmrg_results], dtype=float)

    x_candidates = [float(nqs_x.max())]
    x_min_candidates = [float(nqs_x.min())]
    if has_vqe:
        x_candidates.extend([float(eq_x.max()), float(neq_x.max())])
        x_min_candidates.extend([float(eq_x.min()), float(neq_x.min())])
    if has_dmrg:
        x_candidates.append(float(dmrg_x.max()))
        x_min_candidates.append(float(dmrg_x.min()))
    x_min = max(1.0, 0.8 * min(x_min_candidates))
    x_max = 1.08 * max(x_candidates)

    # ── Colours (consistent across all panels) ──
    c_eq = "#2196F3"
    c_neq = "#FF9800"
    c_nqs = "#4CAF50"
    c_dmrg = "#9C27B0"
    c_ed = "k"

    # ════════════════════════════════════════════════════════════
    # LEFT PANEL: |E - E_ED| vs real parameter count
    # ════════════════════════════════════════════════════════════
    if has_vqe:
        ax1.semilogy(
            eq_x, eq_err, "o-", color=c_eq, ms=7, lw=1.7,
            label="Equivariant VQC",
        )
        ax1.semilogy(
            neq_x, neq_err, "s-", color=c_neq, ms=7, lw=1.7,
            label="Non-equivariant VQC",
        )

    ax1.semilogy(
        nqs_x, nqs_err, "D-", color=c_nqs, ms=8, lw=2.0,
        label="NQS / RBM + SR",
    )

    if has_dmrg:
        ax1.semilogy(
            dmrg_x, dmrg_err, "^-", color=c_dmrg, ms=8, lw=2.0,
            label="DMRG (TeNPy)",
        )

    ax1.set_xscale("log")
    ax1.set_xlim(x_min, x_max)
    ax1.set_xlabel("Real variational parameters", fontsize=12)
    ax1.set_ylabel(r"$|E_{\mathrm{var}} - E_{\mathrm{ED}}|$", fontsize=12)
    ax1.set_title("Accuracy vs parameter count", fontsize=13)
    ax1.grid(True, alpha=0.3)
    ax1.legend(
    fontsize=8.0,
    loc="lower left",
    bbox_to_anchor=(0.02, 0.02),
    framealpha=0.9,
)

    # ════════════════════════════════════════════════════════════
    # CENTRE PANEL: absolute energy
    # ════════════════════════════════════════════════════════════
    if has_vqe:
        ax2.errorbar(
            eq_x, eq_e, yerr=eq_std, fmt="o-", color=c_eq,
            ms=7, lw=1.7, capsize=3, label="Equivariant VQC",
        )
        ax2.errorbar(
            neq_x, neq_e, yerr=neq_std, fmt="s-", color=c_neq,
            ms=7, lw=1.7, capsize=3, label="Non-equivariant VQC",
        )

    ax2.errorbar(
        nqs_x, nqs_e, yerr=nqs_std, fmt="D-", color=c_nqs,
        ms=8, lw=2.0, capsize=3, label="NQS / RBM + SR",
    )

    ax2.axhline(E_EXACT, color=c_ed, ls="--", lw=1.4,
                label=fr"ED exact = {E_EXACT:.6f}")

    if has_dmrg:
        ax2.plot(
            dmrg_x, dmrg_e, "^-", color=c_dmrg, ms=8, lw=2.0,
            label="DMRG (TeNPy)",
        )

    ax2.set_xscale("log")
    ax2.set_xlim(x_min, x_max)
    ax2.set_xlabel("Real variational parameters", fontsize=12)
    ax2.set_ylabel("Final energy", fontsize=12)
    ax2.set_title("Absolute energy", fontsize=13)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8.0, loc="upper right",
               bbox_to_anchor=(0.98, 0.98), framealpha=0.9)

    # ════════════════════════════════════════════════════════════
    # RIGHT PANEL: half-chain entanglement error  ΔS_{N/2}
    # ════════════════════════════════════════════════════════════
    # This is the "bridge" diagnostic.  It shows WHY the accuracy
    # curves in the left panel look the way they do:
    #   • DMRG underestimates S at low χ, then saturates quickly.
    #   • NQS stays close to the target entanglement at modest α.
    #   • VQCs are under-entangled at shallow depth, then recover
    #     as circuit depth increases.
    # Plotting ΔS keeps the comparison honest by centering the exact
    # answer at zero instead of visually exaggerating tiny deviations.

    has_vqc_entropy = (
        has_vqe
        and EQ_ENTROPIES is not None
        and NEQ_ENTROPIES is not None
        and len(EQ_ENTROPIES) == len(P_VALUES)
        and len(NEQ_ENTROPIES) == len(P_VALUES)
    )

    use_entropy_error = not np.isnan(ed_entropy)

    if has_vqc_entropy:
        eq_S = np.asarray(EQ_ENTROPIES, dtype=float)
        neq_S = np.asarray(NEQ_ENTROPIES, dtype=float)
        eq_y = eq_S - ed_entropy if use_entropy_error else eq_S
        neq_y = neq_S - ed_entropy if use_entropy_error else neq_S
        ax3.plot(
            eq_x, eq_y, "o-", color=c_eq, ms=7, lw=1.7,
            label="Equivariant VQC",
        )
        ax3.plot(
            neq_x, neq_y, "s-", color=c_neq, ms=7, lw=1.7,
            label="Non-equivariant VQC",
        )

    # NQS entanglement
    nqs_S_valid = ~np.isnan(nqs_S)
    if np.any(nqs_S_valid):
        nqs_y = nqs_S[nqs_S_valid] - ed_entropy if use_entropy_error else nqs_S[nqs_S_valid]
        ax3.plot(
            nqs_x[nqs_S_valid], nqs_y,
            "D-", color=c_nqs, ms=8, lw=2.0,
            label="NQS / RBM + SR",
        )

    # DMRG entanglement
    if has_dmrg:
        dmrg_y = dmrg_S - ed_entropy if use_entropy_error else dmrg_S
        ax3.plot(
            dmrg_x, dmrg_y, "^-", color=c_dmrg, ms=8, lw=2.0,
            label="DMRG (TeNPy)",
        )

    # ED exact entropy reference
    if use_entropy_error:
        ax3.axhline(
            0.0, color=c_ed, ls="--", lw=1.4,
            label=r"ED exact $\Delta S_{N/2}=0$",
        )

        ax3.annotate(
            "VQCs are under-entangled at shallow depth,\n"
            "then reach the correct half-chain $S$ by moderate depth",
            xy=(0.04, 0.84), xycoords="axes fraction",
            fontsize=7.5, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec="gray", alpha=0.85),
        )
        ax3.set_ylabel(
            r"$\Delta S_{N/2} = S_{N/2} - S_{N/2}^{\rm exact}$",
            fontsize=12,
        )
        ax3.set_title("Half-chain entanglement error", fontsize=13)
    else:
        ax3.set_ylabel(r"$S_{N/2}$  (von Neumann)", fontsize=12)
        ax3.set_title(r"Half-chain entanglement entropy", fontsize=13)

    ax3.set_xscale("log")
    ax3.set_xlim(x_min, x_max)
    ax3.set_xlabel("Real variational parameters", fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=8.0, loc="best", framealpha=0.9)

    # ── Footer ──
    sector_dim = comb(N_SITES, N_SITES // 2)
    fig.suptitle(f"Heisenberg 1D, N={N_SITES}, PBC", fontsize=14, y=0.98)
    fig.text(
        0.5, 0.01,
        (
            f"Fairness note: NQS and DMRG restricted to S$^z$=0 sector "
            f"(dim = C({N_SITES},{N_SITES // 2}) = {sector_dim}). "
            f"{VQC_SYMMETRY_NOTE}\n"
            "NQS counts complex params as 2 real DOF. "
            "DMRG counts MPS tangent-space dimension (gauge-corrected). "
            "Entanglement diagnostic computed from Schmidt decomposition "
            "(DMRG: MPS bond; NQS/ED: full statevector SVD)."
        ),
        ha="center", va="bottom", fontsize=7.5,
    )

    plt.tight_layout(rect=[0, 0.10, 1, 0.95])
    plt.savefig("triangle_heisenberg.png", dpi=150, bbox_inches="tight")
    print("\nSaved: triangle_heisenberg.png")


if __name__ == "__main__":
    make_figure()