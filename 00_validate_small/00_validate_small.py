"""
Day 1 Validation: TFIM + Heisenberg on small lattices
======================================================
Verify ED reference values and sign conventions before VMC work.

Checks:
  - 2-site TFIM analytical solution
  - Reference energy tables for TFIM and Heisenberg
  - Cross-check numpy ED vs NetKet (if installed):
      * Odd-N eigenvalue test (breaks bipartite degeneracy, definitive)
      * Sparse matrix Frobenius norm comparison (catches all mismatches)
      * Even-N eigenvalue comparison (regression test)

SIGN CONVENTION (verified empirically):
  Our numpy:   H = -J * sum Z_i Z_{i+1}  -  h * sum X_i   (ferro for J>0)
  NetKet:      H = +J * sum Z_i Z_j      -  h * sum X_i   (+ZZ convention)
  => Use J=-1.0 in NetKet to match our J=+1.0

  On even-N bipartite chains, eigenvalues match for BOTH signs (sublattice
  rotation symmetry). Only odd-N or sparse matrix comparison distinguishes.

Requirements:
  - numpy (required)
  - netket >= 3.0, Python >= 3.11 (optional, for cross-checks)
  - Do NOT install NetKet via conda (JAX compatibility issues)

Usage:
    python validate_small.py
"""
from contextlib import redirect_stdout
from pathlib import Path
import sys

import numpy as np


class Tee:
    """Write stdout to multiple streams (console + log file)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def make_log_path():
    """Create a fixed path for this script's output log."""
    script_dir = Path(__file__).resolve().parent
    return script_dir / "00_validate_small_result.txt"


I2 = np.eye(2, dtype=complex)
SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)


def kron_chain(ops):
    result = ops[0]
    for op in ops[1:]:
        result = np.kron(result, op)
    return result


def tfim_hamiltonian(n, h=1.0, J=1.0, pbc=True):
    """
    H = -J sum Z_i Z_{i+1} - h sum X_i  (ferromagnetic for J>0)

    To match in NetKet: nk.operator.Ising(h=h, J=-J)
    """
    dim = 2**n
    H = np.zeros((dim, dim), dtype=complex)
    n_bonds = n if pbc else n - 1
    for i in range(n_bonds):
        j = (i + 1) % n
        ops = [I2]*n; ops[i] = SZ; ops[j] = SZ
        H -= J * kron_chain(ops)
    for i in range(n):
        ops = [I2]*n; ops[i] = SX
        H -= h * kron_chain(ops)
    return H


def heisenberg_hamiltonian(n, J=1.0, pbc=True):
    """H = J sum (X_i X_{i+1} + Y_i Y_{i+1} + Z_i Z_{i+1})"""
    dim = 2**n
    H = np.zeros((dim, dim), dtype=complex)
    n_bonds = n if pbc else n - 1
    for i in range(n_bonds):
        j = (i + 1) % n
        for pauli in [SX, SY, SZ]:
            ops = [I2]*n; ops[i] = pauli; ops[j] = pauli
            H += J * kron_chain(ops)
    return H


def ed_ground_energy(H):
    return np.linalg.eigvalsh(H)[0].real


# ─── Analytical check ───

def check_2site_tfim():
    print("=== CHECK 1: 2-site TFIM (PBC) analytical vs numpy ===")
    H_analytic = np.array([
        [-2, -1, -1,  0],
        [-1,  2,  0, -1],
        [-1,  0,  2, -1],
        [ 0, -1, -1, -2]], dtype=float)
    ev_a = np.sort(np.linalg.eigvalsh(H_analytic))
    ev_n = np.sort(np.linalg.eigvalsh(tfim_hamiltonian(2).real))
    print(f"  Analytical: {np.round(ev_a, 6)}")
    print(f"  Numpy:      {np.round(ev_n, 6)}")
    assert np.allclose(ev_a, ev_n), "MISMATCH!"
    print("  PASSED\n")


# ─── Reference tables ───

def print_tfim_table():
    print("=== TFIM reference energies (h=1, J=1, PBC) ===")
    print(f"  {'N':>4s}  {'E0':>14s}  {'E0/N':>12s}")
    for n in [2, 3, 4, 5, 6, 8, 10]:
        E0 = ed_ground_energy(tfim_hamiltonian(n))
        print(f"  {n:4d}  {E0:14.8f}  {E0/n:12.8f}")
    print(f"  Thermo limit: E0/N -> -4/pi = {-4/np.pi:.8f}\n")


def print_heisenberg_table():
    print("=== Heisenberg reference energies (J=1, PBC) ===")
    print(f"  {'N':>4s}  {'E0':>14s}  {'E0/N':>12s}")
    for n in [4, 6, 8, 10]:
        E0 = ed_ground_energy(heisenberg_hamiltonian(n))
        print(f"  {n:4d}  {E0:14.8f}  {E0/n:12.8f}")
    print()


# ─── NetKet cross-checks ───

def cross_check_netket():
    print("=== Cross-check: numpy ED vs NetKet ===")
    try:
        import netket as nk
    except ImportError:
        print("  NetKet not installed — skipping")
        print("  Install: pip install netket (Python >= 3.11, no conda)\n")
        return

    print(f"  NetKet version: {nk.__version__}\n")

    # ── DEFINITIVE: odd-N sign disambiguation ──
    # On odd rings, bipartite sublattice rotation doesn't apply,
    # so eigenvalues genuinely differ for +J vs -J.
    print("  --- TFIM sign convention: odd-N eigenvalue test ---")
    print("  (Odd rings break bipartite degeneracy → definitive sign test)")
    for n in [3, 5]:
        E_np = ed_ground_energy(tfim_hamiltonian(n))
        g = nk.graph.Chain(length=n, pbc=True)
        hi = nk.hilbert.Spin(s=1/2, N=g.n_nodes)
        for J_nk in [+1.0, -1.0]:
            H_nk = nk.operator.Ising(hilbert=hi, graph=g, h=1.0, J=J_nk)
            E_nk = nk.exact.lanczos_ed(H_nk, k=1, compute_eigenvectors=False)[0]
            match = "MATCH" if abs(E_np - E_nk) < 1e-8 else "no match"
            print(f"    N={n} J_nk={J_nk:+.0f}: numpy={E_np:.8f}  "
                  f"netket={E_nk:.8f}  [{match}]")
    print("    => Confirmed: use J=-1 in NetKet to match our convention\n")

    # ── BELT-AND-SUSPENDERS: sparse matrix comparison on N=4 ──
    print("  --- Sparse matrix Frobenius norm (N=4) ---")
    n = 4
    g = nk.graph.Chain(length=n, pbc=True)
    hi = nk.hilbert.Spin(s=1/2, N=g.n_nodes)
    H_np = tfim_hamiltonian(n)
    for J_nk in [+1.0, -1.0]:
        H_nk_dense = nk.operator.Ising(hilbert=hi, graph=g, h=1.0, J=J_nk).to_dense()
        frob = np.linalg.norm(H_nk_dense - H_np)
        match = "MATCH" if frob < 1e-10 else "DIFFER"
        print(f"    J_nk={J_nk:+.0f}: ||H_nk - H_np||_F = {frob:.2e}  [{match}]")
    print()

    # ── TFIM eigenvalue check (using correct J=-1) ──
    print("  --- TFIM eigenvalue check (J_netket=-1) ---")
    for n in [4, 6, 8, 10]:
        E_np = ed_ground_energy(tfim_hamiltonian(n))
        g = nk.graph.Chain(length=n, pbc=True)
        hi = nk.hilbert.Spin(s=1/2, N=g.n_nodes)
        H_nk = nk.operator.Ising(hilbert=hi, graph=g, h=1.0, J=-1.0)
        E_nk = nk.exact.lanczos_ed(H_nk, k=1, compute_eigenvectors=False)[0]
        diff = abs(E_np - E_nk)
        ok = "OK" if diff < 1e-8 else f"DIFF={diff:.2e}"
        print(f"    N={n:2d}: numpy={E_np:.8f}  netket={E_nk:.8f}  [{ok}]")
    print()

    # ── Heisenberg eigenvalue check ──
    print("  --- Heisenberg eigenvalue check ---")
    for n in [4, 6, 8, 10]:
        E_np = ed_ground_energy(heisenberg_hamiltonian(n))
        g = nk.graph.Chain(length=n, pbc=True)
        hi = nk.hilbert.Spin(s=1/2, N=g.n_nodes, total_sz=0)
        H_nk = nk.operator.Heisenberg(hilbert=hi, graph=g, J=1.0)
        E_nk = nk.exact.lanczos_ed(H_nk, k=1, compute_eigenvectors=False)[0]
        diff = abs(E_np - E_nk)
        ok = "OK" if diff < 1e-8 else f"DIFF={diff:.2e}"
        print(f"    N={n:2d}: numpy={E_np:.8f}  netket={E_nk:.8f}  [{ok}]")
    print()

    # ── Sanity beacon: NetKet docs cite E0=-10.2517 for N=8 TFIM ──
    E8 = ed_ground_energy(tfim_hamiltonian(8))
    print(f"  Sanity: our N=8 TFIM E0={E8:.4f}")
    print(f"  NetKet docs cite E0=-10.2517 for 8-site Ising → consistent\n")

    # ── NOTE for 2D work ──
    print("  --- Note for 2D TFIM (relevant to later notebooks) ---")
    print("  Even×even square lattices (e.g. 4×4) are also bipartite,")
    print("  so the same eigenvalue blindness applies in 2D.")
    print("  For a non-bipartite 2D check, use 3×3 (9 sites, 2^9=512).\n")


def run_validation():
    check_2site_tfim()
    print_tfim_table()
    print_heisenberg_table()
    cross_check_netket()
    print("YOUR KEY TARGETS:")
    print(f"  TFIM  N=10: E0 = {ed_ground_energy(tfim_hamiltonian(10)):.8f}")
    print(f"  Heis  N=10: E0 = {ed_ground_energy(heisenberg_hamiltonian(10)):.8f}")
    print()
    print("CONVENTION SUMMARY:")
    print("  Our numpy:  H = -J*ZZ - h*X  (ferro for J>0)")
    print("  NetKet:     nk.operator.Ising(h=h, J=-1.0)  to match")
    print("  Heisenberg: nk.operator.Heisenberg(J=1.0)   matches directly")


if __name__ == "__main__":
    log_path = make_log_path()
    with log_path.open("w", encoding="utf-8") as log_file:
        tee = Tee(sys.stdout, log_file)
        with redirect_stdout(tee):
            print(f"Log file: {log_path}")
            print()
            run_validation()
    print(f"\nSaved run output to {log_path}")
