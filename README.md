# NQS Hybrid Portfolio: Variational Methods for Quantum Many-Body Systems

Benchmarking and bridging **equivariant variational quantum circuits (VQC)**, **neural quantum states (NQS)**, and **tensor network methods** for strongly correlated quantum systems — with a focus on hybrid quantum-classical simulation workflows.

## Motivation

Modern variational approaches to quantum many-body problems fall into three broad families: parameterized quantum circuits, classical neural-network wavefunctions, and tensor networks. Each has distinct tradeoffs in expressivity, trainability, and scalability. This repository provides a unified benchmark across all three, with extensions toward hybrid data-assisted training and time-dependent neural quantum states.

## Repository Structure

```
nqs-hybrid-portfolio/
│
├── 00_validation/
│   └── validate_small.py             # ED sanity checks (analytical + reference tables)
│                                      # Cross-checks: numpy vs NetKet (sparse matrix + eigenvalue)
│
├── 01_triangle_comparison/
│   └── triangle_heisenberg.py        # VQC (equivariant + non-equivariant) vs NQS (RBM)
│                                      # vs ED on 1D Heisenberg — the "signature figure"
│
├── 02_tfim_ground_state/
│   └── tfim_2d_ground_state.py       # 2D TFIM ground state: RBM vs ARNN (autoregressive)
│
├── 03_dynamics/
│   └── quench_dynamics.py            # TDVP/tVMC time evolution after quantum quench
│
├── 04_tnqs_interval/
│   └── tnqs_interval.py              # t-NQS interval training (Van de Walle et al. 2024)
│
├── 05_hybrid_snapshots/
│   └── hybrid_snapshot_pretraining.py  # Synthetic snapshot pretraining + VMC refinement (§6)
│
├── 06_bose_hubbard/
│   └── bose_hubbard_ground_state.py   # BH ground state: n_max convergence + U/t sweep + 2D
│
├── baselines/
│   └── dmrg_tebd_baselines.py        # DMRG ground states + TEBD quench dynamics (TeNPy)
│
└── figures/
```

## What Each Component Demonstrates

| Directory | Method | Application claim |
|-----------|--------|-------------------|
| `00_validation/` | ED cross-checks | Rigorous numerical foundations across frameworks |
| `01_triangle_comparison/` | VQC + NQS + ED | Unique bridge between equivariant circuits, NQS, and tensor networks |
| `02_tfim_ground_state/` | RBM + ARNN / VMC+SR | 2D NQS ground state — the regime where NQS outperform tensor networks |
| `03_dynamics/` | TDVP/tVMC | Quench dynamics with ED validation |
| `04_tnqs_interval/` | t-NQS (JAX/Flax) | From-scratch implementation of Van de Walle, Schmitt & Bohrdt (2024) |
| `05_hybrid_snapshots/` | Data-assisted NQS | Pipeline ready for experimental quantum simulator data |
| `06_bose_hubbard/` | VMC+SR on Fock space |  target model — spin pipeline transfers to bosons |
| `baselines/` | DMRG/TEBD (TeNPy) | Established tensor network expertise for benchmarking |

## Reference Energies

### TFIM (h=1, J=1, PBC)
| N  | E0           | E0/N       | Note |
|----|-------------|------------|------|
| 2  | -2.82842712 | -1.41421356 | |
| 3  | -4.00000000 | -1.33333333 | odd (frustrated) |
| 4  | -5.22625186 | -1.30656297 | |
| 5  | -6.47213595 | -1.29442719 | odd (frustrated) |
| 6  | -7.72740661 | -1.28790110 | |
| 8  | -10.25166179 | -1.28145772 | matches NetKet docs |
| 10 | -12.78490649 | -1.27849065 | |
| ∞  |             | -1.27323954 (= -4/π) | |

### Heisenberg (J=1, PBC)
| N  | E0           | E0/N       |
|----|-------------|------------|
| 4  | -8.00000000 | -2.00000000 |
| 6  | -11.21110255 | -1.86851709 |
| 8  | -14.60437380 | -1.82554672 |
| 10 | -18.06178542 | -1.80617854 |

## Context

This work builds on existing implementations of:
- **Equivariant variational quantum circuits** for Heisenberg and TFIM models following [Meyer et al., PRX Quantum (2023)](https://doi.org/10.1103/PRXQuantum.4.010328)
- **DMRG with TeNPy** for Schwinger model and spin chains (up to N=40 sites)
- **VQE benchmarks** with symmetry-preserving ansätze

Extended here with neural quantum state methods targeting hybrid quantum-classical simulation workflows, inspired by:
- [Lange, Bohrdt et al. (2025)](https://arxiv.org/abs/2406.00091) — Transformer NQS + quantum simulator data (*Quantum* **9**, 1675)
- [Van de Walle, Schmitt, Bohrdt (2024)](https://arxiv.org/abs/2412.11830) — Time-dependent neural quantum states

The theoretical framework (`NQS_Theoretical_Framework.md`) additionally covers the **Bose–Hubbard model** — the target system for FOR 5919 Project 01 — including Fock-space conventions, the hybrid experiment–NQS loop, and how time evolution enables approximate many-body state tomography.

## Sign Convention (verified empirically)

Our numpy TFIM uses H = −J Σ Z_iZ_j − h Σ X_i (ferromagnetic for J>0).
NetKet's `nk.operator.Ising(J, h)` implements H = +J Σ Z_iZ_j − h Σ X_i.

**Use `J=-1.0` in NetKet to match our `J=+1.0`.**

This was verified via odd-N eigenvalue tests (N=3,5) where bipartite symmetry
cannot mask the sign difference, plus sparse matrix Frobenius norm comparison.
On even-N chains, both signs give identical eigenvalues — only odd-N or direct
matrix comparison is definitive. See `00_validation/validate_small.py`.

## Quickstart

```bash
# Verify sign conventions and reference energies
python 00_validation/validate_small.py

# Signature figure: VQC vs NQS vs ED (requires VQE data pasted in)
python 01_triangle_comparison/triangle_heisenberg.py

# 2D TFIM ground state (RBM vs autoregressive)
python 02_tfim_ground_state/tfim_2d_ground_state.py --Lx 4 --Ly 4

# Quench dynamics: ED vs NQS TDVP
python 03_dynamics/quench_dynamics.py --N 8

# t-NQS interval training
python 04_tnqs_interval/tnqs_interval.py --N 8 --n-epochs 3000

# Hybrid snapshot pretraining vs cold-start VMC
python 05_hybrid_snapshots/hybrid_snapshot_pretraining.py --N 10

# Bose–Hubbard ground state (target model)
python 06_bose_hubbard/bose_hubbard_ground_state.py --N 8

# DMRG/TEBD baselines
python baselines/dmrg_tebd_baselines.py --N 20
```

Each script produces `.png` and `.pdf` figures in its working directory.

## Dependencies

```
# Python >= 3.11 required for NetKet
netket >= 3.0        # NQS/VMC (do NOT install via conda)
jax >= 0.4
flax                 # for t-NQS custom models (04_tnqs_interval/)
pennylane            # for VQC baselines
physics-tenpy        # for DMRG/TEBD baselines
optax
scipy
matplotlib
numpy
```
