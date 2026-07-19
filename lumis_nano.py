#!/usr/bin/env python3
# =============================================================================
#  LUMIS-NANO :: PHOTONIC MATRIX COMPILER -- EDUCATIONAL CORE
#  Axiom Photonics / Open Source Division            License: MIT
# -----------------------------------------------------------------------------
#  THE IDEA: any weight matrix W factors as  W = U @ diag(S) @ Vt  (SVD).
#  U and Vt are UNITARY -> they conserve energy -> they can be built from
#  lossless 2x2 optical rotations (Mach-Zehnder Interferometers, MZIs).
#  diag(S) only scales -> it maps to per-channel optical ATTENUATORS.
#  Result: a full matrix-vector product executed by light in ~picoseconds,
#  at ZERO switching energy in the mesh itself. This file shows the math.
# =============================================================================
import argparse, numpy as np

E_MAC_DIGITAL = 4.6e-12   # J per fp32 multiply-accumulate (45nm, Horowitz 2014)
CLOCK_HZ      = 1.0e9     # assumed inference rate: 1e9 matrix-vector ops / s

def make_weights(n, seed):
    """A random dense layer. Uses torch if installed, numpy otherwise."""
    try:
        import torch; torch.manual_seed(seed)
        return torch.randn(n, n).numpy().astype(np.float64)
    except ImportError:
        return np.random.default_rng(seed).standard_normal((n, n))

def reck_decompose(U):
    """Compile a unitary into a triangular mesh of MZIs (Reck et al. 1994).
    Each MZI couples two adjacent waveguides (rails p, p+1) and applies:
        G(theta,phi) = [[ cos(t),          e^{i*phi}*sin(t)],
                        [-e^{-i*phi}*sin(t), cos(t)        ]]
    theta -> internal phase shifter, sets the split ratio (the 'rotation').
    phi   -> external phase shifter, sets the relative phase.
    We Givens-rotate U to diagonal; each rotation IS one physical MZI.
    Returns (mzis=[(p, theta, phi), ...], residual_phases D)."""
    V, mzis = U.astype(complex).copy(), []
    N = V.shape[0]
    for j in range(N - 1):                    # zero the sub-diagonal, col by col
        for i in range(N - 1, j, -1):         # climb from the bottom row up
            a, b = V[i - 1, j], V[i, j]
            if abs(b) < 1e-12: mzis.append((i - 1, 0.0, 0.0)); continue
            theta = np.arctan2(abs(b), abs(a))          # split ratio angle
            phi   = np.angle(a) - np.angle(b)           # phase alignment
            G = np.array([[np.cos(theta), np.exp(1j*phi)*np.sin(theta)],
                          [-np.exp(-1j*phi)*np.sin(theta), np.cos(theta)]])
            V[[i - 1, i], :] = G @ V[[i - 1, i], :]     # apply -> b is nulled
            mzis.append((i - 1, theta, phi))
    # Gk..G1 U = D  =>  U = G1'..Gk' D  (' = dagger). Light hits D first, then
    # Gk' .. G1' -- so return the list in LIGHT-PROPAGATION order (reversed).
    return mzis[::-1], np.diag(V).copy()      # V is now diag(unit phases) = D

def rebuild(mzis, D):
    """Physics check: push light back through the mesh -> must equal U."""
    U = np.diag(D).astype(complex)
    for p, t, f in mzis:                      # propagation order: D, Gk'..G1'
        Gd = np.array([[np.cos(t), -np.exp(1j*f)*np.sin(t)],
                       [np.exp(-1j*f)*np.sin(t), np.cos(t)]])   # G^dagger
        U[[p, p + 1], :] = Gd @ U[[p, p + 1], :]
    return U

def pack(mzis):
    """Greedy layout: MZIs on disjoint rails act simultaneously (same column),
    exactly as they are physically tiled on the die."""
    cols = []
    for p, t, f in mzis:
        if cols and all(abs(p - q) > 1 for q, _, _ in cols[-1]):
            cols[-1].append((p, t, f))
        else:
            cols.append([(p, t, f)])
    return cols

def draw_mesh(n, cols_v, sigma, cols_u, width=100):
    """ASCII light path:  input -> [Vt mesh] -> [S attenuators] -> [U mesh]."""
    segs, k = [], 0                            # each seg = n strings, 1 column
    for tag, cols in (("Vt", cols_v), ("U", cols_u)):
        if tag == "U":                         # the diagonal sits between meshes
            segs.append([f"[{s:4.2f}]" for s in sigma])
        for col in cols:
            seg = ["------"] * n
            for p, _, _ in col:
                seg[p] = seg[p + 1] = f"={k:02d}==="   # MZI straddles 2 rails
                k += 1
            segs.append(seg)
    per = max(1, (width - 12) // 7)            # wrap wide meshes across chunks
    for c0 in range(0, len(segs), per):
        for r in range(n):
            body = "".join(s[r] for s in segs[c0:c0 + per])
            pre = f"IN{r} >-" if c0 == 0 else "     -"
            post = f"-> D{r}" if c0 + per >= len(segs) else "-"
            print(f"  {pre}{body}{post}")
        print()

def table(name, mzis, k0):
    """Phase-shifter settings, indexed identically to the mesh drawing."""
    print(f"  {name} MESH SETTINGS " + "-" * 45)
    print("  MZI   RAILS   THETA [rad]   PHI [rad]")
    for k, (p, t, f) in enumerate(mzis, start=k0):
        print(f"  {k:03d}   {p}-{p+1}     {t:+10.6f}   {f:+10.6f}")

def main():
    ap = argparse.ArgumentParser(description="lumis-nano photonic compiler demo")
    ap.add_argument("--size", type=int, default=4, choices=(4, 8))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--no-table", action="store_true")
    a = ap.parse_args()
    bar = "=" * 78
    print(f"\n{bar}\n  LUMIS-NANO v0.1 :: DETERMINISTIC PHOTONIC COMPILE PIPELINE\n"
          f"  TARGET: MZI TRIANGULAR MESH   LAYER: {a.size}x{a.size}   "
          f"SEED: {a.seed}\n{bar}\n")

    # [1] The layer. In production this is one nn.Linear pulled from PyTorch.
    W = make_weights(a.size, a.seed)
    print("  [1] WEIGHT MATRIX W (the digital artifact to be compiled):\n")
    for row in W: print("      " + " ".join(f"{x:+7.3f}" for x in row))

    # [2] SVD: the entire compiler is one line of linear algebra.
    U, S, Vt = np.linalg.svd(W)
    print("\n  [2] SVD FACTORIZATION  W = U @ diag(S) @ Vt")
    print("      U, Vt unitary  -> LOSSLESS. Energy in equals energy out.")
    print("      A unitary is a rigid rotation of the optical field; rotations")
    print("      decompose into 2x2 planar rotations -> one MZI each. The mesh")
    print("      is PASSIVE GLASS: set phases once, multiply forever, 0 J/op.")
    print("      diag(S) only attenuates -> per-channel amplitude modulators.")
    print(f"      Singular values: {np.array2string(S, precision=4)}")
    sig = S / S.max()   # passive optics cannot amplify: normalize, fold S.max()
    print(f"      Attenuators S/S_max = {np.array2string(sig, precision=4)}"
          f"  (global gain {S.max():.4f} folded into laser power)")

    # [3] Compile both unitaries into physical phase-shifter settings.
    mz_u, D_u = reck_decompose(U)
    mz_v, D_v = reck_decompose(Vt)
    n_mzi = len(mz_u) + len(mz_v)             # = 2 * N(N-1)/2

    # [4] The die, as light sees it. IN -> Vt rotates -> S attenuates -> U rotates.
    print(f"\n  [3] OPTICAL CIRCUIT :: {n_mzi} MZIs + {a.size} ATTENUATORS")
    print("      Light enters LEFT:  x -> [Vt mesh] -> [S] -> [U mesh] -> "
          "detectors D\n      (residual diagonal phase screens D_v, D_u are "
          "lossless and not drawn)\n")
    draw_mesh(a.size, pack(mz_v), sig, pack(mz_u))
    if not a.no_table:
        table("Vt", mz_v, 0); print(); table("U", mz_u, len(mz_v))

    # [5] Verification. A compiler you cannot verify is a rumor.
    e_svd  = np.abs(U @ np.diag(S) @ Vt - W).max()
    e_mesh = max(np.abs(rebuild(mz_u, D_u) - U).max(),
                 np.abs(rebuild(mz_v, D_v) - Vt).max())
    print(f"\n  [4] VERIFICATION\n      max|U.S.Vt - W|      = {e_svd:.3e}"
          f"\n      max|mesh - unitary|  = {e_mesh:.3e}"
          f"\n      STATUS: {'PASS' if max(e_svd, e_mesh) < 1e-9 else 'FAIL'}"
          " (machine precision = exact compile)")

    # [6] The energy argument. A digital MAC burns energy moving charge every
    #     single cycle. A phase shifter is set once; after that the multiply is
    #     just propagation -- interference does the arithmetic for free.
    macs  = a.size ** 2
    p_dig = macs * E_MAC_DIGITAL * CLOCK_HZ
    print(f"\n  [5] POWER LEDGER @ {CLOCK_HZ:.0e} matrix-vector ops/sec")
    print(f"      DIGITAL : {macs} MACs x {E_MAC_DIGITAL*1e12:.1f} pJ x rate "
          f"= {p_dig*1e3:8.3f} mW")
    print(f"      OPTICAL : passive mesh compute               =    0.000 mW")
    print(f"      SAVED   : {p_dig*1e3:8.3f} mW  (100.0% of MAC energy)")
    print("      NOTE: laser, DAC/ADC and thermal phase-hold power are I/O-")
    print("      boundary costs, amortized across the whole mesh; the O(N^2)")
    print(f"      arithmetic itself is free.\n{bar}\n  END OF COMPILE\n{bar}")

if __name__ == "__main__":
    main()
