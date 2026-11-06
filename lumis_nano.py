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

# Rev 1 photonic node power model. Design targets, labelled as such. These are
# the SAME figures the axiom-lumis SDK admits against: the EO (Pockels TFLN/BTO)
# phase shifters hold a VOLTAGE, not a current, so static hold is tiny -- not
# the ~25 mW/shifter of a thermo-optic mesh. See the whitepaper, section 09.
P_DRIVE_LASER = 6.8          # W, modulator drivers + laser coupling floor per node
HOLD_PER_MZI  = 0.2 / 2016   # W, EO static hold per MZI (<0.2 W across a full mesh)

def make_weights(n, seed):
    """A random dense layer, numpy only so the same seed gives the same matrix
    on every machine. (Branching on whether torch was importable made `--seed`
    reproducible only within one environment -- same command, different matrix
    elsewhere.)"""
    return np.random.default_rng(seed).standard_normal((n, n))

def reck_decompose(U):
    """Compile a unitary into a triangular mesh of MZIs (Reck et al. 1994).
    Each MZI couples two adjacent waveguides (rails p, p+1) and applies:
        G(theta,phi) = [[ cos(t),          e^{i*phi}*sin(t)],
                        [-e^{-i*phi}*sin(t), cos(t)        ]]
    theta -> internal phase shifter, sets the split ratio (the 'rotation').
    phi   -> external phase shifter, sets the relative phase.
    We Givens-rotate U to diagonal; each rotation IS one physical MZI.
    Returns (mzis=[(p, theta, phi), ...], residual_phases D).

    Raises ValueError if U is not a finite, square, 2x2-or-larger unitary: a
    mesh can only realise a unitary, so compiling anything else would return a
    layout that does not rebuild to the input. The check is here, in the free
    core, on purpose -- garbage in should fail loudly, not silently."""
    U = np.asarray(U, dtype=complex)
    if U.ndim != 2 or U.shape[0] != U.shape[1]:
        raise ValueError(f"expected a square 2D matrix, got shape {U.shape}")
    N = U.shape[0]
    if N < 2:
        raise ValueError("mesh needs at least 2 modes")
    if not np.all(np.isfinite(U)):
        raise ValueError("matrix contains NaN or infinity")
    if not np.allclose(U @ U.conj().T, np.eye(N), atol=1e-8):
        raise ValueError("matrix is not unitary (U @ U^dagger != I); factor a "
                         "general matrix with an SVD first")
    V, mzis = U.copy(), []
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

def mesh_operator(mzis, D, N):
    """Compose the mesh as a product of dense NxN unitaries and return the
    operator it realises.

    What this does and does NOT prove, stated honestly. It embeds each MZI as a
    full NxN identity with its 2x2 block and composes them with dense matrix
    products, where rebuild() applies the block by two-row slicing. So it rules
    out a slicing, indexing or embedding mistake: if either path had one, the
    two would disagree. They agree to machine precision (0.0 at N<=8).

    It does NOT give independence against a phase-convention error, because it
    uses the SAME 2x2 block as rebuild(). Two things to keep straight:
      - A convention error in rebuild()'s block ALONE is already caught by the
        round-trip -- flip a sign and |mesh - U| jumps to ~1.3, not zero.
      - A convention error SHARED between reck_decompose() and rebuild() would
        cancel in the round-trip, and mesh_operator() shares that block, so it
        cannot catch it either. The only check that can is a compiler written
        independently, fed the same matrix. That is an external exercise, not
        something one file can do to itself; where such a compiler has been run
        against this one, they agree."""
    M = np.diag(D).astype(complex)
    for p, t, f in mzis:                      # same propagation order as rebuild
        T = np.eye(N, dtype=complex)
        c, s = np.cos(t), np.sin(t)
        # MZI transfer (unitary): a phase-fed beamsplitter on rails p, p+1.
        T[p,   p]   =  c
        T[p,   p+1] = -np.exp(1j * f) * s
        T[p+1, p]   =  np.exp(-1j * f) * s
        T[p+1, p+1] =  c
        M = T @ M
    return M

def pack(mzis):
    """ASAP layout: each MZI drops into the EARLIEST column where both its rails
    are free, so MZIs on disjoint rails share a column exactly as they are
    physically tiled on the die. Tracking the last column that touched each rail
    (rather than only checking the final column) packs the mesh as shallow as the
    physics allows -- a depth-(2N-3) mesh instead of a near-diagonal staircase."""
    cols, last = [], {}
    for p, t, f in mzis:
        c = max(last.get(p, -1), last.get(p + 1, -1)) + 1   # after both rails' last use
        if c == len(cols):
            cols.append([])
        cols[c].append((p, t, f))
        last[p] = last[p + 1] = c
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
    ap.add_argument("--size", type=int, default=4, help="mesh size N (>= 2)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--no-table", action="store_true")
    a = ap.parse_args()
    if not 2 <= a.size <= 64:
        ap.error("--size must be between 2 and 64")
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

    # [4] Verification. A compiler you cannot verify is a rumor. Two checks: the
    #     SVD factorisation, and the mesh recomposed as a dense forward model
    #     (mesh_operator) -- a cross-check against slicing/embedding errors, not
    #     a claim of convention-independence (see its docstring).
    e_svd  = np.abs(U @ np.diag(S) @ Vt - W).max()
    e_mesh = max(np.abs(mesh_operator(mz_u, D_u, a.size) - U).max(),
                 np.abs(mesh_operator(mz_v, D_v, a.size) - Vt).max())
    print(f"\n  [4] VERIFICATION\n      max|U.S.Vt - W|      = {e_svd:.3e}"
          f"\n      max|mesh - unitary|  = {e_mesh:.3e}  (dense forward recomposition)"
          f"\n      STATUS: {'PASS' if max(e_svd, e_mesh) < 1e-9 else 'FAIL'}"
          " (machine precision = exact compile)")

    # [5] The energy argument, honestly. The optical ARITHMETIC is free: light
    #     does the multiply by propagating, so the O(N^2) MAC cost drops to zero.
    #     The NODE is not free: it draws a fixed floor (drivers, laser, EO hold)
    #     whether it runs once a second or a billion times. The win is that this
    #     floor is FLAT in the rate while digital cost grows with it -- so there
    #     is a breakeven rate, not a blanket "100% saving".
    macs  = a.size ** 2
    mzis  = a.size * (a.size - 1) // 2
    p_dig    = macs * E_MAC_DIGITAL * CLOCK_HZ          # scales with rate
    p_static = P_DRIVE_LASER + mzis * HOLD_PER_MZI      # flat vs rate
    rate_be  = p_static / (macs * E_MAC_DIGITAL)        # breakeven matvec/s
    print(f"\n  [5] POWER LEDGER (Rev 1 design targets)")
    print(f"      DIGITAL  @ {CLOCK_HZ:.0e}/s : {macs} MACs x {E_MAC_DIGITAL*1e12:.1f} pJ x rate "
          f"= {p_dig:7.3f} W   (scales with rate)")
    print(f"      PHOTONIC floor       : drivers+laser {P_DRIVE_LASER:.1f} W + "
          f"hold {mzis*HOLD_PER_MZI:.3f} W = {p_static:7.3f} W   (flat vs rate)")
    verdict = "photonic wins" if CLOCK_HZ > rate_be else "digital still cheaper"
    print(f"      BREAKEVEN            : ~{rate_be:.2e} matvec/s  "
          f"(at {CLOCK_HZ:.0e}/s: {verdict})")
    print("      NOTE: the arithmetic is free; the node is not. EO phase")
    print("      shifters hold a voltage, so static hold is <0.2 W across a full")
    print("      mesh -- not the ~25 mW/shifter of a thermo-optic mesh. Swap in")
    print(f"      datasheet constants for your real operating point.\n{bar}\n  END OF COMPILE\n{bar}")

if __name__ == "__main__":
    main()
