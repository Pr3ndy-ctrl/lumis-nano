#!/usr/bin/env python3
"""
Baseline regression suite for lumis-nano v0.1.0.

Supplied to a reviewer as a starting point, not as a definition of done.
Every test here is one we have already run; the value of the engagement is
whatever it does NOT catch.

    pip install numpy pytest
    pytest -v test_lumis_nano.py

Runs standalone too:
    python3 test_lumis_nano.py
"""
import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SRC = HERE / "lumis_nano.py"

_spec = importlib.util.spec_from_file_location("lumis_nano", SRC)
ln = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ln)

TOL = 1e-9          # "exact compile" threshold used by the tool itself
rng = np.random.default_rng(20260811)


def random_unitary(n, complex_=True):
    m = rng.standard_normal((n, n))
    if complex_:
        m = m + 1j * rng.standard_normal((n, n))
    q, r = np.linalg.qr(m)
    return q * (np.diag(r) / np.abs(np.diag(r)))     # fix QR sign ambiguity


def roundtrip_error(U):
    mzis, D = ln.reck_decompose(U.astype(complex))
    return np.abs(ln.rebuild(mzis, D) - U).max()


# ---------------------------------------------------------------- correctness
def test_roundtrip_complex_unitaries():
    for n in (2, 3, 4, 5, 8, 16):
        e = roundtrip_error(random_unitary(n))
        assert e < TOL, f"N={n} reconstruction error {e:.3e}"


def test_roundtrip_real_unitaries():
    for n in (2, 4, 8):
        e = roundtrip_error(random_unitary(n, complex_=False))
        assert e < TOL, f"real N={n} error {e:.3e}"


def test_edge_case_matrices():
    cases = {
        "identity": np.eye(4, dtype=complex),
        "anti_diagonal": np.fliplr(np.eye(4)).astype(complex),
        "phase_diagonal": np.diag([1, -1, 1j, -1j]).astype(complex),
        "single_swap": np.array([[0, 1, 0, 0], [1, 0, 0, 0],
                                 [0, 0, 1, 0], [0, 0, 0, 1]], dtype=complex),
    }
    for name, U in cases.items():
        e = roundtrip_error(U)
        assert e < TOL, f"{name} error {e:.3e}"


def test_mzi_count_matches_theory():
    """A Reck mesh needs exactly N(N-1)/2 interferometers."""
    for n in (2, 4, 8, 16):
        mzis, _ = ln.reck_decompose(random_unitary(n))
        assert len(mzis) == n * (n - 1) // 2, f"N={n} emitted {len(mzis)}"


def test_svd_identity_holds():
    W = ln.make_weights(8, 7)
    U, S, Vt = np.linalg.svd(W)
    assert np.abs(U @ np.diag(S) @ Vt - W).max() < 1e-12


def test_error_growth_is_benign():
    """Reconstruction error should not blow up with N."""
    errs = {n: roundtrip_error(random_unitary(n)) for n in (4, 8, 16, 32)}
    assert errs[32] < 1e-12, f"error at N=32 is {errs[32]:.3e}"
    assert errs[32] < errs[4] * 1e4, f"error growing too fast: {errs}"


def test_packing_preserves_all_mzis():
    """Column packing must not drop or duplicate an interferometer."""
    mzis, _ = ln.reck_decompose(random_unitary(8))
    packed = ln.pack(mzis)
    assert sum(len(c) for c in packed) == len(mzis)


def test_packed_columns_use_disjoint_rails():
    """Two MZIs in the same column must not share a waveguide."""
    mzis, _ = ln.reck_decompose(random_unitary(8))
    for col in ln.pack(mzis):
        rails = [r for p, _, _ in col for r in (p, p + 1)]
        assert len(rails) == len(set(rails)), f"rail collision in {col}"


# ------------------------------------------------------------ reproducibility
def test_seed_determinism_within_process():
    assert np.array_equal(ln.make_weights(4, 7), ln.make_weights(4, 7))


def test_different_seeds_differ():
    assert not np.array_equal(ln.make_weights(4, 1), ln.make_weights(4, 2))


def test_cli_run_is_reproducible():
    def run():
        return subprocess.run([sys.executable, str(SRC), "--size", "4",
                               "--seed", "7", "--no-table"],
                              capture_output=True, text=True, timeout=60).stdout
    assert run() == run(), "identical invocations produced different output"


# ------------------------------------------------------------------ known gap
def test_KNOWN_F2_non_unitary_input_is_not_rejected():
    """F-2, known. reck_decompose does not validate unitarity.

    This test DOCUMENTS current behaviour rather than endorsing it. If a fix
    lands that raises on non-unitary input, this test should be inverted.
    """
    M = rng.standard_normal((4, 4))                 # emphatically not unitary
    mzis, D = ln.reck_decompose(M.astype(complex))
    err = np.abs(ln.rebuild(mzis, D) - M).max()
    assert err > 1e-6, "input was closer to unitary than intended"
    # No exception raised. That is the finding.


# ------------------------------------------------------------------------ cli
def test_cli_exits_clean_on_valid_input():
    for args in (["--size", "4"], ["--size", "8", "--no-table"],
                 ["--size", "4", "--seed", "0"]):
        r = subprocess.run([sys.executable, str(SRC)] + args,
                           capture_output=True, text=True, timeout=60)
        assert r.returncode == 0, f"{args} -> rc={r.returncode}: {r.stderr[:200]}"
        assert "PASS" in r.stdout, f"{args} did not report PASS"


def test_cli_rejects_invalid_size():
    for bad in ("0", "-4", "5", "1000", "abc"):
        r = subprocess.run([sys.executable, str(SRC), "--size", bad],
                           capture_output=True, text=True, timeout=60)
        assert r.returncode != 0, f"--size {bad} was accepted"


def test_verification_block_reports_pass():
    r = subprocess.run([sys.executable, str(SRC), "--size", "8", "--no-table"],
                       capture_output=True, text=True, timeout=60)
    assert "STATUS: PASS" in r.stdout


# ------------------------------------------------------------- documentation
def test_readme_energy_arithmetic():
    """README quotes ~295 mW for an 8x8 layer at 1e9 ops/s."""
    p_mw = 8 * 8 * ln.E_MAC_DIGITAL * ln.CLOCK_HZ * 1e3
    assert abs(p_mw - 294.4) < 0.1, f"got {p_mw:.1f} mW"


if __name__ == "__main__":
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    passed = failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except Exception as exc:                     # noqa: BLE001
            print(f"  FAIL  {name}: {exc}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed, {len(fns)} total")
    sys.exit(1 if failed else 0)
