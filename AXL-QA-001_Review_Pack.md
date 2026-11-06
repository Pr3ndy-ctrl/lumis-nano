# AXL-QA-001: Lumis Nano review pack

| Field | Value |
|---|---|
| Document ID | AXL-QA-001 |
| Revision | 1.0 |
| Date | 2026-08-11 |
| Subject | `lumis-nano` v0.1.0, the open educational demonstrator |
| Repository | https://github.com/Pr3ndy-ctrl/lumis-nano — MIT licensed, already public |
| Prepared for | Pavan Parihar and team, first engagement |
| Status | Draft. Release on NDA execution, although this artefact is public and needs none |

---

## 1. What this is, and what it is not

`lumis-nano` is a **single-file educational demonstrator**, 166 lines of Python, that shows how a neural network weight matrix compiles into settings for a mesh of optical interferometers. It is MIT licensed and already public.

**It is not the Lumis SDK.** The commercial SDK is a separate, licensed product. This is the chalkboard, not the fab.

**Why we are starting here.** It exercises the same core mathematics as the commercial product, it is already public so there is nothing to protect, and it lets you begin immediately without waiting on paperwork. If this engagement goes well we extend to the licensed SDK under NDA. That sequence is about giving you a clean start, not about withholding.

**Please review it as what it is.** An educational demonstrator has different obligations from production software. Absence of input hardening is a legitimate finding, but we would weight it differently from a numerical error.

---

## 2. What it does, in one paragraph

Any weight matrix `W` factors by singular value decomposition into `U · diag(S) · Vᵀ`. `U` and `Vᵀ` are unitary, meaning energy-conserving, meaning they can in principle be built from lossless optical rotations. Any N×N unitary decomposes into N(N−1)/2 two-by-two rotations (Reck 1994), and each such rotation is one Mach-Zehnder interferometer with two phase shifters. `diag(S)` only scales, so it maps to per-channel attenuators. The script performs that decomposition, prints the phase-shifter settings, draws the optical path, and verifies the result by reconstructing the unitary from the mesh.

---

## 3. Baseline: what a correct run looks like

Run before you change anything, so you have a reference.

```
python3 lumis_nano.py --size 4 --seed 7 --no-table
```

**Expected on a machine without PyTorch installed:**

| Check | Expected |
|---|---|
| Singular values | `[2.3537 0.6402 0.5006 0.3209]` |
| MZI count, 4×4 | 12 total (6 per unitary mesh) |
| `max\|U·S·Vᵀ − W\|` | ~4.4e-16 |
| `max\|mesh − unitary\|` | ~2.2e-16 |
| Status line | `PASS` |
| Digital power ledger | 73.600 mW |
| Exit code | 0 |

**Note the PyTorch dependency, see finding F-1.** With PyTorch installed the weight matrix differs and every number above changes. That is expected behaviour of the current code, not a fault you have introduced.

---

## 4. Known state, so you do not spend time re-finding these

We have run a first pass ourselves. These are already known. **Finding anything beyond this list is the value of the engagement.**

| Ref | Finding | Severity, our view |
|---|---|---|
| **F-1** | `make_weights()` branches on whether `torch` is importable. `torch.randn` with seed 7 and `numpy.default_rng(7)` produce different matrices, so identical commands on two machines give different output depending only on whether PyTorch happens to be installed. **Reproducibility defect** | Medium |
| **F-2** | `reck_decompose()` assumes its input is unitary and does not check. Fed a non-unitary matrix it returns silently, and `rebuild()` then differs from the input by order 1.0. No exception, no warning | Medium |
| **F-3** | `--seed` accepts arbitrarily large integers without bound. Negative seeds raise a raw `ValueError` from numpy rather than a handled message | Low |
| **F-4** | `--size` is restricted to 4 or 8 by `argparse choices`. The underlying maths is correct at other sizes; the restriction is presentational | Informational |

**Verified correct in our pass**, so no need to duplicate unless you want to confirm:

- Reconstruction accuracy across random complex unitaries at N = 2, 3, 4, 5, 8, 16. Worst case 5.6e-16.
- Edge cases: identity, anti-diagonal permutation, pure phase diagonal. All exact or near-exact.
- MZI count equals N(N−1)/2 at N = 4, 8, 16.
- Determinism within a single process for a fixed seed.
- The README energy arithmetic: 64 MACs × 4.6 pJ × 1e9/s = 294.4 mW, quoted as ~295 mW.

---

## 5. Scope of this engagement

### In scope

| Area | What we are asking for |
|---|---|
| **Numerical correctness** | Does the decomposition hold across sizes, seeds, degenerate and adversarial inputs? Where does it break? |
| **Robustness** | Malformed arguments, extreme values, resource exhaustion, unhandled exceptions |
| **Reproducibility** | Same input, same output, across machines and Python versions |
| **Dependency and supply chain** | `numpy`, optional `torch`. Version pinning, known advisories, install-time risk from `setup.py` |
| **Code quality** | Anything you would flag in review. Be blunt |

### Out of scope

- The commercial Lumis SDK. Separate artefact, separate engagement.
- Photonics hardware, the MGOC-H120 architecture, anything to do with the ASIC.
- Performance benchmarking. This is a demonstrator; it is not optimised and does not claim to be.

### Deliverable

A written report with findings, each carrying a severity rating, reproduction steps and, where you have one, a suggested fix. Format is yours.

---

## 6. Suggested test surface

Offered as a starting point, not a constraint. You will think of things we have not.

**Numerical**
- Sweep N from 2 to 64 on random complex unitaries; plot reconstruction error against N
- Ill-conditioned inputs: near-singular matrices, repeated singular values, matrices with a very large condition number
- Real-only unitaries, permutation matrices, matrices with exact zeros in awkward places
- The `abs(b) < 1e-12` early-exit branch. Is 1e-12 the right threshold? What happens just either side of it?
- Accumulated floating-point error: does it grow as N², N log N, or worse?

**Robustness**
- Fuzz the CLI
- Import the module and call `reck_decompose` directly with non-square, empty, NaN and inf inputs
- Memory and time behaviour at large N

**Supply chain**
- `setup.py` declares `install_requires=['numpy']` with no version bound
- `python_requires='>=3.6'`. Does it actually run on 3.6?
- Advisories against the numpy versions that would be resolved

---

## 7. Environment

```
python3 -m venv .venv && source .venv/bin/activate
pip install numpy
git clone https://github.com/Pr3ndy-ctrl/lumis-nano.git
cd lumis-nano
python3 lumis_nano.py --size 4 --seed 7
```

Both with and without `torch` installed, please, given F-1.

---

## 8. Contact

Technical questions on the mathematics come to me directly. Contractual matters to Tafadzwa Matimba.

Dwain Prendergast
Founder, CEO and CTO, Axiom Photonics Ltd
founder@axiomprocessor.com
