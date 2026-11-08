# LUMIS-NANO

**The matrix multiply is free. You've been paying for it your whole career.**

`lumis-nano` is a single-file, dependency-minimal demonstration of how a neural
network layer compiles into photons. It is the educational core of the math
behind Axiom Photonics' MGOC-H120 deterministic inference engine. No cloud, no
framework, no mercy: one script, `numpy`, and the terminal.

## INSTALL

```
pip install lumis-nano
```

or straight from source:

```
pip install git+https://github.com/Pr3ndy-ctrl/lumis-nano.git
```

Then run it:

```
lumis-nano --size 4                  # full mesh + phase table
lumis-nano --size 8 --seed 42 --no-table
lumis-nano --size 16                 # any size from 2 to 64
```

No install needed either — it's one file:

```
python3 lumis_nano.py --size 8
```

## THE PHYSICS, WITHOUT APOLOGY

Every dense layer is a matrix `W`. Every matrix factors as

```
W = U · diag(S) · Vᵀ        (Singular Value Decomposition)
```

That factorization is not a numerical trick. It is a **hardware schematic**:

1. `U` and `Vᵀ` are unitary. Unitary means energy-conserving. Energy-conserving
   means **lossless optics**. Any N×N unitary decomposes into N(N−1)/2 planar
   2×2 rotations (Reck et al., *PRL* 1994) - and a 2×2 optical rotation is
   exactly one **Mach-Zehnder Interferometer**: two beamsplitters, two phase
   shifters (θ sets the split ratio, φ sets the relative phase). Etch the mesh,
   set the phases, done.
   > The core uses the Reck triangular ordering for clarity. The production compiler uses the Clements rectangular ordering. Both use N(N-1)/2 interferometers; the difference is optical depth. At N=64 that is 64 stages against 125, so Clements accumulates roughly half the insertion loss and half the phase error along a path.
2. `diag(S)` only scales. Scaling is **attenuation** - per-channel amplitude
   modulators. Passive optics can't amplify, so we normalize by S_max and fold
   the global gain into laser power. Physics keeps the books balanced.

Encode your activation vector as optical amplitudes, launch it into the left
edge of the chip, and interference performs all N² multiply-accumulates
*during propagation* - picoseconds, in parallel, deterministically. There is no
clock, no instruction stream, no cache hierarchy, no HBM stall. The answer
arrives at the photodetectors at the speed of light in silicon because the
circuit **is** the matrix.

## THE ENERGY ARGUMENT

A digital MAC moves charge through transistors every single cycle - roughly
4.6 pJ per fp32 MAC in 45 nm CMOS (Horowitz, ISSCC 2014). An 8×8 layer at 10⁹
inferences/sec burns ~295 mW doing arithmetic a piece of structured glass does
for 0 J. Phase shifters are set **once** at compile time; afterward the
multiply is a boundary condition, not a computation, so the O(N²) arithmetic
cost drops to zero.

The node itself is not free, and the ledger says so. Drivers, the laser and the
electro-optic phase-hold draw a **fixed floor** - about 7 W for a full 64-mode
Rev 1 node - whether it runs once a second or a billion times. The point is not
"100% saving"; it is that this floor is **flat in the rate** while the digital
cost grows with it. So there is a breakeven: below it the fixed floor costs more
than the arithmetic it replaces, above it the photonic node wins and keeps
winning. At a full 64-node that crossover is ~3.7×10⁸ matvec/s (design target).
EO shifters hold a voltage rather than a current, which is why the hold is
<0.2 W across a full mesh and not the ~25 mW/shifter of a thermo-optic mesh -
swap in datasheet numbers for your own operating point. GPUs hit a thermal wall;
waveguides don't, but a laser still has to be on.

## WHY DETERMINISTIC MATTERS

A GPU gives you throughput with a probability distribution around it: cache
misses, warp scheduling, thermal throttling. A photonic mesh gives you a
**fixed, physical latency** - light through a known path length. Same input,
same time-of-flight, every time. That is what inference at the edge, in
vehicles, and in control loops actually requires.

## WHAT THE SCRIPT DOES

Generates a random weight matrix (NumPy, deterministic from `--seed` on every
machine) → performs the SVD → compiles both unitaries into explicit MZI (θ, φ) settings
via Givens rotations → renders the die as raw ASCII, light path left to right
→ verifies the mesh reconstructs `U` and `Vᵀ` to machine precision → prints
the power ledger. A compiler you cannot verify is a rumor; this one proves
itself every run.

## LICENSE

MIT. Take it, learn it, build the future with it.

### Two tiers

`lumis-nano` is the free, open teaching core: one file, the Reck mesh, the SVD,
the ASCII die. Take it, learn it, build with it.

The full SDK is **`axiom-lumis`**, and it is live:

```
pip install axiom-lumis
```

It uses the production Clements rectangular compiler, enforces the Rev 1 power
envelope (it refuses a schedule the chip could not run instead of pretending),
and will generate a feasibility report for a whole model (`lumis study`). It is
pre-silicon: every power and latency figure it prints is a Rev 1 design target,
and it says so on every line. There is no free tier of `axiom-lumis`: `lumis-nano` above is the free one, and
it already runs the full 64-mode node. Commercial `axiom-lumis` seats add the
production compiler, batch compile, export and support.

The chip itself is not fabricated yet. We would rather say that plainly than let
a roadmap read as a product. Follow along at
[axiomprocessor.com](https://axiomprocessor.com).

*Axiom Photonics - the speed of light is not a bottleneck. It's the roadmap.*
