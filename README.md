# hyperbolic1d — Transient Compressible Flow by Hyperbolic PDE Methods

Python framework for transient compressible-flow problems governed by
hyperbolic PDEs, developed toward simulation of **vertical-well CO₂
blowdown** with real-fluid thermodynamics, gravity/friction sources, and
wellbore–formation heat transfer.

The starting point is the M.Sc. thesis of **T. R. Gessner (2010, UFSC;
advisor J. R. Barbosa Jr.)**, *Modelagem numérica do escoamento anular
gás-líquido transiente pelo método da divisão da matriz de coeficientes*,
whose Split Coefficient Matrix Method (SCMM) solvers — originally in
Fortran with the Aziz & Settari `GBAND` banded solver — are reconstructed
here and benchmarked against modern Godunov-type finite-volume methods.

## Project phases

1. **Phase 1 (this repo, done for single phase):** faithful Python
   reconstruction of the original SCMM solvers; validation on the original
   shock-tube benchmarks against exact solutions and the thesis error tables.
2. **Phase 2 (started):** Godunov-type finite-volume solvers (exact/HLL/HLLC
   Riemann fluxes, first-order and MUSCL-Hancock with limiters) compared with
   SCMM on identical benchmarks: accuracy, shock resolution, conservation,
   diffusion, stability, cost.
3. **Phase 3:** extension to transient blowdown of a 2,000-ft vertical well —
   N₂ ideal gas → real-gas CO₂ → dense-phase/transcritical → two-phase.
4. **Phase 4:** transient radial heat transfer to tubing/casing/cement/formation.

## Layout

```
src/hyperbolic1d/       solver package
  exact_riemann.py      exact Riemann solver (validation reference)
  scmm.py               SCMM reconstruction (Gessner 2010, Ch. 3)
  godunov.py            finite-volume Godunov/HLLC/MUSCL solvers
  eos.py                EOS interface (ideal gas now; real-fluid CO2 later)
  wellbore.py           Phase-3 wellbore blowdown model (gravity, friction,
                        choke BC, wall thermal-model interface)
  benchmarks.py         benchmark cases + error metrics (thesis Table 3.3)
scripts/                benchmark drivers
notebooks/              Colab-ready notebooks
docs/                   analysis of the original Fortran codes & formulation
legacy_fortran/         archived original Fortran sources (P1.FOR, GBAND)
tests/                  pytest regression suite
results/                generated benchmark outputs
```

## Quick start (Colab or local)

```bash
pip install "hyperbolic1d[plot] @ git+https://github.com/JaderBarbosaLSU/hyperco2-wellbore"
```

```python
from hyperbolic1d import solve_scmm_shock_tube, solve_godunov_shock_tube
from hyperbolic1d.benchmarks import mean_abs_errors, exact_reference

r = solve_scmm_shock_tube(formulation="conservative", order=2)
print(mean_abs_errors(r))   # compare with thesis Table 3.3

g = solve_godunov_shock_tube(flux="hllc", reconstruction="muscl")
print(mean_abs_errors(g))
```

For the CO₂ (Span–Wagner) stages add the `co2` extra:

```bash
pip install "hyperbolic1d[plot,co2] @ git+https://github.com/JaderBarbosaLSU/hyperco2-wellbore"
```

### Notebooks (one-click Colab)

| Notebook | Covers | Open |
|---|---|---|
| 01 shock tube | SCMM vs Godunov/HLLC vs exact Riemann | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JaderBarbosaLSU/hyperco2-wellbore/blob/main/notebooks/01_single_phase_shock_tube.ipynb) |
| 02 N₂ blowdown | wellbore model, choke BC, vessel-ODE check | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JaderBarbosaLSU/hyperco2-wellbore/blob/main/notebooks/02_wellbore_blowdown.ipynb) |
| 03 formation heat transfer | conjugate completion/formation model | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JaderBarbosaLSU/hyperco2-wellbore/blob/main/notebooks/03_formation_heat_transfer.ipynb) |
| 04 CO₂ blowdown | Span–Wagner real-fluid path, N₂ comparison | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JaderBarbosaLSU/hyperco2-wellbore/blob/main/notebooks/04_co2_blowdown.ipynb) |
| 05 CO₂ + geothermal + HEM choke | stage-2 headline case: HEM choke, geothermal IC, saturation margin, fiber sensors | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JaderBarbosaLSU/hyperco2-wellbore/blob/main/notebooks/05_co2_geothermal_hem.ipynb) |

Each notebook installs the package from GitHub in its first cell, so a
fresh Colab runtime needs no setup.

## Benchmark: single-phase shock tube (Gessner Sec. 3.4)

Air, 4 m closed tube, diaphragm at midspan; left 500 kPa / 400 K, right
100 kPa / 300 K; 4001 nodes (Δx = 1 mm), Δt = 1 µs, output at t = 3 ms;
reference = exact Riemann solution. See
`docs/original_formulation.md` for the full analysis of the original
method and `results/` for current error tables.

## Tests

```bash
pip install -e ".[test]"
pytest tests/
```
