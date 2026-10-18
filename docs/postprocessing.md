# Post-Processing, Fiber-Optic Sensors, and Figure Quality Notes

Module: `hyperbolic1d.postprocess` (+ extensions in `animate.py`).
Demo driver: `scripts/run_postprocess_demo.py`. Outputs land in
`results/` and `results/export/`.

## 1. Fiber-optic sensor representation

`FiberSensor(name, depth, location, r=None)` places a sampling station at
a depth below the wellhead and a radial location: `fluid` (bore),
`casing_in` (7-5/8" CRA casing node), `casing_ext` (node nearest the
13-3/8" casing — the external-fiber position from the well schematic),
`cement`, `formation` (requires `r`), or an explicit radius. A
`SensorArray` passed as `WellboreModel.solve(sensors=...)` is sampled at
the history cadence; fluid stations record (T, p, ρ, u), solid stations
the nearest radial-node temperature of the conjugate wall model.
`lsu_fiber_sensors()` builds the default set: bore DTS points + external
13-3/8" fiber at 50/500/1000/1500/1950 ft.

## 2. Outputs

**CSV** (`export_csv`): `wellhead_history.csv` (t, p, T, face velocity,
ṁ, mass, choked flag), `sensors_T.csv` (T(t) per sensor with
location/depth/radius metadata header), `sensors_fluid_pru.csv`
(p, ρ, u at fluid stations), `profiles_t*.csv` (depth, z, p, T, ρ, u at
selected times).

**Publication figures** (`pub_style` context: serif/STIX, in-ticks,
300 dpi): `plot_sensor_histories` (T(t) grouped by radial station),
`plot_profiles_at_times` (vertical-depth orientation, depth increasing
downward, fiber depths marked), `plot_velocity_publication` (see §3),
`plot_wellhead_histories`.

**Animations**: `animate_blowdown(..., sensor_depths=...)` marks fiber
depths on all four field panels; `animate_temperature` shows the fluid
T(z) field with markers at the *sampled* sensor values (fluid dots,
external-fiber squares) beside the growing sensor T(t) traces with a
time cursor — the direct visual link between the computed field and what
the fibers will measure.

## 3. Velocity-figure diagnosis (what was wrong and what was not)

The original profile figure overlaid seven velocity snapshots taken at
100-s intervals. Grid-refinement analysis (`results/velocity_grid_check.png`,
200/400/1200 cells) shows the "messy" content was **grid-converged
physics, not numerics**:

1. The oscillations superposed on the linear quasi-static profile are a
   traveling acoustic wave launched by the valve opening and by the
   gravitational settling of the uniform (non-equilibrium) initial
   column. Residual amplitude about the linear trend at t = 100 s:
   1.18 / 1.20 / 1.21 m/s at 200/400/1200 cells — identical, i.e.
   converged. The front steepens nonlinearly into a weak, sharp wave
   (N-wave formation) and persists because the baseline configuration is
   loss-free (adiabatic, frictionless, little choke dissipation).
2. Enabling wall friction damps the reverberation by a factor ~150
   (residual 1.18 → 0.008 m/s at t = 100 s) while the discrete energy
   budget still closes to machine precision (friction enters only the
   momentum equation; dissipation is captured by the conservative energy
   formulation). The publication set therefore runs with friction
   enabled — a physical modeling choice, not cosmetic smoothing.
3. The figure itself aliased the wave at arbitrary phases. The revised
   `plot_velocity_publication` separates (a) the acoustic phase (clean
   propagating front from a hydrostatic initialization) from (b) the
   quasi-static discharge, with few, well-chosen times.

No smoothing is applied anywhere; the CSV export carries the raw
histories.

## 4. Physical features surfaced by the improved figures (not artifacts)

* **Thermally driven internal flow.** Under the conjugate thermal model
  the late-time velocity profiles show near-bottom *downward* flow
  (−0.5 to −2 m/s) and localized structure near z ≈ 100 m: cooling fluid
  contracts and the column redistributes mass downward while discharge
  continues above. Grid-converged; absent in the adiabatic case.
* **Near-bottom cold layer.** The deepest cells cool most (fluid there
  is nearly stagnant, so the Dittus-Boelter h falls to its laminar
  floor and heat ingress is weakest). This layer is sensitive to the
  convection closure at u → 0; in reality internal natural convection
  would mix it. Flagged as a closure limitation, to revisit alongside
  the CO₂ stage.
* **Residual acoustic ripple.** The wellhead T and u_f histories carry a
  small sustained oscillation (≈0.3–0.5 K band) at the acoustic period
  2L/a ≈ 3.5 s, damped by friction but continuously re-excited by the
  transient. At print resolution the 3.5-s ripple renders as a band
  whose apparent ~1-min "teeth" are a moiré effect of rasterization; the
  true period is visible in the CSV data.
* **External-fiber response.** The 13-3/8" external fiber barely moves
  (≤0.5 K over 30 min) because the air annulus isolates it from the bore
  transient — an experimentally relevant prediction: for blowdown
  thermometry the bore DTS carries essentially all of the signal, and
  the external fiber primarily constrains the annulus conductance.
