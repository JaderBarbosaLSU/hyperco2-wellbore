# Real-Fluid CO₂ Thermodynamics (Phase 5, single-phase gas stage)

Module: `hyperbolic1d.realfluid.SpanWagnerCO2`. Campaign driver:
`scripts/run_co2_campaign.py`; verification: `tests/test_realfluid.py`.

## 1. Property package and range of validity

The **Span & Wagner (1996)** reference equation of state for CO₂,
evaluated through **CoolProp 8.0** (Helmholtz-energy formulation). The
underlying EOS is valid 216.6–1100 K up to 800 MPa; the tabulated envelope
used by the solver (gas side, 0.5–300 kg/m³ × 220–340 K by default) is far
inside it. All properties — p, T, e, h, s, c_p, and the **full
thermodynamic speed of sound** a² = (∂p/∂ρ)_s — derive from the same
Helmholtz form and are therefore mutually consistent.

## 2. Coupling to the conservative variables

The solver's per-cell closure (ρ, e) → (p, T, a) is served from
precomputed tables on a uniform (ln ρ, T) grid (512×512 default): T is
recovered by a vectorized Newton iteration on the tabulated e(ln ρ, T)
(monotone in T), then p and a by bilinear evaluation. Interpolation is the
only consistency loss and is quantified (`table_verification`, automated
in the tests): **max relative error 5×10⁻⁵ (p), 1×10⁻⁵ (T), 6×10⁻⁶ (a)**
against direct Span–Wagner evaluation. Closure cost ≈ 0.5 ms per 300-cell
call. Table nodes inside the vapor dome are filled by gas-side
extrapolation purely to keep boundary-grazing lookups finite; model
validity is governed by the saturation monitor, never by those nodes.

Face states for the general-EOS HLLC flux reconstruct (ρ, u, p, e, a) as
primitives (MUSCL + Hancock, with the pressure predictor coefficient
ρa² replacing the ideal-gas γp — identical for ideal gases); wave speeds
use the Davis/Einfeldt direct bounds. The O(Δz²) thermodynamic mismatch of
independently reconstructed face triples is standard practice for
real-gas MUSCL schemes.

## 3. Boundary (choke) thermodynamics

Full-fidelity scalar CoolProp evaluations (low-level `AbstractState`
flashes), split by time scale:

* **Demand side** (slowly varying): stagnation state from (h₀, s); real-gas
  isentropic-nozzle flux G(p_t) = ρ(p_t, s₀)·√(2(h₀−h(p_t, s₀))) maximized
  over the throat pressure (choked when the maximizer exceeds the back
  pressure). Recomputed when the wellhead pressure drifts by 0.2 %.
  At 5 MPa/300 K the real-gas critical flux is ~1.2× the ideal-gas value
  (G ∝ √(ρ₀p₀), Z = 0.69) — verified in the tests, with recovery of the
  ideal limit at low pressure. If the throat isentrope reaches the vapor
  dome before the sonic point, the flux is evaluated at the dome bound and
  flagged (`last_choke_dome_limited`): the physical choke would then
  involve condensing nozzle flow, outside single-phase scope.
* **Supply side** (must track the boundary cell **every step**): the C⁺
  characteristic/isentrope coupling is applied in linearized form,
  u_f = u_i + (p_i−p_f)/(ρᵢaᵢ), ρ_f = ρᵢ + (p_f−p_i)/aᵢ², solved against
  the demand in closed form. The linearization error is
  O((Δp_face/ρa²)²) ≈ 10⁻⁵ for choke-dominated discharge. The exact
  nonlinear solve (`wellhead_face_state_general`, quadrature invariant +
  Brent) is retained for verification and for large-Δp configurations.

**Negative result (recorded deliberately).** The first implementation
cached the *entire* face flux over a 5-step stride to amortize CoolProp
cost. The frozen outflow flux decouples from the boundary cell and the
pair oscillates unstably, draining the wellhead cell by ~2 MPa within a
second — which initially masqueraded as physical dome entry. With the
split above (per-step supply response, strided demand) the boundary is
stable and the cost is ~3 ms/step total. Lesson: the fast-acoustic part of
an outflow boundary cannot be time-lagged.

## 3b. HEM two-phase choke (flashing across the valve only)

Phase change associated with the strong expansion across the wellhead
restriction is handled **algebraically in the choke model** while the
distributed wellbore flow remains strictly single phase (the per-step
dome monitor of §4 is unchanged and still stops the run if any *cell*
enters the dome). With `hem_choke=True` (default) the throat isentrope
in `orifice_mass_flux` is allowed to continue into the vapor dome: ρ and
h along s₀ are then the homogeneous-equilibrium (flashed) mixture
properties from the same Span–Wagner EOS, so

G(p_t) = ρ_HEM(p_t, s₀) √(2(h₀ − h_HEM(p_t, s₀)))

is the standard HEM nozzle flux, and its maximization over p_t yields
the HEM critical (choked) flow — the thermodynamic-equilibrium limit of
flashing nozzle flow (Starkman et al. 1964; Henry & Fauske 1971 discuss
non-equilibrium corrections, not included). Reported per call:
`last_choke_flashing` (throat inside the dome), `last_choke_throat_quality`
(HEM vapor quality at the throat), and `last_choke_dome_limited` (now
meaning the expansion was truncated by the EOS *validity* boundary —
sublimation region — rather than the sonic condition). The solver
records the flashing flag and throat quality in the result history
(`choke_flashing`, `choke_quality`).

At 5 MPa/300 K stagnation the sonic point lies just inside the dome:
throat quality 0.975, and the HEM critical flux exceeds the previous
single-phase dome bound by only **1.4 %** — i.e. the earlier
dome-limited flux was already an excellent approximation there. At
330 K stagnation the sonic point is dry and the model reduces exactly
to the single-phase result (automated test). The upstream state at the
choke must be single phase; two-phase conditions *inside the wellbore*
remain out of scope and still halt the run with a flagged report.

## 3c. Geothermal initial temperature profile

`WellboreModel.solve(..., geothermal=Geothermal(T_surface, gradient))`
replaces the uniform initial temperature with T(z,0) = T_surface + G·z
(z = depth) and integrates the hydrostatic initialization with the
local temperature, dp/dz = −ρ(p, T(z))g. Both parameters are plain
dataclass inputs, not hard-coded. For the Louisiana-summer case:

* **T_surface = 305.4 K (32.2 °C / 90 °F)** — a representative warm
  summer-day wellhead/fluid temperature for Baton Rouge (NOAA 1991–2020
  climate normals: July daily mean ≈ 28.5 °C, mean daily high ≈ 33 °C).
* **G = 25.5 K/km (1.4 °F/100 ft)** — a typical Louisiana Gulf-Coast
  geothermal gradient (regional range ≈ 22–30 K/km, SMU Geothermal
  Laboratory heat-flow compilation); identical to the default used by
  the formation thermal model since Phase 4, so fluid and formation are
  initialized consistently by passing the same `Geothermal` instance to
  `FormationCoupledWall(geothermal=...)`.

Resulting initial fluid temperatures for the 2,000-ft well (5 MPa
wellhead, hydrostatic): 305.5 K (wellhead), 309.2 K (500 ft), 313.3 K
(1,000 ft), 317.1 K (1,500 ft), 320.9 K (bottomhole), with initial
saturation margins 18.0 → 27.6 K top to bottom (versus 12.6 → 6.5 K
for the uniform 300 K column — note the uniform column's margin
*decreases* with depth because hydrostatic compression raises T_sat,
while the geothermal gradient more than compensates).

Caveat (documented approximation): the seasonal temperature wave
penetrates only the top ~10 m of the ground, so initializing the
*formation* with the summer surface temperature slightly warms the
shallow formation relative to the annual-mean profile (294.3 K
surface). For blowdown time scales (≲ hours) the effect on the fluid
is confined to the near-wellhead cells and is small compared with the
18 K initial margin; both profiles remain available as inputs.

## 4. Single-phase validity monitoring

`saturation_margin(p, T) = T − T_sat(p)` (T_sat from a 200-node table,
< 0.01 K interpolation error; above p_crit the gate is T_crit) is checked
**every step** over all cells. On the first negative margin the run stops
and reports (t, depth, p, T, ρ, T_sat) in `result.two_phase_entry`; the
margin history is recorded in `result.sat_margin_min`, together with
the depth of the minimum (`result.sat_margin_depth`); the full margin
field m(z,t) = T − T_sat(p) is recoverable from the (p, T) snapshots. The p–T
trajectories of wellhead and bottom cells are plotted against the
saturation curve (`cmp_co2_saturation_path.png`).

## 5. Matched-comparison conditions (deviation, identified and explained)

The original N₂ baseline (13.79 MPa/300 K) is **retained untouched**; CO₂
at that state is a dense liquid-like fluid, not a single-phase gas. The
fluid-comparison pair therefore runs both fluids at a **5 MPa wellhead /
300 K**, hydrostatically initialized — the physically correct shut-in
state, which for CO₂ requires integrating dp/dz = −ρ(p,T₀)g with the real
EOS (the ideal barometric formula errs by 1/Z ≈ 45 % in the gradient).
Initial superheat at the wellhead: 12.6 K; at the (hydrostatically
compressed, ~5.9 MPa) bottom: ~6.5 K. A wellhead pressure of 6.5 MPa was
shown to be inadmissible: the hydrostatic column crosses p_sat(300 K) =
6.71 MPa at depth, i.e. the initial condition itself is two-phase — caught
by the monitor at t = 0. Everything else (geometry, grid, choke, friction,
thermal models, tolerances, output cadence, sensors, post-processing) is
identical between the fluids.

## 6. Verification summary (all automated)

1. Table closure vs direct CoolProp: < 10⁻³ max (measured ~5×10⁻⁵).
2. Speed-of-sound consistency: tabulated a vs finite-difference
   (∂p/∂ρ)_s along CoolProp isentropes, < 0.2 %.
3. Real-gas orifice: exceeds ideal at 5 MPa (bounded), recovers ideal at
   0.3 MPa within 4 %.
4. Low-pressure limit: full real-fluid solver vs ideal-gas CO₂ model at
   0.5 MPa — ṁ(t) within 5 %.
5. Cross-path check: real-fluid *nitrogen* through the general code path
   vs the validated ideal-gas path — ṁ(t) within 5 %, budgets exact.
6. Dome-entry flagging: low-margin case stops within 1 s with a complete
   event record.
7. Mass and energy budgets at machine precision for the real-fluid path
   (the budget bookkeeping uses h₀ = (E_f+p_f)/ρ_f, exact for any EOS).

## 7. Known approximations

* Transport properties (μ, Pr) and the c_p used inside the Dittus–Boelter
  correlation are constants (gas-phase reference values); the energy
  equation itself never uses them.
* The linearized supply-side BC assumes Δp_face ≪ ρa² (valid for
  choke-dominated discharge; the exact solver remains available).
* Table envelope limits T ≥ 220 K and ρ ≤ 300 kg/m³; states outside are
  clipped, and in practice the dome monitor halts first.
