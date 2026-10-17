# Coupled Completion/Formation Thermal Model (Phase 4)

Module: `hyperbolic1d.thermal`. Implements the preferred conjugate
formulation of the thermal hierarchy: at every axial cell, a 1-D radial
thermal network represents the concentric completion layers and the
surrounding formation; it is advanced implicitly each flow step and coupled
to the wellbore energy equation through the fluid-side convective
conductance. `q'' = -k_f ∂T_f/∂r |_{r_w}` emerges from the network rather
than being prescribed.

## 1. Radial stack — LSU CO₂ research well (upper 2,000 ft)

From the facility design slides (casing table + API wall thicknesses):

| Region | r_in → r_out [mm] | Model |
|---|---|---|
| Fluid bore (7-5/8" ID, 35.8#) | 0 → 85.2 | 1-D wellbore flow; Dittus–Boelter h (laminar floor Nu = 3.66) |
| 7-5/8" CRA casing (13% Cr) | 85.2 → 96.8 | solid: k = 19, ρ = 7750, cp = 460 |
| Air annulus | 96.8 → 156.8 | massless nonlinear conductance (below) |
| 13-3/8" casing (72#, L80) | 156.8 → 169.9 | solid: k = 45, ρ = 7850, cp = 490 |
| Cement A (inside 20") | 169.9 → 237.9 | solid: k = 0.9, ρ = 2000, cp = 900 (2 shells) |
| 20" surface casing (133#) | 237.9 → 254.0 | solid |
| Cement B (26" borehole) | 254.0 → 330.2 | solid (2 shells) |
| Formation | 330.2 → 5,000 | k = 2.1, ρ = 2300, cp = 1000; 16 log-spaced shells |

Outer boundary: undisturbed geothermal T(z) = T_surface + Γ·depth
(defaults 294.3 K, 25.5 K/km — Gulf-coast). For blowdown time scales the
rock penetration depth √(αt) is centimeters, so r_far = 5 m is generous.
The 30" conductor (top ~150 ft only) is neglected. Every number above is a
parameter of `lsu_co2_well_stack()` / the layer dataclasses.

**Air annulus.** Natural convection is expected to dominate (user-specified
configuration: annulus currently air-filled). Modeled as the
wellbore-standard Willhite (1967) treatment: effective conductivity from
the Dropkin & Somerscales correlation,
k_eff = 0.049 (Gr Pr)^{1/3} Pr^{0.074} k_air (floored at pure conduction,
Gr on the gap width), in parallel with gray-body radiation between
concentric cylinders (ε = 0.8 both surfaces, linearized conductance). The
annulus is treated as massless (ρcp of air negligible against steel) and
its conductance is re-evaluated every step from the current surface
temperatures (coefficient lagging), keeping the implicit solve linear.

## 2. Numerics and coupling

Solid shells carry capacitance C' = ρcp·π(r_out²−r_in²) lumped at the
log-mean radius, connected by cylindrical-shell conduction resistances
ln(r_b/r_a)/(2πk). Backward-Euler per step; one tridiagonal (Thomas) solve
per axial cell, vectorized across cells. The fluid temperature enters the
solid solve explicitly; the returned wall flux uses the *implicit* wall
temperature through the same conductance, so the energy exchanged by the
fluid source term equals the energy absorbed by the network exactly. The
wellbore solver's global budget (total + potential energy + enthalpy
outflux − wall heat input) therefore still closes to machine precision —
verified for the conjugate case.

## 3. Verification (tests/test_thermal.py — all passing)

1. **Steady state:** interface heat flows identical across every radial
   interface (series-resistance consistency), to 1e-6.
2. **Energy conservation:** with an insulated outer boundary, heat
   absorbed from the fluid equals the change of stored network energy to
   1e-10 (backward Euler is exactly conservative).
3. **Constant-temperature cylinder:** a formation-only network matches an
   independent reference solution (6000-node log-radial Crank–Nicolson,
   `scripts/make_thermal_reference.py`) to < 1% over τ = ατ/r_w² ∈
   [10, 1000]. **Negative result worth recording:** three convenient
   references were tried first and all proved inadequate — the
   leading-order log asymptote is 8–15% off in this τ range; Ramey /
   Hasan–Kabir-type T_D correlations address the constant-*flux* boundary
   condition, not constant temperature; and naive quadrature of the exact
   Carslaw–Jaeger integral mis-handles the singular u→0 region that
   dominates at large τ. The network initially appeared 3–7% "in error"
   against these; the independent FD reference showed the network was
   right to 0.3% and the references wrong.
4. **Radial grid convergence** of q''(t) under formation-shell refinement.
5. **Annulus conductance:** positive, monotone in ΔT, bounded below by
   conduction + radiation.
6. **Conjugate blowdown budgets** at machine precision.

## 4. Model limitations / notes

* The flow bore is treated as a clean circle at the 7-5/8" ID; the 2-7/8"
  tubing, injection line and gauges inside the casing are not yet
  represented (they reduce flow area and add internal surfaces). A
  annulus-flow hydraulic option is a straightforward extension if the
  blowdown is to be run with the completion in place.
* Dittus–Boelter is used for the fluid-side h with constant transport
  properties; adequate for N₂, to be revisited for transcritical CO₂
  (property variation across the boundary layer).
* Axial conduction in casings/formation is neglected (radial Peclet
  reasoning; axial gradients are ~K/100 m).
* The annulus correlation set (Willhite/Dropkin–Somerscales) is the
  standard wellbore choice but was developed for modest gap aspect
  ratios; if annulus heat transfer proves to be a controlling resistance,
  a dedicated tall-annulus correlation study is warranted.
