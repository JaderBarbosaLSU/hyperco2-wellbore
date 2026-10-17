# Wellbore Blowdown Model — Formulation and Verification (Phase 3, Stage 1)

Module: `hyperbolic1d.wellbore` (+ `hyperbolic1d.eos`). Stage 1 implements the
adiabatic, single-phase, ideal-gas (N₂) blowdown of a closed-bottom vertical
well discharging through a wellhead choke, on the Godunov/HLLC architecture
selected in Phase 2.

## 1. Governing equations

With $z$ measured upward from the closed bottom ($z=0$) to the wellhead
($z=L$), constant flow area $A$, diameter $D$:

$$\frac{\partial \rho}{\partial t} + \frac{\partial (\rho u)}{\partial z} = 0$$

$$\frac{\partial (\rho u)}{\partial t} + \frac{\partial (\rho u^2 + p)}{\partial z} = -\rho g - \frac{f_D\,\rho u |u|}{2D}$$

$$\frac{\partial E}{\partial t} + \frac{\partial\left[u(E+p)\right]}{\partial z} = -\rho u g + \frac{4 q_w''}{D}$$

where $E = \rho\,(e + u^2/2)$, $q_w''$ is the wall heat flux into the fluid
(zero for the adiabatic baseline), and $f_D$ the Darcy friction factor
(Churchill 1977 correlation; disabled in the baseline). Wall shear enters
only the momentum equation: a stationary wall does no net work, and
frictional dissipation is recovered automatically by the conservative
total-energy formulation.

**EOS abstraction.** The solver closes the system only through the interface
in `eos.py` (`primitives(rho, e) -> (p, T, a)`, isentrope, Riemann-invariant
velocity, orifice flux). `IdealGasEOS` (N₂: R = 296.8 J/kg·K, γ = 1.4)
implements it analytically; a real-fluid CO₂ model replaces this class and
evaluates the same calls numerically — the FV solver and the boundary
condition are untouched.

## 2. Numerics

MUSCL-Hancock (primitive-variable reconstruction, van Leer limiter) with
HLLC fluxes — the kernel validated in Phase 2 — plus:

* **gravity in the Hancock half step** (formal 2nd order with sources);
* **Heun corrector** for momentum-gravity, friction and wall-heat sources;
* **energy-consistent gravitational work**: the energy source $-\rho u g$ is
  discretized as $-g\,(F^{\rho}_{j-1/2}+F^{\rho}_{j+1/2})/2$ using the
  time-centered face mass fluxes. With this form the discrete budget

  $$\underbrace{\sum_j (E_j + \rho_j g z_j)\,A\,\Delta z}_{\text{total + potential energy}} \;+\; \int_0^t \dot m\,(h_0 + gL)\,dt' \;=\; \text{const}$$

  telescopes exactly (machine precision), whereas the naive cell-centered
  $-\rho u g$ left a non-convergent $O(10^{-4})$ budget residual — found and
  fixed during verification;
* **exact solid wall** at the bottom: the wall-face left state is the mirror
  of the reconstructed interior right state, which makes the HLLC wall mass
  flux identically zero.

The scheme is **not well-balanced**: a closed well initialized in isothermal
hydrostatic equilibrium retains residual velocities ~5 cm/s (200 cells).
Irrelevant during blowdown transients; revisit (e.g. hydrostatic
reconstruction) if long-time post-blowdown equilibria become of interest.

## 3. Wellhead choke boundary condition

The wellhead is **not** a fixed-pressure boundary. The pipe-side face state
$(\rho_f, u_f, p_f)$ at $z = L$ is required to satisfy simultaneously
(`wellhead_face_state`):

1. **Interior coupling** — the face lies on the isentrope through the last
   interior cell state and on its outgoing $C^+$ characteristic:
   $u_f = u_i + \frac{2}{\gamma-1}\,(a_i - a_f)$;
2. **Valve mass balance** — $\rho_f u_f A = C_d A_t\, G(p_{0f}, T_{0f}, p_b)$,
   where $(p_{0f}, T_{0f})$ is the stagnation state of the face flow and $G$
   the isentropic-nozzle mass flux, automatically **choked** when
   $p_b/p_{0f} \le (2/(\gamma+1))^{\gamma/(\gamma-1)}$ and subcritical
   otherwise.

The scalar unknown $p_f$ is bracketed between the **sonic face pressure**
(the $C^+$ coupling forbids a supersonic pipe exit; if the valve demands
more than the sonic pipe supply, the exit runs sonic — the open-pipe
choking limit $A_t \to A$) and the pressure at which the invariant brings
the face to rest (compression branch), and solved with Brent iteration.
If the interior stagnation pressure falls below $p_b$, the valve closes
(check-valve clamp; no backflow). The face flux
$[\rho_f u_f,\ \rho_f u_f^2 + p_f,\ u_f(E_f+p_f)]$ replaces the HLLC flux at
the top face, so the interior update and the discharge are automatically
consistent, through the entire choked → unchoked → closed sequence as the
well depressurizes.

## 4. Baseline case

| Item | Value |
|---|---|
| Fluid | N₂, ideal gas (R = 296.8, γ = 1.4) |
| Depth | 2,000 ft = 609.6 m |
| Well ID | 0.100 m (parameter) |
| Initial state | uniform 13.79 MPa (2,000 psi), 300 K, at rest |
| Choke | 12.7 mm bore, C_d = 0.84, back pressure 101.325 kPa |
| Walls | adiabatic; friction off (baseline) |

## 5. Verification results

`scripts/run_blowdown_verification.py`; numbers in
`results/blowdown_verification.json`, figures in `results/blowdown_*.png`.

1. **Conservation.** Closed well: fields remain exactly at rest. Open well,
   gravity on: mass and energy budgets (incl. potential energy and enthalpy
   outflux) close to machine precision (~1e-16) — by construction, see §2.
2. **0-D vessel limit (gravity off).** The acoustic transit time (~1.7 s) is
   small against the discharge time (~300 s), so the 1-D solution must
   approach the lumped adiabatic-vessel ODE. Agreement in wellhead pressure
   and mass flow to a fraction of a percent after the initial wave transient
   (quantified in the JSON report).
3. **Grid independence.** $\dot m(t)$ for 100/200/400 cells vs 800 cells.
4. **CFL sensitivity.** CFL = 0.2/0.5 vs 0.9.
5. **Choke sensitivity.** Bore sweep 1/4 in → open pipe; monotonic mass
   flow, stable in the open-pipe (pipe-choked) limit, conservation holds
   for all cases.

## 6. Thermal-model hierarchy (interface in place)

`wall_model` accepts any object with
`heat_flux(z, t, T_fluid, rho, u, p, geom, eos)`:

1. `AdiabaticWall` — baseline (this stage, verified);
2. `PrescribedWallTemperature` — fixed $T_w(z)$, Dittus-Boelter $h$
   (implemented; benchmark use);
3. prescribed/calculated flux — trivial to add against the same interface;
4. **coupled formation model** (next stage): transient radial conduction in
   completion layers + formation, coupled through
   $q'' = -k_f\,\partial T_f/\partial r |_{r_w}$, with tubing/casing/cement
   resistances and capacitances as lumped or discretized radial nodes.

## 7. Known limitations / forward notes

* Ideal-gas N₂ only at this stage; CO₂ real-gas model replaces `IdealGasEOS`
  (Joule-Thomson and dense-phase effects absent until then — N₂ blowdown
  cooling here is pure expansion work).
* The choke model is quasi-steady and isentropic to the throat; no valve
  dynamics, no friction inside the restriction beyond $C_d$.
* Single-phase; no liquid loading or condensation.
* Not well-balanced (§2).
