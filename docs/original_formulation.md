# Analysis of the Original Fortran Codes and Numerical Formulation

**Source material.** Two artifacts were available in `CO2 - HyperCO2/`:

1. **Gessner, T. R. (2010).** *Modelagem numérica do escoamento anular gás-líquido
   transiente pelo método da divisão da matriz de coeficientes.* M.Sc. thesis,
   Federal University of Santa Catarina (advisor: J. R. Barbosa Jr.).
   `Dissertacao_TobiasGessner_REVB.pdf`. This is the primary document for the
   shock-tube work: single-phase (Ch. 3), two-phase homogeneous (Ch. 4),
   two-fluid/separated and annular models (Chs. 5–6).
2. **`ACOPLAMENTO HE/`** — a Compaq/Visual Fortran project containing
   `P1.FOR` (a single-phase transient *wellbore* code coupling a reservoir
   inflow boundary to a vertical pipe, with friction and gravity), `GBAND.for`
   (the banded solver), `DADOS.DAT` (inputs), and REFPROP 7.0 object files.
   A variant `versao reinjecao/P1.FOR` differs only in the inlet-entropy
   reference node (injection instead of production). **The shock-tube Fortran
   programs themselves are not in this folder** — only the thesis documents
   them — so Phase 1 validation is against (a) the exact analytical solutions
   and (b) the error tables and figures reported in the thesis.

The recollected ingredients map onto the sources as follows:

| Recollection | What it is |
|---|---|
| Conservative / non-conservative formulations | Thesis Secs. 3.1.1–3.1.2: unknowns `[P, u, s]` vs `[ρ, ρu, E]` |
| Entropy-based formulation | Entropy is a primary unknown in the non-conservative form; energy conservation is written as an entropy-transport equation (Eq. 3.11); an analytic Rankine–Hugoniot entropy correction (Eq. 3.61) repairs its isentropic bias at shocks |
| Split matrix coefficient method | The **Split Coefficient Matrix Method (SCMM)** of Chakravarthy, Anderson & Salas (1980), extended to two-phase by Romstedt (1990) and Städtke (2006) |
| "G-band" | `GBAND`, the banded Gaussian-elimination solver (no pivoting) from Aziz & Settari, *Petroleum Reservoir Simulation* (1979), used for the block-banded implicit systems |

---

## 1. Governing equations (single phase, Ch. 3)

### 1.1 Non-conservative formulation — unknowns $\vec U = [P,\,u,\,s]^T$

Mass, momentum and energy conservation are recast (Eqs. 3.5, 3.11, 3.19) as

$$\frac{\partial P}{\partial t} + u\frac{\partial P}{\partial x} + \rho a^2 \frac{\partial u}{\partial x} = \frac{\beta a^2}{c_p}\left(\dot Q^w + F^w u\right)$$

$$\frac{\partial u}{\partial t} + u\frac{\partial u}{\partial x} + \frac{1}{\rho}\frac{\partial P}{\partial x} = -\frac{F^w}{\rho} - g_x$$

$$\frac{\partial s}{\partial t} + u\frac{\partial s}{\partial x} = \frac{1}{\rho T}\left(\dot Q^w + F^w u\right)$$

i.e. $\partial_t \vec U + G\,\partial_x \vec U = \vec C$ with

$$G=\begin{bmatrix} u & \rho a^2 & 0\\ 1/\rho & u & 0\\ 0 & 0 & u\end{bmatrix},\qquad
a^2 = \left(\frac{\partial P}{\partial \rho}\right)_s,\quad \beta = -\frac{1}{\rho}\left(\frac{\partial\rho}{\partial T}\right)_P .$$

For the shock tube all source terms vanish. Because $s$ is transported
isentropically, this formulation cannot produce the entropy rise across a
shock; the thesis tracks the shock and corrects the entropy field with the
ideal-gas Rankine–Hugoniot jump (Eq. 3.61),

$$\Delta s = c_v \ln\!\left[ R_p\left(\frac{(\chi-1)R_p+(\chi+1)}{(\chi+1)R_p+(\chi-1)}\right)^{\chi}\,\right],\qquad R_p = \frac{p_2}{p_1},\ \chi=\frac{c_p}{c_v}.$$

The tracking/correction *algorithm* is not spelled out in the thesis; the
Python port implements a documented interpretation (contact surface advected
with the local velocity; right-chamber material between contact and shock set
to $s_R+\Delta s$).

### 1.2 Conservative formulation — unknowns $\vec V = [\rho,\,G,\,E]^T$

With $G=\rho u$, $E=\rho(e+u^2/2)$, the system is transformed by the Jacobian
$J=\partial\vec V/\partial\vec U$ into $\partial_t\vec V + H\,\partial_x\vec V = \vec D$,
$H = JGJ^{-1}$ (Eq. 3.30). For a perfect gas $H$ coincides with the Euler flux
Jacobian $\partial F/\partial V$. **Important structural point:** the SCMM
discretizes the *quasi-linear* form $H\,\partial_x \vec V$, not the flux
divergence $\partial_x F(\vec V)$; even the "conservative" variant is
therefore not discretely conservative at shocks — it merely uses conservative
*variables*. This matters for shock speeds and is the principal theoretical
motivation for the Phase-2 finite-volume comparison.

## 2. The Split Coefficient Matrix Method (Sec. 2.5.1)

Eigen-decomposition $G = T\Lambda T^{-1}$, with
$\lambda_{1,2}=u\pm a$, $\lambda_3=u$ (and eigenvectors Eq. 3.42), is used to
build matrices associated with positive/negative characteristic speeds:

$$G^{\pm} = T\,\Lambda^{\pm}\,T^{-1},\qquad \Lambda^{\pm}=\tfrac12\left(\Lambda\pm|\Lambda|\right),$$

so that $\partial_t\vec U + G^+\partial_x\vec U + G^-\partial_x\vec U = \vec C$.
Upwinding is then applied term-by-term:

* $G^+\partial_x\vec U$ → **backward** differences (information from upstream),
* $G^-\partial_x\vec U$ → **forward** differences,

with one-sided stencils of **first order** (2-point) or **second order**
(3-point, Eqs. 2.17/2.21; on a uniform grid the coefficients are
$\tfrac{1}{2\Delta x}[\,q_{i-2}-4q_{i-1}+3q_i\,]$ and its mirror). Central
differences are never used. Time discretization is **implicit backward
Euler**; within each time step the coefficient matrices are refreshed from
the latest iterate and the linear system re-solved until the fields converge
(Picard iteration; tolerances in thesis Tables 3.2/4.3).

### 2.1 Linear system and GBAND

The implicit system is block-banded: 11 scalar diagonals for first order
(3-eq. system), 17 for second order; two-phase: 23 (4-eq. equilibrium) and
29 (5-eq. non-equilibrium) diagonals. `GBAND` performs banded Gaussian
elimination **without pivoting** (Aziz & Settari, 1979). The Python port uses
LAPACK `gbsv` (partial pivoting) via `scipy.linalg.solve_banded`; this is the
same elimination with added robustness and does not change converged results.

### 2.2 Boundary treatment (Sec. 2.5.1.2, Table 3.1)

* Node 1: the $G^+$ derivative is dropped (no upstream nodes); node $N$: the
  $G^-$ derivative is dropped.
* Nodes 2 and $N-1$: one-sided stencils limited to first order.
* Closed-tube walls: $u=0$ (or $G=0$) prescribed at both ends; $P,s$ (or
  $\rho,E$) at the walls evolve using only the inward-propagating
  characteristics.

## 3. Single-phase shock-tube benchmark (Sec. 3.4)

| Item | Value |
|---|---|
| Fluid | Air, perfect gas |
| Geometry | 4 m closed horizontal tube, diaphragm at 2.0 m |
| Left chamber | 500 kPa, 400 K, at rest |
| Right chamber | 100 kPa, 300 K, at rest |
| Grid | 4001 nodes, $\Delta x = 1$ mm |
| Time step | $\Delta t = 10^{-6}$ s, output at $t=3$ ms |
| Reference | Exact solution (Anderson, 1982) |

Thesis Table 3.3 — mean absolute errors over the 4001 nodes at $t=3$ ms
(the Phase-1 validation targets):

| Formulation | P [Pa] | U [m/s] | ρ [kg/m³] | T [K] | CPU [s]¹ |
|---|---|---|---|---|---|
| Non-conservative, 1st order | 3339 | 4.65 | 0.0253 | 2.14 | 90 |
| Non-conservative, 2nd order | 2713 | 3.87 | 0.0201 | 1.67 | 203 |
| Conservative, 1st order | 1555 | 2.02 | 0.0161 | 1.74 | 178 |
| Conservative, 2nd order | 311.6 | 0.331 | 0.00354 | 0.310 | 336 |

¹ circa-2009 hardware; useful only for relative comparisons.

Thesis findings: conservative variables track the shock position better;
first order shows numerical diffusion, second order numerical dispersion
(oscillations near discontinuities — the scheme has no limiter); second-order
conservative is the most accurate and most expensive.

## 4. Two-phase homogeneous model (Ch. 4)

Both phases share one velocity $u$ (homogeneous model, valid for dispersed
patterns). Two closures:

* **Non-equilibrium** (5 eqs.): $\vec U = [P,\ \alpha_g,\ u,\ s_g,\ s_l]^T$,
  phase temperatures coupled through an interfacial heat flux
  $\dot Q_g^{int} = -C_T\,\alpha_{l\,(\text{or }g)}\,(T_g-T_l)$ with a tunable
  volumetric coefficient $C_T$ (numerically unstable above
  $C_T \sim 10^9\,\mathrm{W\,m^{-3}K^{-1}}$).
* **Equilibrium** (4 eqs.): $T_g=T_l$; the liquid-entropy equation is replaced
  by a thermodynamic relation that fixes $\dot Q_g^{int}$ (Eqs. 4.71/4.73).

The coefficient matrix (Eq. 4.25) has eigenvalues $u\pm a$ (pressure waves),
and $u$ (void-fraction wave + entropy waves), with the homogeneous
**mixture sound speed** (a Wood-type formula, Eq. 4.27):

$$a = \sqrt{\dfrac{\rho_g a_g^2\, \rho_l a_l^2}{\rho\left(\alpha_g \rho_l a_l^2 + \alpha_l \rho_g a_g^2\right)}},\qquad \rho = \alpha_g\rho_g + \alpha_l\rho_l .$$

**Two-phase shock-tube benchmark (Sec. 4.5, after Städtke 2006):** air–water
dispersion, 4 m tube, diaphragm at 2.5 m; left: 500 kPa, 400 K,
$\alpha_g=0.05$; right: 100 kPa, 300 K, $\alpha_g=0.10$; 4001 nodes,
$\Delta t = 5\times10^{-6}$ s, $t_{end}=15$ ms. Gas: ideal; liquid properties
from REFPROP 7.0. Reference: Städtke's iterative analytical solution.
Reported mean errors (non-equilibrium, $C_T=10^9$): 7.4 kPa, 0.0016 (α),
0.053 m/s; equilibrium: 7.8 kPa, 0.0018, 0.061 m/s.

## 5. The wellbore code `P1.FOR` (relevant to Phase 3)

`P1.FOR` is **not** the shock-tube code; it is a single-phase transient
vertical-well simulator built on the same machinery and is the natural
reference for Phase 3:

* Non-conservative $[P, u]$ + segregated entropy equation $s$ (solved in an
  outer loop, first-order upwind on $u^{\pm}$), second-order SCMM stencils
  with the exact $(β,φ)$ non-uniform-grid coefficients of Eqs. (2.17)/(2.21);
* Characteristic 2×2 blocks $G_{1,2}$ built from $(u\pm a)$ — the split
  matrices of Sec. 3.2.1 restricted to $[P,u]$;
* Source terms: gravity ($-9.81$) and Colebrook-White friction
  $f|u|u/2D$ (iterative solution for $f$, capped at $10^3$);
* Ideal gas; properties updated from $(P,s)$ exactly as in CalcPROP:
  $T = T_{ref}\,e^{s/c_p}(P/P_{ref})^{R/c_p}$, $T_{ref}=300$ K,
  $P_{ref}=101325$ Pa;
* Boundary conditions: prescribed inlet mass flux from a `RESERVOIR`
  subroutine (constant $G_{in}=50$ kg/m²s, $T_{in}=300$ K in the archived
  version — a stub for an inflow-performance relation) at the bottom node and
  prescribed outlet pressure at the top; an outermost loop iterates the
  inlet flux to convergence;
* Nested convergence loops: pressure/velocity (inner, GBAND on 2N unknowns),
  entropy (outer, GBAND on N unknowns), reservoir coupling (outermost);
* `DADOS.DAT` in the archive is configured for a 1 m × 0.1 m pipe test with
  methane-like molar mass (16) but CP overridden to 1005 in `INIT` — i.e. a
  development configuration, not a production case.

## 6. Known limitations of the original method (inputs to Phase 2)

1. **No discrete conservation** — quasi-linear (non-flux) differencing in
   both formulations; shock speeds are only approximately captured, and the
   non-conservative form needs an external entropy correction.
2. **No limiters** — second order is linear and oscillatory at
   discontinuities (dispersive); first order is strongly diffusive
   (Godunov's theorem in action).
3. **Implicit Picard + banded solves** are expensive per step relative to an
   explicit FV update at CFL ≈ 1 on the same grid, and the implicit solver is
   run at CFL < 1 anyway for accuracy (thesis: Δt = 1 µs ⇒ CFL ≈ 0.4–0.6).
4. **Split matrices assume subsonic flow** in the boundary logic (λ₁ > 0 >
   λ₂); choked/transonic wellbore flow (expected in CO₂ blowdown) needs care.
5. **Entropy-based non-conservative form** couples naturally to arbitrary
   EOS via $(a, β, c_p)$ — an advantage worth retaining for real-fluid CO₂ —
   but inherits points 1–2.
