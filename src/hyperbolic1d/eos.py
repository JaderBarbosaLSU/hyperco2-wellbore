"""Equation-of-state interface for the wellbore finite-volume solver.

The FV solver and the choke boundary condition interact with the fluid
*only* through this interface, so that the ideal-gas model used for the
nitrogen verification stage can later be replaced by a real-fluid CO2
model (CoolProp/REFPROP span-wagner, or a tabulated flash) without
touching the discretization.

Required capabilities (all vectorized over cells unless noted):

* ``primitives(rho, e)``      -> (p, T, a) from density and specific
                                 internal energy — the conserved-to-
                                 primitive closure used every step;
* ``rho_e_from_pT(p, T)``     -> (rho, e) for initialization;
* ``enthalpy(p, rho, e)``     -> specific enthalpy;
* ``isentrope(p, p_ref, rho_ref)`` -> density on the isentrope through
                                 (p_ref, rho_ref) — used by the outlet
                                 boundary condition;
* ``sound_speed(p, rho)``     -> speed of sound.

For a real fluid the isentrope and the Riemann-invariant integral in the
boundary condition must be evaluated numerically; the boundary-condition
module is written against scalar EOS calls for exactly this reason.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["IdealGasEOS", "NITROGEN", "AIR", "CO2_IDEAL"]


@dataclass(frozen=True)
class IdealGasEOS:
    """Calorically perfect gas: p = rho R T, e = cv T."""

    R: float
    gamma: float
    name: str = "ideal gas"

    @property
    def cp(self) -> float:
        return self.gamma * self.R / (self.gamma - 1.0)

    @property
    def cv(self) -> float:
        return self.R / (self.gamma - 1.0)

    # --- conserved-variable closure ---------------------------------------
    def primitives(self, rho, e):
        p = (self.gamma - 1.0) * rho * e
        T = e / self.cv
        a = np.sqrt(self.gamma * p / rho)
        return p, T, a

    def rho_e_from_pT(self, p, T):
        rho = p / (self.R * T)
        e = self.cv * T
        return rho, e

    def enthalpy(self, p, rho, e=None):
        return (self.cp / self.R) * p / rho  # cp*T

    def sound_speed(self, p, rho):
        return np.sqrt(self.gamma * p / rho)

    def temperature(self, p, rho):
        return p / (self.R * rho)

    # --- isentropic relations (used by the choke/outlet BC) ----------------
    def isentrope(self, p, p_ref, rho_ref):
        """Density at pressure p on the isentrope through (p_ref, rho_ref)."""
        return rho_ref * (p / p_ref) ** (1.0 / self.gamma)

    def riemann_invariant_velocity(self, p, p_ref, rho_ref, u_ref):
        """Outgoing-wave (C+) relation: velocity at pressure p on the
        isentrope through (p_ref, rho_ref), from state (u_ref, p_ref):

            u(p) = u_ref + 2/(gamma-1) * (a_ref - a(p)).
        """
        a_ref = self.sound_speed(p_ref, rho_ref)
        a = a_ref * (p / p_ref) ** ((self.gamma - 1.0) / (2.0 * self.gamma))
        return u_ref + 2.0 / (self.gamma - 1.0) * (a_ref - a)

    # --- compressible-orifice (choke) mass flux ----------------------------
    def critical_pressure_ratio(self) -> float:
        g = self.gamma
        return (2.0 / (g + 1.0)) ** (g / (g - 1.0))

    def orifice_mass_flux(self, p0, T0, p_back):
        """Isentropic-nozzle mass flux per unit throat area from stagnation
        (p0, T0) to back pressure p_back; chokes automatically.
        Returns (G_throat [kg/m^2/s], choked [bool])."""
        g = self.gamma
        r = max(p_back / p0, 0.0)
        r_crit = self.critical_pressure_ratio()
        if r <= r_crit:  # choked
            G = p0 * np.sqrt(g / (self.R * T0)) * (2.0 / (g + 1.0)) ** (
                (g + 1.0) / (2.0 * (g - 1.0)))
            return G, True
        if r >= 1.0:
            return 0.0, False
        G = p0 * np.sqrt(
            2.0 * g / ((g - 1.0) * self.R * T0) * (r ** (2.0 / g) - r ** ((g + 1.0) / g))
        )
        return float(G), False


NITROGEN = IdealGasEOS(R=296.80, gamma=1.40, name="N2 (ideal)")
AIR = IdealGasEOS(R=287.00, gamma=1.40, name="air (ideal)")
# placeholder for the first CO2 step (ideal-gas cp at ~300 K); the real-gas
# model replaces this class, not the solver
CO2_IDEAL = IdealGasEOS(R=188.92, gamma=1.289, name="CO2 (ideal, ~300 K)")
