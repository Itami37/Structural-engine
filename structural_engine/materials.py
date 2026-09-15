"""Concrete and steel constitutive data (SI: Pa, kg/m^3)."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class Concrete:
    name: str = "C30"
    fc: float = 30e6
    Ec: float = 30e9
    ft: float | None = None
    nu: float = 0.2
    rho: float = 2400.0

    def __post_init__(self):
        if self.ft is None:
            # ACI 318: fr = 0.62 * sqrt(f'c)  with f'c and fr both in MPa.
            # fc is stored in Pa, so convert in, compute, convert back out.
            fc_MPa = self.fc / 1e6
            self.ft = 0.62 * (fc_MPa ** 0.5) * 1e6  # Pa


@dataclass
class Steel:
    name: str = "Fe500"
    fy: float = 500e6
    Es: float = 200e9
    fu: float | None = None
    nu: float = 0.3
    rho: float = 7850.0

    def __post_init__(self):
        if self.fu is None:
            self.fu = 1.1 * self.fy