"""Persona registry — one place that knows every investor's key."""

from __future__ import annotations

from importlib import import_module

from v2.personas.base import Persona

#: ``key -> (module, class name)``. Import lazily so a syntax slip in one
#: persona does not take the whole package down.
_SPECS: dict[str, tuple[str, str]] = {
    "warren_buffett": ("v2.personas.warren_buffett", "WarrenBuffett"),
    "charlie_munger": ("v2.personas.charlie_munger", "CharlieMunger"),
    "ben_graham": ("v2.personas.ben_graham", "BenGraham"),
    "peter_lynch": ("v2.personas.peter_lynch", "PeterLynch"),
    "phil_fisher": ("v2.personas.phil_fisher", "PhilFisher"),
    "bill_ackman": ("v2.personas.bill_ackman", "BillAckman"),
    "cathie_wood": ("v2.personas.cathie_wood", "CathieWood"),
    "michael_burry": ("v2.personas.michael_burry", "MichaelBurry"),
    "mohnish_pabrai": ("v2.personas.mohnish_pabrai", "MohnishPabrai"),
    "stanley_druckenmiller": ("v2.personas.stanley_druckenmiller", "StanleyDruckenmiller"),
    "aswath_damodaran": ("v2.personas.aswath_damodaran", "AswathDamodaran"),
    "nassim_taleb": ("v2.personas.nassim_taleb", "NassimTaleb"),
    "rakesh_jhunjhunwala": ("v2.personas.rakesh_jhunjhunwala", "RakeshJhunjhunwala"),
}

PERSONAS: tuple[str, ...] = tuple(_SPECS)
_CACHE: dict[str, Persona] = {}


def get_persona(key: str) -> Persona:
    if key not in _SPECS:
        raise KeyError(f"unknown persona {key!r}; known: {', '.join(PERSONAS)}")
    if key not in _CACHE:
        module_name, class_name = _SPECS[key]
        _CACHE[key] = getattr(import_module(module_name), class_name)()
    return _CACHE[key]


def list_personas(keys: list[str] | tuple[str, ...] | None = None) -> list[Persona]:
    return [get_persona(k) for k in (keys or PERSONAS)]
