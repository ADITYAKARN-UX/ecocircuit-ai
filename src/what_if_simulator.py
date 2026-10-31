"""
EcoCircuit AI — Sustainability What-If Simulator (Module F)
===========================================================
Interactive scenario comparison for e-waste reduction strategies.

All results are MODELLED ESTIMATES based on user-supplied parameters.
They are NOT measured environmental outcomes.
All equations and assumptions are documented below.

Equations:
  total_ewaste = new_production * (1 - reuse_rate) * (1 - collection_rate)
               + (new_production * reuse_rate - reused_returned) * eventual_disposal_rate

  Simplified model used here:
    ewaste_generated = production_volume * (1 - collection_rate - reuse_rate)
                     (constrained to >= 0)

  e-waste_reduction = baseline_ewaste - scenario_ewaste

  recovered_value = production_volume * reuse_rate * recovery_value_per_unit

  CO2 equivalent avoided (illustrative estimate only):
    Based on published per-unit lifecycle CO2e factors from UNEP / GEM literature.
    These are ESTIMATES. Units: kg CO2e per device.

Author: EcoCircuit AI Team
"""

import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

logger = logging.getLogger("what_if_simulator")

# ── Published per-unit CO2e estimates (illustrative; sources below) ─────────
# Source: Various lifecycle assessment studies cited in GEM 2024 supplementary.
# These are ORDER-OF-MAGNITUDE estimates for illustration.
# Do NOT use as precise carbon accounting figures.
CO2E_PER_UNIT_KG = {
    "Smartphone": 70,
    "Tablet": 130,
    "Laptop": 340,
    "Desktop Computer": 600,
    "Monitor": 400,
    "Television": 600,
    "Refrigerator": 1000,
    "Washing Machine": 800,
    "Air Conditioner": 900,
    "Microwave": 200,
    "Printer": 300,
    "Router": 80,
    "Server": 3000,
    "Default": 200,
}

WEIGHT_KG_PER_UNIT = {
    "Smartphone": 0.18,
    "Tablet": 0.45,
    "Laptop": 2.0,
    "Desktop Computer": 8.0,
    "Monitor": 4.5,
    "Television": 12.0,
    "Refrigerator": 65.0,
    "Washing Machine": 60.0,
    "Air Conditioner": 30.0,
    "Microwave": 12.0,
    "Printer": 8.0,
    "Router": 0.5,
    "Server": 20.0,
    "Default": 5.0,
}


@dataclass
class ScenarioConfig:
    """Configuration for a single sustainability scenario."""
    name: str
    production_volume: float          # units produced
    product_category: str = "Default"
    lifespan_years: float = 5.0       # extended lifespan
    repair_rate: float = 0.0          # fraction of units repaired
    reuse_rate: float = 0.0           # fraction reused / donated
    collection_rate: float = 0.25     # fraction formally collected at EOL
    recycling_efficiency: float = 0.75 # fraction of collected waste actually recycled
    recovery_yield: float = 0.0       # fraction of components recovered for reuse
    description: str = ""


@dataclass
class ScenarioResult:
    """Output of a scenario simulation."""
    name: str
    ewaste_generated_units: float
    ewaste_generated_kg: float
    formally_collected_units: float
    recycled_units: float
    recovered_units: float
    co2e_avoided_kg: float           # vs. no circular action
    production_volume: float
    effective_new_production_needed: float
    assumptions: List[str] = field(default_factory=list)
    caveats: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


# ── Core simulator ─────────────────────────────────────────────────────────

def simulate_scenario(cfg: ScenarioConfig) -> ScenarioResult:
    """
    Simulate a single sustainability scenario.

    Simplified e-waste model:
        ewaste_at_eol = production_volume
                        * (1 - reuse_rate)        → units entering formal/informal EOL
                        * (1 - recovery_yield)     → units not recovered as components

        formally_collected = ewaste_at_eol * collection_rate
        recycled = formally_collected * recycling_efficiency
        ewaste_residual = ewaste_at_eol - recycled

    CO2e avoided compared to zero circular action:
        Each unit repaired or reused avoids one unit of new production.
        CO2e avoided = (reuse_rate + repair_rate) * production_volume * CO2e_per_unit
    """
    vol = max(0.0, cfg.production_volume)
    r_repair = max(0.0, min(1.0, cfg.repair_rate))
    r_reuse = max(0.0, min(1.0, cfg.reuse_rate))
    r_collect = max(0.0, min(1.0, cfg.collection_rate))
    r_recycle = max(0.0, min(1.0, cfg.recycling_efficiency))
    r_recover = max(0.0, min(1.0, cfg.recovery_yield))

    # Ensure circular rates don't exceed 1
    total_circular = r_repair + r_reuse + r_recover
    if total_circular > 1.0:
        logger.warning(
            "Scenario '%s': repair+reuse+recovery > 100%% (%.0f%%). Capping at 100%%.",
            cfg.name, total_circular * 100,
        )
        scale = 1.0 / total_circular
        r_repair *= scale
        r_reuse *= scale
        r_recover *= scale

    # Units that become e-waste at end of life
    units_repaired = vol * r_repair
    units_reused = vol * r_reuse
    units_recovered = vol * r_recover
    units_to_eol = vol - units_repaired - units_reused

    # Of EOL units — formal collection and recycling
    units_collected = units_to_eol * r_collect
    units_recycled = units_collected * r_recycle
    units_residual = units_to_eol - units_recycled  # uncollected + uncycled

    # Physical weight
    weight_kg = WEIGHT_KG_PER_UNIT.get(cfg.product_category, WEIGHT_KG_PER_UNIT["Default"])
    ewaste_kg = units_residual * weight_kg

    # CO2e avoided (illustrative)
    co2e_per_unit = CO2E_PER_UNIT_KG.get(cfg.product_category, CO2E_PER_UNIT_KG["Default"])
    co2e_avoided = (units_repaired + units_reused + units_recovered) * co2e_per_unit

    # New production needed (after circular supply covers some demand)
    effective_production = max(0, vol - units_repaired - units_reused - units_recovered)

    caveats = [
        "All quantities are MODELLED ESTIMATES based on user-supplied parameters.",
        "CO2e values are illustrative order-of-magnitude estimates from LCA literature.",
        "Actual outcomes depend on local conditions, supply chains, and consumer behaviour.",
    ]

    return ScenarioResult(
        name=cfg.name,
        ewaste_generated_units=round(units_residual, 2),
        ewaste_generated_kg=round(ewaste_kg, 2),
        formally_collected_units=round(units_collected, 2),
        recycled_units=round(units_recycled, 2),
        recovered_units=round(units_recovered, 2),
        co2e_avoided_kg=round(co2e_avoided, 2),
        production_volume=vol,
        effective_new_production_needed=round(effective_production, 2),
        assumptions=[
            f"lifespan_years={cfg.lifespan_years}",
            f"repair_rate={r_repair:.0%}",
            f"reuse_rate={r_reuse:.0%}",
            f"collection_rate={r_collect:.0%}",
            f"recycling_efficiency={r_recycle:.0%}",
            f"recovery_yield={r_recover:.0%}",
        ],
        caveats=caveats,
    )


def compare_scenarios(configs: List[ScenarioConfig]) -> List[ScenarioResult]:
    """Simulate and return results for a list of scenario configurations."""
    results = [simulate_scenario(cfg) for cfg in configs]
    baseline = results[0] if results else None

    for i, r in enumerate(results):
        if baseline and i > 0:
            delta = baseline.ewaste_generated_kg - r.ewaste_generated_kg
            pct = 100.0 * delta / max(baseline.ewaste_generated_kg, 1e-9)
            logger.info(
                "Scenario '%s': e-waste=%.0f kg (delta vs baseline: %+.0f kg, %+.1f%%)",
                r.name, r.ewaste_generated_kg, -delta, pct,
            )
        else:
            logger.info("Baseline '%s': e-waste=%.0f kg", r.name, r.ewaste_generated_kg)

    return results


def default_scenario_set(product_category: str = "Laptop", volume: float = 10000) -> List[ScenarioConfig]:
    """Return a pre-built set of standard comparison scenarios."""
    return [
        ScenarioConfig(
            name="Current Baseline",
            production_volume=volume,
            product_category=product_category,
            lifespan_years=4.0,
            repair_rate=0.05,
            reuse_rate=0.05,
            collection_rate=0.25,
            recycling_efficiency=0.70,
            recovery_yield=0.0,
            description="Current average scenario (low circular economy activity)",
        ),
        ScenarioConfig(
            name="Improved Repair & Refurb",
            production_volume=volume,
            product_category=product_category,
            lifespan_years=5.5,
            repair_rate=0.20,
            reuse_rate=0.10,
            collection_rate=0.35,
            recycling_efficiency=0.75,
            recovery_yield=0.05,
            description="Increased repair rate and moderate reuse",
        ),
        ScenarioConfig(
            name="Extended Lifespan + High Collection",
            production_volume=volume,
            product_category=product_category,
            lifespan_years=7.0,
            repair_rate=0.15,
            reuse_rate=0.20,
            collection_rate=0.60,
            recycling_efficiency=0.80,
            recovery_yield=0.10,
            description="Extended product lifespan and high formal collection",
        ),
        ScenarioConfig(
            name="Circular Economy Target",
            production_volume=volume,
            product_category=product_category,
            lifespan_years=8.0,
            repair_rate=0.25,
            reuse_rate=0.30,
            collection_rate=0.75,
            recycling_efficiency=0.90,
            recovery_yield=0.20,
            description="Ambitious circular economy targets across all levers",
        ),
    ]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")

    scenarios = default_scenario_set(product_category="Laptop", volume=10000)
    results = compare_scenarios(scenarios)

    print("\nEcoCircuit AI — What-If Scenario Comparison")
    print("=" * 75)
    print(f"{'Scenario':<30} {'E-Waste (kg)':>14} {'CO2e Avoided (kg)':>18} {'Collected':>12}")
    print("-" * 75)
    for r in results:
        print(
            f"{r.name:<30} {r.ewaste_generated_kg:>14,.0f} "
            f"{r.co2e_avoided_kg:>18,.0f} {r.formally_collected_units:>12,.0f}"
        )
    print("-" * 75)
    print("\n  ⚠ All values are modelled estimates — not measured environmental outcomes.")
