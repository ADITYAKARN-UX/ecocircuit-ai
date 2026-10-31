"""
EcoCircuit AI — Production Optimizer Module (Module D)
======================================================
Calculates recommended new production quantity using a
transparent inventory-aware formula.

Core equation:
  Recommended New Production = MAX(0,
      Forecast_Demand
      + Required_Safety_Stock
      - Usable_Existing_Inventory
      - Expected_Usable_Returns
      - Expected_Recovered_Usable_Components
  )

All double-counting is prevented by design.
All inputs and assumptions are documented in the output.

Author: EcoCircuit AI Team
"""

import logging
from typing import Dict, List, Optional
import numpy as np

logger = logging.getLogger("production_optimizer")


def calculate_production(
    forecast_demand: float,
    usable_inventory: float,
    safety_stock: float = 0.0,
    repairable_inventory: float = 0.0,
    repair_success_rate: float = 0.0,
    expected_usable_returns: float = 0.0,
    recoverable_components: float = 0.0,
    component_recovery_yield: float = 0.0,
    overlap_returns_in_usable: bool = False,
    overlap_components_in_usable: bool = False,
) -> Dict:
    """
    Calculate recommended new production quantity.

    Parameters
    ----------
    forecast_demand : float
        Expected demand for the planning period.
    usable_inventory : float
        Current inventory already verified as usable (ready to deploy).
        Must NOT include items already counted in returns or recovered components.
    safety_stock : float
        Additional buffer stock required.
    repairable_inventory : float
        Number of units that COULD be repaired.
        These must NOT already be counted in usable_inventory.
    repair_success_rate : float
        Fraction (0–1) of repairable items expected to be successfully repaired.
    expected_usable_returns : float
        Units expected to be returned and directly reusable.
        Must NOT already be counted in usable_inventory.
        If overlap_returns_in_usable=True, this value is set to 0 to prevent double-counting.
    recoverable_components : float
        Number of component sets recoverable from end-of-life products.
        Must NOT already be counted in usable_inventory or returns.
        If overlap_components_in_usable=True, this value is set to 0.
    component_recovery_yield : float
        Fraction (0–1) of recoverable components that produce usable units.
    overlap_returns_in_usable : bool
        Set True if expected_usable_returns are already included in usable_inventory.
        Prevents double-counting.
    overlap_components_in_usable : bool
        Set True if recovered components are already included in usable_inventory.
        Prevents double-counting.

    Returns
    -------
    dict
        Full calculation breakdown, recommended production, and assumptions.
    """

    # Input validation
    def _validate_non_negative(val: float, name: str) -> float:
        if val < 0:
            logger.warning("%s cannot be negative. Clipping to 0.", name)
            return 0.0
        return float(val)

    def _validate_rate(val: float, name: str) -> float:
        if not 0 <= val <= 1:
            logger.warning("%s should be between 0 and 1. Clipping.", name)
            return float(np.clip(val, 0, 1))
        return float(val)

    forecast_demand = _validate_non_negative(forecast_demand, "forecast_demand")
    usable_inventory = _validate_non_negative(usable_inventory, "usable_inventory")
    safety_stock = _validate_non_negative(safety_stock, "safety_stock")
    repairable_inventory = _validate_non_negative(repairable_inventory, "repairable_inventory")
    repair_success_rate = _validate_rate(repair_success_rate, "repair_success_rate")
    expected_usable_returns = _validate_non_negative(expected_usable_returns, "expected_usable_returns")
    recoverable_components = _validate_non_negative(recoverable_components, "recoverable_components")
    component_recovery_yield = _validate_rate(component_recovery_yield, "component_recovery_yield")

    # Overlap prevention
    if overlap_returns_in_usable:
        logger.info("expected_usable_returns already in usable_inventory — setting to 0 to prevent double-counting.")
        expected_usable_returns = 0.0

    if overlap_components_in_usable:
        logger.info("recoverable_components already in usable_inventory — setting to 0 to prevent double-counting.")
        recoverable_components = 0.0

    # Derived quantities
    expected_repaired = repairable_inventory * repair_success_rate
    expected_recovered = recoverable_components * component_recovery_yield

    # Core calculation
    effective_supply = (
        usable_inventory
        + expected_repaired
        + expected_usable_returns
        + expected_recovered
    )

    total_need = forecast_demand + safety_stock
    raw_production = total_need - effective_supply
    recommended_production = max(0.0, raw_production)
    unmet_demand = max(0.0, forecast_demand - effective_supply)

    # Circular economy waste avoided
    waste_avoided = expected_repaired + expected_recovered

    result = {
        "inputs": {
            "forecast_demand": forecast_demand,
            "usable_inventory": usable_inventory,
            "safety_stock": safety_stock,
            "repairable_inventory": repairable_inventory,
            "repair_success_rate": repair_success_rate,
            "expected_usable_returns": expected_usable_returns,
            "recoverable_components": recoverable_components,
            "component_recovery_yield": component_recovery_yield,
        },
        "derived": {
            "expected_repaired_units": round(expected_repaired, 2),
            "expected_recovered_units": round(expected_recovered, 2),
            "total_effective_supply": round(effective_supply, 2),
            "total_demand_with_safety_stock": round(total_need, 2),
            "new_production_avoidable_units": round(waste_avoided, 2),
        },
        "outputs": {
            "recommended_new_production": round(recommended_production, 2),
            "unmet_demand": round(unmet_demand, 2),
            "supply_vs_demand_gap": round(effective_supply - total_need, 2),
        },
        "double_counting_checks": {
            "returns_overlap_resolved": overlap_returns_in_usable,
            "components_overlap_resolved": overlap_components_in_usable,
        },
        "interpretation": _interpret_result(recommended_production, unmet_demand, waste_avoided),
    }

    logger.info(
        "Production calc: demand=%.1f + safety=%.1f | supply=%.1f | recommended=%.1f | unmet=%.1f",
        forecast_demand, safety_stock, effective_supply,
        recommended_production, unmet_demand,
    )
    return result


def _interpret_result(production: float, unmet: float, waste_avoided: float) -> str:
    """Generate a plain-language interpretation."""
    if production == 0 and unmet == 0:
        return (
            "Existing inventory and circular economy sources fully cover demand. "
            "No new production is recommended at this time."
        )
    elif production == 0 and unmet > 0:
        return (
            f"Supply from existing stock and circular sources cannot fully meet demand "
            f"(shortfall: {unmet:.1f} units). Consider increasing collection, repair, "
            f"or recovery rates before committing to new production."
        )
    else:
        msg = f"Recommended new production: {production:.1f} units."
        if waste_avoided > 0:
            msg += (
                f" Circular economy actions (repair + recovery) contribute {waste_avoided:.1f} units, "
                f"reducing the production requirement."
            )
        if unmet > 0:
            msg += f" Unmet demand remains: {unmet:.1f} units."
        return msg


# ── Scenario comparison ─────────────────────────────────────────────────────

def compare_scenarios(scenarios: List[Dict]) -> List[Dict]:
    """
    Run production calculation for multiple named scenarios and return comparison.

    Each scenario dict should contain a 'name' key plus all production inputs.
    """
    results = []
    for scenario in scenarios:
        name = scenario.pop("name", f"Scenario {len(results)+1}")
        result = calculate_production(**scenario)
        result["scenario_name"] = name
        results.append(result)
        logger.info(
            "Scenario '%s': recommended_production=%.1f",
            name, result["outputs"]["recommended_new_production"],
        )
    return results


def compute_what_if_sensitivity(
    base_inputs: Dict,
    parameter: str,
    values: List[float],
) -> List[Dict]:
    """
    Compute production recommendations across a range of values for one parameter.
    Useful for sensitivity / what-if analysis.
    """
    rows = []
    for val in values:
        inputs = base_inputs.copy()
        inputs[parameter] = val
        result = calculate_production(**inputs)
        rows.append({
            parameter: val,
            "recommended_production": result["outputs"]["recommended_new_production"],
            "unmet_demand": result["outputs"]["unmet_demand"],
            "effective_supply": result["derived"]["total_effective_supply"],
            "waste_avoided": result["derived"]["new_production_avoidable_units"],
        })
    return rows


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")

    print("=" * 55)
    print("EcoCircuit AI — Production Optimizer Test")
    print("=" * 55)

    result = calculate_production(
        forecast_demand=1000,
        usable_inventory=300,
        safety_stock=100,
        repairable_inventory=200,
        repair_success_rate=0.75,
        expected_usable_returns=50,
        recoverable_components=150,
        component_recovery_yield=0.6,
    )

    print(f"\nInputs:")
    for k, v in result["inputs"].items():
        print(f"  {k}: {v}")

    print(f"\nDerived:")
    for k, v in result["derived"].items():
        print(f"  {k}: {v}")

    print(f"\nOutputs:")
    for k, v in result["outputs"].items():
        print(f"  {k}: {v}")

    print(f"\nInterpretation:\n  {result['interpretation']}")

    # Sensitivity analysis
    print("\n--- Sensitivity: repair_success_rate ---")
    sens = compute_what_if_sensitivity(
        base_inputs={
            "forecast_demand": 1000, "usable_inventory": 300,
            "safety_stock": 100, "repairable_inventory": 200,
            "repair_success_rate": 0.0,
            "expected_usable_returns": 50,
            "recoverable_components": 150, "component_recovery_yield": 0.6,
        },
        parameter="repair_success_rate",
        values=[0.0, 0.25, 0.5, 0.75, 1.0],
    )
    for row in sens:
        print(f"  repair_rate={row['repair_success_rate']:.2f} → production={row['recommended_production']:.0f}")
