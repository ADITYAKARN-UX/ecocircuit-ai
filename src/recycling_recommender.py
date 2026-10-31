"""
EcoCircuit AI — Recycling and Repair Recommender (Module E)
============================================================
Transparent rule-based recommendation engine for product end-of-life decisions.

Possible recommendations:
  1. Continue using
  2. Repair
  3. Refurbish
  4. Reuse (resell / donate)
  5. Recover usable components
  6. Send to authorized recycler
  7. Specialist disposal (hazardous)

All decision logic is documented and inspectable.
This is a rule-based system — NOT an ML model.
Results are explicitly labelled as rule-based recommendations.

Author: EcoCircuit AI Team
"""

import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

logger = logging.getLogger("recycling_recommender")


@dataclass
class ProductInput:
    """Input parameters for an end-of-life recommendation."""
    product_category: str
    condition: str              # "excellent", "good", "fair", "poor", "non-functional"
    age_years: float
    expected_lifespan_years: float
    repair_cost_estimate: float = 0.0
    replacement_cost: float = 0.0
    estimated_remaining_life_years: float = 0.0
    repair_success_probability: float = 0.5
    recovery_value_estimate: float = 0.0
    recycling_available: bool = True
    documented_safety_risk: bool = False
    battery_health_pct: Optional[float] = None  # 0-100
    product_id: Optional[str] = None


@dataclass
class Recommendation:
    """Output recommendation with full reasoning."""
    action: str
    reasoning: str
    assumptions: List[str] = field(default_factory=list)
    estimated_waste_avoided_kg: Optional[float] = None
    estimated_cost_saving: Optional[float] = None
    caveats: List[str] = field(default_factory=list)
    confidence: str = "medium"   # "high", "medium", "low"
    recommendation_type: str = "rule_based"  # Always rule_based — never claim ML

    def to_dict(self) -> Dict:
        return asdict(self)


# ── Average product weights (kg) for waste-avoided estimates ───────────────
PRODUCT_WEIGHTS_KG = {
    "Smartphone": 0.18,
    "Tablet": 0.45,
    "Laptop": 2.0,
    "Desktop Computer": 8.0,
    "Monitor": 4.5,
    "Smartwatch": 0.05,
    "Television": 12.0,
    "Refrigerator": 65.0,
    "Washing Machine": 60.0,
    "Air Conditioner": 30.0,
    "Microwave": 12.0,
    "Printer": 8.0,
    "Scanner": 4.0,
    "Router": 0.5,
    "Charger": 0.1,
    "Keyboard": 0.3,
    "Mouse": 0.1,
    "Gaming Console": 3.5,
    "Headphones": 0.25,
    "Server": 20.0,
    "Default": 2.0,
}

# Products with documented battery safety concerns (thermal runaway risk)
BATTERY_SAFETY_CATEGORIES = {
    "Smartphone", "Tablet", "Laptop", "Smartwatch", "E-reader",
    "Gaming Console", "Rechargeable Battery Pack",
}

# Products for which repair is generally NOT cost-effective
LOW_REPAIRABILITY_CATEGORIES = {
    "Charger", "Cable", "Earphones", "Mouse",
}

# Hazardous disposal required categories
SPECIALIST_DISPOSAL_CATEGORIES = {
    "Refrigerator", "Air Conditioner",  # refrigerants
    # Fluorescent Lamps, CRT monitors (if present)
}


# ── Core recommendation engine ─────────────────────────────────────────────

def recommend(inp: ProductInput) -> Recommendation:
    """
    Compute an end-of-life recommendation for an electronic product.
    Returns a Recommendation with full decision reasoning.
    """

    assumptions = [
        "Recommendation is rule-based using documented decision criteria.",
        "No personal device data, telemetry, or manufacturer records are accessed.",
        f"Expected lifespan: {inp.expected_lifespan_years} years for {inp.product_category}.",
    ]
    caveats = []

    # Step 1: Safety risk check — always first
    if inp.documented_safety_risk:
        return Recommendation(
            action="Specialist Disposal",
            reasoning=(
                "A documented safety risk has been identified for this product. "
                "Repair, reuse, or donation is not appropriate. "
                "Follow the manufacturer's or regulatory authority's guidance for safe disposal."
            ),
            assumptions=assumptions,
            caveats=[
                "Do NOT repair or resell a product with a documented safety defect.",
                "Contact the manufacturer or your local e-waste authority for disposal guidance.",
            ],
            confidence="high",
        )

    # Step 2: Battery health check for battery-critical products
    if (
        inp.product_category in BATTERY_SAFETY_CATEGORIES
        and inp.battery_health_pct is not None
        and inp.battery_health_pct < 40
    ):
        caveats.append(
            f"Battery health is critically low ({inp.battery_health_pct:.0f}%). "
            "A degraded lithium battery poses a thermal risk. "
            "Battery replacement or recycling is strongly recommended before any reuse."
        )

    # Step 3: Condition-based early decisions
    condition_lower = inp.condition.lower()
    if condition_lower in ("excellent", "good") and inp.age_years < inp.expected_lifespan_years * 0.7:
        weight_kg = PRODUCT_WEIGHTS_KG.get(inp.product_category, PRODUCT_WEIGHTS_KG["Default"])
        return Recommendation(
            action="Continue Using",
            reasoning=(
                f"Product is in {inp.condition} condition and is within {inp.age_years:.1f} of "
                f"its expected {inp.expected_lifespan_years}-year lifespan. "
                "Continued use avoids premature disposal and reduces e-waste generation."
            ),
            assumptions=assumptions,
            estimated_waste_avoided_kg=round(weight_kg, 2),
            confidence="high",
            caveats=caveats,
        )

    # Step 4: Non-functional or very poor condition
    if condition_lower in ("non-functional", "poor"):
        return _handle_non_functional(inp, assumptions, caveats)

    # Step 5: Fair condition — evaluate repair vs. replace
    return _handle_fair_condition(inp, assumptions, caveats)


def _handle_non_functional(
    inp: ProductInput,
    assumptions: List[str],
    caveats: List[str],
) -> Recommendation:
    """Logic for non-functional or poor-condition products."""
    weight_kg = PRODUCT_WEIGHTS_KG.get(inp.product_category, PRODUCT_WEIGHTS_KG["Default"])

    # Cannot repair low-repairability items
    if inp.product_category in LOW_REPAIRABILITY_CATEGORIES:
        return Recommendation(
            action="Authorized Recycler",
            reasoning=(
                f"{inp.product_category} items are generally not economically repairable. "
                "Send to an authorized e-waste recycler for material recovery."
            ),
            assumptions=assumptions,
            estimated_waste_avoided_kg=None,
            caveats=caveats + [
                "Ensure the recycler is certified (e.g., R2, e-Stewards, or equivalent national standard)."
            ],
            confidence="high",
        )

    # Specialist disposal categories
    if inp.product_category in SPECIALIST_DISPOSAL_CATEGORIES:
        return Recommendation(
            action="Specialist Disposal",
            reasoning=(
                f"{inp.product_category} contains refrigerants or hazardous materials "
                "requiring specialist handling. "
                "Standard e-waste recyclers may not be equipped; contact a specialist."
            ),
            assumptions=assumptions,
            caveats=caveats + [
                "Refrigerants must be recovered by a certified technician.",
                "Improper disposal of refrigerants contributes to ozone depletion and climate change.",
            ],
            confidence="high",
        )

    # Repair viability check
    if (
        inp.repair_cost_estimate > 0
        and inp.replacement_cost > 0
        and inp.repair_cost_estimate < 0.6 * inp.replacement_cost
        and inp.repair_success_probability >= 0.5
        and inp.estimated_remaining_life_years >= 1.5
    ):
        return Recommendation(
            action="Repair",
            reasoning=(
                f"Repair cost (estimated {inp.repair_cost_estimate:.0f}) is less than 60% of "
                f"replacement cost ({inp.replacement_cost:.0f}), "
                f"with a {inp.repair_success_probability*100:.0f}% success probability and "
                f"~{inp.estimated_remaining_life_years:.1f} years of remaining useful life expected. "
                "Repair is the recommended action to reduce waste."
            ),
            assumptions=assumptions + [
                "Repair cost estimate is indicative only.",
                "Actual success rate may vary by technician and fault type.",
            ],
            estimated_waste_avoided_kg=round(weight_kg, 2),
            estimated_cost_saving=round(inp.replacement_cost - inp.repair_cost_estimate, 2),
            caveats=caveats,
            confidence="medium",
        )

    # Component recovery
    if inp.recovery_value_estimate > 0 or inp.age_years < inp.expected_lifespan_years * 1.5:
        return Recommendation(
            action="Recover Usable Components",
            reasoning=(
                "Full product repair is not clearly viable, but usable components "
                "may be recovered for reuse in other devices. "
                "Component recovery reduces the need for newly manufactured parts."
            ),
            assumptions=assumptions + [
                "Component recovery value depends on demand for specific parts.",
                "A qualified repair technician should assess component condition.",
            ],
            estimated_waste_avoided_kg=round(weight_kg * 0.3, 2),
            caveats=caveats,
            confidence="low",
        )

    # Default: authorized recycler
    return Recommendation(
        action="Authorized Recycler",
        reasoning=(
            "Product is non-functional and repair or recovery is not clearly cost-effective. "
            "Responsible recycling through a certified e-waste recycler is the recommended path."
        ),
        assumptions=assumptions,
        caveats=caveats + [
            "Verify recycler certification (R2, e-Stewards, or national equivalent).",
            "Data-bearing devices (phones, laptops, hard drives) should have data wiped before recycling.",
        ],
        confidence="medium",
    )


def _handle_fair_condition(
    inp: ProductInput,
    assumptions: List[str],
    caveats: List[str],
) -> Recommendation:
    """Logic for products in fair condition — near or past expected lifespan."""
    weight_kg = PRODUCT_WEIGHTS_KG.get(inp.product_category, PRODUCT_WEIGHTS_KG["Default"])
    age_ratio = inp.age_years / max(inp.expected_lifespan_years, 0.1)

    # Still within lifespan — assess refurbishment / reuse
    if age_ratio < 1.2:
        if (
            inp.repair_cost_estimate > 0
            and inp.replacement_cost > 0
            and inp.repair_cost_estimate < 0.4 * inp.replacement_cost
        ):
            return Recommendation(
                action="Refurbish",
                reasoning=(
                    f"Product is in fair condition (age ratio: {age_ratio:.1f}×). "
                    f"Refurbishment cost ({inp.repair_cost_estimate:.0f}) is less than 40% "
                    f"of replacement ({inp.replacement_cost:.0f}). "
                    "Refurbished devices can be reused or resold, extending product life."
                ),
                assumptions=assumptions,
                estimated_waste_avoided_kg=round(weight_kg, 2),
                estimated_cost_saving=round(inp.replacement_cost - inp.repair_cost_estimate, 2),
                caveats=caveats,
                confidence="medium",
            )

        return Recommendation(
            action="Reuse",
            reasoning=(
                f"Product is in fair condition and still within expected lifespan "
                f"(age {inp.age_years:.1f} yr vs {inp.expected_lifespan_years} yr expected). "
                "Donation or resale extends the product's useful life."
            ),
            assumptions=assumptions,
            estimated_waste_avoided_kg=round(weight_kg, 2),
            caveats=caveats + ["Ensure the product is safe and functional before donating or selling."],
            confidence="medium",
        )

    # Past expected lifespan
    return Recommendation(
        action="Authorized Recycler",
        reasoning=(
            f"Product has exceeded its expected lifespan "
            f"({inp.age_years:.1f} yr vs {inp.expected_lifespan_years} yr). "
            "Responsible recycling is the recommended end-of-life path."
        ),
        assumptions=assumptions,
        caveats=caveats,
        confidence="medium",
    )


# ── Batch recommendation ────────────────────────────────────────────────────

def batch_recommend(products: List[Dict]) -> List[Dict]:
    """Generate recommendations for a list of product input dicts."""
    results = []
    for prod_dict in products:
        try:
            inp = ProductInput(**prod_dict)
            rec = recommend(inp)
            results.append({
                "product_id": inp.product_id or "unknown",
                "product_category": inp.product_category,
                "recommendation": rec.to_dict(),
            })
        except Exception as exc:
            logger.error("Recommendation failed for %s: %s", prod_dict.get("product_id", "?"), exc)
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")

    test_cases = [
        ProductInput("Smartphone", "good", 1.5, 2.5, battery_health_pct=85, product_id="P001"),
        ProductInput("Laptop", "fair", 3.5, 4.0, 120, 800, 1.5, 0.7, product_id="P002"),
        ProductInput("Refrigerator", "non-functional", 16, 15, product_id="P003"),
        ProductInput("Laptop", "poor", 5, 4.0, 50, 600, 0.5, 0.4, product_id="P004"),
        ProductInput("Smartphone", "good", 1.0, 2.5, documented_safety_risk=True, product_id="P005"),
        ProductInput("Television", "fair", 9, 8, 80, 500, 1.5, 0.6, product_id="P006"),
    ]

    print("EcoCircuit AI — Recycling Recommender Test")
    print("=" * 60)
    for inp in test_cases:
        rec = recommend(inp)
        print(f"\n[{inp.product_id}] {inp.product_category} | Condition: {inp.condition}")
        print(f"  → Action: {rec.action} ({rec.confidence} confidence)")
        print(f"  → Reasoning: {rec.reasoning[:100]}...")
        if rec.caveats:
            print(f"  → Caveats: {rec.caveats[0]}")
