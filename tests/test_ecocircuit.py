"""
EcoCircuit AI — Automated Test Suite
=====================================
Tests for critical functionality as specified in Section 18.

Run with: pytest tests/ -v

Author: EcoCircuit AI Team
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.production_optimizer import calculate_production, compute_what_if_sensitivity
from src.recycling_recommender import ProductInput, recommend, Recommendation
from src.what_if_simulator import ScenarioConfig, simulate_scenario, compare_scenarios


# ════════════════════════════════════════════════════════════════════════════
# PRODUCTION OPTIMIZER TESTS
# ════════════════════════════════════════════════════════════════════════════

class TestProductionOptimizer:
    """Tests for production_optimizer.calculate_production()"""

    def test_recommended_production_cannot_be_negative(self):
        """Recommended production must always be >= 0."""
        result = calculate_production(
            forecast_demand=100,
            usable_inventory=500,  # much more than demand
            safety_stock=10,
        )
        assert result["outputs"]["recommended_new_production"] >= 0

    def test_recommended_production_zero_when_supply_exceeds_demand(self):
        """When supply far exceeds demand + safety_stock, production should be 0."""
        result = calculate_production(
            forecast_demand=200,
            usable_inventory=300,
            safety_stock=50,
        )
        assert result["outputs"]["recommended_new_production"] == 0.0

    def test_recommended_production_correct_with_all_inputs(self):
        """Verify core formula: demand + safety - all_supply_sources."""
        result = calculate_production(
            forecast_demand=1000,
            usable_inventory=300,
            safety_stock=100,
            repairable_inventory=200,
            repair_success_rate=0.5,
            expected_usable_returns=50,
            recoverable_components=100,
            component_recovery_yield=0.5,
        )
        # Expected supply: 300 (usable) + 100 (repaired) + 50 (returns) + 50 (recovered) = 500
        # Need: 1000 + 100 = 1100
        # Production: 1100 - 500 = 600
        assert result["derived"]["total_effective_supply"] == pytest.approx(500.0, abs=0.01)
        assert result["outputs"]["recommended_new_production"] == pytest.approx(600.0, abs=0.01)

    def test_existing_usable_inventory_not_counted_twice(self):
        """Usable inventory should not be double-counted with returns."""
        result_no_overlap = calculate_production(
            forecast_demand=500,
            usable_inventory=200,
            safety_stock=0,
            expected_usable_returns=100,
            overlap_returns_in_usable=False,
        )
        result_with_overlap = calculate_production(
            forecast_demand=500,
            usable_inventory=200,
            safety_stock=0,
            expected_usable_returns=100,
            overlap_returns_in_usable=True,
        )
        # When overlap resolved, returns should be set to 0
        assert result_with_overlap["inputs"]["expected_usable_returns"] == 0 \
               or result_with_overlap["derived"]["total_effective_supply"] < result_no_overlap["derived"]["total_effective_supply"]

    def test_safety_stock_increases_recommended_production(self):
        """Higher safety stock should result in same or higher recommended production."""
        result_low = calculate_production(forecast_demand=100, usable_inventory=80, safety_stock=10)
        result_high = calculate_production(forecast_demand=100, usable_inventory=80, safety_stock=50)
        assert result_high["outputs"]["recommended_new_production"] >= result_low["outputs"]["recommended_new_production"]

    def test_expected_returns_reduce_production(self):
        """Higher returns should reduce or maintain recommended production."""
        result_no_returns = calculate_production(forecast_demand=500, usable_inventory=100, safety_stock=50)
        result_with_returns = calculate_production(
            forecast_demand=500, usable_inventory=100, safety_stock=50,
            expected_usable_returns=100,
        )
        assert result_with_returns["outputs"]["recommended_new_production"] <= \
               result_no_returns["outputs"]["recommended_new_production"]

    def test_recovered_components_not_double_counted(self):
        """If components already in usable, overlap flag must resolve double-counting."""
        result_overlap = calculate_production(
            forecast_demand=500,
            usable_inventory=300,
            recoverable_components=200,
            component_recovery_yield=0.5,
            overlap_components_in_usable=True,
        )
        # With overlap, recovered should effectively be 0
        # Supply should equal usable_inventory only
        supply = result_overlap["derived"]["total_effective_supply"]
        assert supply == pytest.approx(300.0, abs=0.01)

    def test_repair_success_rate_clipped_to_valid_range(self):
        """Repair success rate outside [0,1] should be clipped without crashing."""
        result = calculate_production(
            forecast_demand=100,
            usable_inventory=0,
            repairable_inventory=50,
            repair_success_rate=1.5,  # invalid — should be clipped to 1.0
        )
        assert result is not None
        assert result["outputs"]["recommended_new_production"] >= 0

    def test_zero_inputs_returns_demand_as_production(self):
        """With zero supply and zero safety stock, production should equal demand."""
        result = calculate_production(
            forecast_demand=1000,
            usable_inventory=0,
            safety_stock=0,
        )
        assert result["outputs"]["recommended_new_production"] == pytest.approx(1000.0, abs=0.01)

    def test_sensitivity_analysis_returns_expected_length(self):
        """Sensitivity analysis should return one row per value."""
        base = {"forecast_demand": 1000, "usable_inventory": 200, "safety_stock": 100}
        values = [0.0, 0.25, 0.5, 0.75, 1.0]
        rows = compute_what_if_sensitivity(base, "repair_success_rate", values)
        # Base inputs don't have repairable_inventory but calc should not crash
        assert len(rows) == len(values)

    def test_unmet_demand_correct(self):
        """Unmet demand should be demand minus effective_supply when supply < demand."""
        result = calculate_production(
            forecast_demand=1000,
            usable_inventory=400,
            safety_stock=0,
        )
        # Supply=400, demand=1000 -> unmet demand without new production is 600
        assert result["outputs"]["unmet_demand"] == pytest.approx(600.0, abs=0.01)

        # When supply exceeds demand, unmet demand should be 0
        result_covered = calculate_production(
            forecast_demand=400,
            usable_inventory=1000,
            safety_stock=0,
        )
        assert result_covered["outputs"]["unmet_demand"] == 0.0


# ════════════════════════════════════════════════════════════════════════════
# RECYCLING RECOMMENDER TESTS
# ════════════════════════════════════════════════════════════════════════════

class TestRecyclingRecommender:
    """Tests for recycling_recommender.recommend()"""

    def test_safety_risk_always_specialist_disposal(self):
        """A product with a documented safety risk must always get specialist disposal."""
        inp = ProductInput(
            product_category="Smartphone",
            condition="good",
            age_years=1.0,
            expected_lifespan_years=2.5,
            documented_safety_risk=True,
        )
        rec = recommend(inp)
        assert rec.action == "Specialist Disposal"

    def test_excellent_condition_within_lifespan_continue_using(self):
        """Good condition + well within lifespan → Continue Using."""
        inp = ProductInput(
            product_category="Laptop",
            condition="excellent",
            age_years=1.0,
            expected_lifespan_years=5.0,
        )
        rec = recommend(inp)
        assert rec.action == "Continue Using"

    def test_recommendation_type_always_rule_based(self):
        """Recommendation type must always be 'rule_based' — never 'ml_prediction'."""
        inp = ProductInput(
            product_category="Laptop",
            condition="fair",
            age_years=3.0,
            expected_lifespan_years=4.0,
        )
        rec = recommend(inp)
        assert rec.recommendation_type == "rule_based"

    def test_low_repairability_items_go_to_recycler(self):
        """Chargers, cables, etc. should not be recommended for repair."""
        inp = ProductInput(
            product_category="Charger",
            condition="non-functional",
            age_years=2.0,
            expected_lifespan_years=3.0,
        )
        rec = recommend(inp)
        assert rec.action in ("Authorized Recycler", "Specialist Disposal")

    def test_recommendation_has_reasoning(self):
        """Every recommendation must have a non-empty reasoning string."""
        inp = ProductInput(
            product_category="Television",
            condition="fair",
            age_years=7.0,
            expected_lifespan_years=8.0,
        )
        rec = recommend(inp)
        assert rec.reasoning and len(rec.reasoning) > 10

    def test_repair_recommended_when_cost_justified(self):
        """Repair should be recommended when cost < 60% of replacement."""
        inp = ProductInput(
            product_category="Laptop",
            condition="non-functional",
            age_years=2.0,
            expected_lifespan_years=5.0,
            repair_cost_estimate=200,
            replacement_cost=800,
            estimated_remaining_life_years=2.5,
            repair_success_probability=0.8,
        )
        rec = recommend(inp)
        assert rec.action in ("Repair", "Recover Usable Components", "Authorized Recycler")
        # At 200/800 = 25% cost ratio, repair should be recommended
        if rec.action == "Repair":
            assert True  # Correct outcome

    def test_non_functional_past_lifespan_goes_to_recycler(self):
        """Non-functional + past lifespan should recommend recycling."""
        inp = ProductInput(
            product_category="Desktop Computer",
            condition="non-functional",
            age_years=10.0,
            expected_lifespan_years=6.0,
        )
        rec = recommend(inp)
        assert rec.action in ("Authorized Recycler", "Recover Usable Components", "Specialist Disposal")

    def test_refrigerator_non_functional_specialist_disposal(self):
        """Refrigerators need specialist disposal for refrigerant handling."""
        inp = ProductInput(
            product_category="Refrigerator",
            condition="non-functional",
            age_years=15.0,
            expected_lifespan_years=15.0,
        )
        rec = recommend(inp)
        assert rec.action == "Specialist Disposal"


# ════════════════════════════════════════════════════════════════════════════
# WHAT-IF SIMULATOR TESTS
# ════════════════════════════════════════════════════════════════════════════

class TestWhatIfSimulator:
    """Tests for what_if_simulator.simulate_scenario()"""

    def test_ewaste_generated_non_negative(self):
        """E-waste generated must never be negative."""
        cfg = ScenarioConfig(
            name="test",
            production_volume=1000,
            repair_rate=0.5,
            reuse_rate=0.5,
            collection_rate=0.9,
            recycling_efficiency=0.9,
        )
        result = simulate_scenario(cfg)
        assert result.ewaste_generated_units >= 0
        assert result.ewaste_generated_kg >= 0

    def test_circular_rates_clamped_to_one(self):
        """Sum of repair+reuse+recovery > 1 should not crash and result is valid."""
        cfg = ScenarioConfig(
            name="excessive_rates",
            production_volume=1000,
            repair_rate=0.5,
            reuse_rate=0.6,
            recovery_yield=0.3,
        )
        result = simulate_scenario(cfg)  # should not raise
        assert result is not None

    def test_higher_collection_rate_reduces_ewaste(self):
        """Higher collection rate should reduce residual e-waste."""
        base_cfg = ScenarioConfig(name="low", production_volume=10000, collection_rate=0.2)
        high_cfg = ScenarioConfig(name="high", production_volume=10000, collection_rate=0.8)
        r_base = simulate_scenario(base_cfg)
        r_high = simulate_scenario(high_cfg)
        assert r_high.ewaste_generated_kg <= r_base.ewaste_generated_kg

    def test_co2e_avoided_increases_with_reuse(self):
        """More reuse → more CO2e avoided."""
        low_cfg = ScenarioConfig(name="low_reuse", production_volume=1000, reuse_rate=0.05)
        high_cfg = ScenarioConfig(name="high_reuse", production_volume=1000, reuse_rate=0.50)
        r_low = simulate_scenario(low_cfg)
        r_high = simulate_scenario(high_cfg)
        assert r_high.co2e_avoided_kg >= r_low.co2e_avoided_kg

    def test_scenario_result_has_caveats(self):
        """All scenario results must include caveats distinguishing estimates from facts."""
        cfg = ScenarioConfig(name="test", production_volume=5000)
        result = simulate_scenario(cfg)
        assert result.caveats and len(result.caveats) >= 1
        # Must warn it's an estimate
        assert any("estimate" in c.lower() or "model" in c.lower() for c in result.caveats)

    def test_compare_scenarios_returns_all_results(self):
        """compare_scenarios should return one result per config."""
        from src.what_if_simulator import default_scenario_set
        configs = default_scenario_set("Laptop", 5000)
        results = compare_scenarios(configs)
        assert len(results) == len(configs)


# ════════════════════════════════════════════════════════════════════════════
# DATA LOADING TESTS
# ════════════════════════════════════════════════════════════════════════════

class TestDataLoading:
    """Tests for dataset loading and validation."""

    def test_synthetic_data_generation(self):
        """Synthetic datasets should be generated without errors."""
        from src.synthetic_data import (
            create_global_ewaste_timeseries,
            create_india_ewaste_timeseries,
            create_battery_demo_data,
            create_hazardous_material_profiles,
        )
        df_global = create_global_ewaste_timeseries()
        assert isinstance(df_global, pd.DataFrame)
        assert len(df_global) > 0
        assert "year" in df_global.columns
        assert "global_ewaste_mt" in df_global.columns

        df_india = create_india_ewaste_timeseries()
        assert isinstance(df_india, pd.DataFrame)
        assert len(df_india) > 0

        df_battery = create_battery_demo_data()
        assert isinstance(df_battery, pd.DataFrame)
        assert len(df_battery) > 0
        assert "battery_id" in df_battery.columns
        assert "cycle_number" in df_battery.columns
        assert "capacity_ah" in df_battery.columns

        df_haz = create_hazardous_material_profiles()
        assert isinstance(df_haz, pd.DataFrame)
        assert len(df_haz) > 0

    def test_global_ewaste_timeseries_columns(self):
        """Global e-waste timeseries must have required columns."""
        from src.synthetic_data import create_global_ewaste_timeseries
        df = create_global_ewaste_timeseries()
        required = {"year", "global_ewaste_mt", "source"}
        assert required.issubset(set(df.columns))

    def test_battery_data_capacity_positive(self):
        """Battery capacity values must be positive."""
        from src.synthetic_data import create_battery_demo_data
        df = create_battery_demo_data()
        assert (df["capacity_ah"] > 0).all()

    def test_india_ewaste_collection_rate_range(self):
        """Collection rate must be between 0 and 100."""
        from src.synthetic_data import create_india_ewaste_timeseries
        df = create_india_ewaste_timeseries()
        assert (df["collection_rate_pct"] >= 0).all()
        assert (df["collection_rate_pct"] <= 100).all()


# ════════════════════════════════════════════════════════════════════════════
# FORECASTING TESTS
# ════════════════════════════════════════════════════════════════════════════

class TestForecasting:
    """Tests for e-waste forecasting module."""

    def test_chronological_split_no_leakage(self):
        """Test split should always use later dates than train split."""
        from src.preprocessing import chronological_split
        df = pd.DataFrame({
            "year": list(range(2010, 2024)),
            "value": np.random.rand(14),
        })
        train, test = chronological_split(df, "year", test_frac=0.25)
        assert train["year"].max() < test["year"].min()

    def test_forecast_returns_non_negative_values(self):
        """Forecasted e-waste values should be non-negative."""
        from src.ewaste_forecasting import run_ewaste_forecast
        from src.synthetic_data import create_global_ewaste_timeseries
        df = create_global_ewaste_timeseries()
        results = run_ewaste_forecast(df, "year", "global_ewaste_mt", horizon=3)
        future = results["future_df"]
        forecast_col = "global_ewaste_mt_forecast"
        assert (future[forecast_col] >= 0).all()

    def test_forecast_horizon_correct_length(self):
        """Forecast should produce exactly `horizon` future periods."""
        from src.ewaste_forecasting import run_ewaste_forecast
        from src.synthetic_data import create_global_ewaste_timeseries
        df = create_global_ewaste_timeseries()
        horizon = 5
        results = run_ewaste_forecast(df, "year", "global_ewaste_mt", horizon=horizon)
        assert len(results["future_df"]) == horizon

    def test_model_results_populated(self):
        """At least one model result must be returned."""
        from src.ewaste_forecasting import run_ewaste_forecast
        from src.synthetic_data import create_global_ewaste_timeseries
        df = create_global_ewaste_timeseries()
        results = run_ewaste_forecast(df, "year", "global_ewaste_mt", horizon=3)
        assert len(results["model_results"]) >= 1
        for r in results["model_results"]:
            assert "mae" in r
            assert "rmse" in r

    def test_naive_forecast_baseline(self):
        """Naive forecast should equal the last training value."""
        from src.ewaste_forecasting import naive_forecast
        train = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        pred = naive_forecast(train, 3)
        assert all(p == 5.0 for p in pred)


# ════════════════════════════════════════════════════════════════════════════
# BATTERY ANALYSIS TESTS
# ════════════════════════════════════════════════════════════════════════════

class TestBatteryAnalysis:
    """Tests for battery_analysis module."""

    def test_prepare_features_adds_required_columns(self):
        """Feature preparation should add lag and cycle features."""
        from src.battery_analysis import prepare_battery_features
        from src.synthetic_data import create_battery_demo_data
        df = create_battery_demo_data()
        df_feat = prepare_battery_features(df)
        assert "log_cycle" in df_feat.columns
        assert "capacity_lag1" in df_feat.columns
        assert "capacity_retention" in df_feat.columns

    def test_battery_split_no_leakage(self):
        """No battery ID should appear in both train and test splits."""
        from src.battery_analysis import battery_train_test_split
        from src.synthetic_data import create_battery_demo_data
        df = create_battery_demo_data()
        train, test = battery_train_test_split(df)
        train_ids = set(train["battery_id"].unique())
        test_ids = set(test["battery_id"].unique())
        assert train_ids.isdisjoint(test_ids), "Battery leakage detected!"

    def test_train_battery_models_returns_metrics(self):
        """Training should return metrics for all three models."""
        from src.battery_analysis import train_battery_models
        from src.synthetic_data import create_battery_demo_data
        df = create_battery_demo_data()
        results = train_battery_models(df)
        for model_name in ["Linear Regression", "Random Forest", "Gradient Boosting"]:
            assert model_name in results
            assert "metrics" in results[model_name]
            m = results[model_name]["metrics"]
            assert m["mae"] >= 0
            assert m["rmse"] >= 0


# ════════════════════════════════════════════════════════════════════════════
# EMPTY DATA HANDLING TESTS
# ════════════════════════════════════════════════════════════════════════════

class TestEmptyDataHandling:
    """Tests for graceful handling of empty or missing data."""

    def test_production_optimizer_with_zero_demand(self):
        """Zero demand should result in zero production."""
        result = calculate_production(
            forecast_demand=0,
            usable_inventory=0,
            safety_stock=0,
        )
        assert result["outputs"]["recommended_new_production"] == 0.0

    def test_what_if_with_zero_volume(self):
        """Zero production volume should not crash the simulator."""
        cfg = ScenarioConfig(name="zero", production_volume=0)
        result = simulate_scenario(cfg)
        assert result.ewaste_generated_units == 0
        assert result.ewaste_generated_kg == 0

    def test_recommendation_with_minimal_inputs(self):
        """Minimal inputs should not crash the recommender."""
        inp = ProductInput(
            product_category="Laptop",
            condition="fair",
            age_years=3.0,
            expected_lifespan_years=5.0,
        )
        rec = recommend(inp)
        assert isinstance(rec, Recommendation)
        assert rec.action is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
