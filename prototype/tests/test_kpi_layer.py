"""
KPI Intelligence Layer — Test Suite (15 tests).
Uses mocked Groq responses — no real API key required.
"""
import sys
import os
import json
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

# Ensure src/ is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.kpi.models import (
    EnhancedKPI, KPIAggregation, KPICondition, KPIDriver,
    KPIThreshold, ComputabilityResult, SchemaMetadata
)
from src.kpi.introspector import SchemaIntrospector
from src.kpi.computability import validate as kpi_validate
from src.kpi.executor import compute_kpi_series
from src.kpi.knowledge_base import KPIKnowledgeBase


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def telco_df():
    """Telecom churn DataFrame with physical KPI column."""
    np.random.seed(42)
    n = 500
    return pd.DataFrame({
        "customer_id": range(n),
        "churn_status": np.random.randint(0, 2, n),
        "MonthlyCharges": np.random.uniform(20, 120, n),
        "TotalCharges": np.random.uniform(100, 5000, n),
        "tenure_months": np.random.randint(1, 72, n),
        "Contract": np.random.choice(["Month-to-month", "One year", "Two year"], n),
    })


@pytest.fixture
def timeseries_df():
    """Supply chain DataFrame with date + numeric KPI."""
    dates = pd.date_range("2023-01-01", periods=60, freq="ME")
    np.random.seed(0)
    return pd.DataFrame({
        "date": dates,
        "total_quantity": np.random.randint(800, 1200, 60),
        "avg_price_min": np.random.uniform(10, 50, 60),
        "modal_price": np.random.uniform(20, 80, 60),
    })


@pytest.fixture
def ecommerce_df():
    """E-commerce DataFrame for derived/ratio KPI tests."""
    np.random.seed(7)
    n = 300
    return pd.DataFrame({
        "order_id": range(n),
        "revenue": np.random.uniform(50, 500, n),
        "cost": np.random.uniform(20, 200, n),
        "quantity": np.random.randint(1, 10, n),
        "customer_id": np.random.randint(1, 100, n),
        "converted": np.random.randint(0, 2, n),
        "total_users": np.full(n, 1000),
        "order_date": pd.date_range("2024-01-01", periods=n, freq="D"),
    })


def _make_simple_kpi(col: str, type_: str = "sum", drivers=None) -> EnhancedKPI:
    return EnhancedKPI(
        name=f"Test KPI ({col})",
        type=type_,
        definition=f"Sum of {col}",
        formula=f"SUM({col})",
        source_columns=[col],
        aggregation=KPIAggregation(method="sum", column=col),
        drivers=drivers or [],
        confidence=0.8,
    )


def _make_ratio_kpi(num_col: str, den_col: str, name="Ratio KPI") -> EnhancedKPI:
    return EnhancedKPI(
        name=name,
        type="ratio",
        definition=f"{num_col} / {den_col}",
        formula=f"SUM({num_col}) / SUM({den_col})",
        source_columns=[num_col, den_col],
        numerator=KPIAggregation(method="sum", column=num_col),
        denominator=KPIAggregation(method="sum", column=den_col),
        drivers=[],
        confidence=0.75,
    )


# ─── Tests ────────────────────────────────────────────────────────────────────

# Test 1: Schema Introspector — basic
def test_introspector_basic(telco_df):
    introspector = SchemaIntrospector()
    meta = introspector.introspect(telco_df)
    assert meta.total_rows == 500
    assert "churn_status" in meta.binary_columns or "churn_status" in meta.numeric_columns
    assert "customer_id" in meta.id_columns or "customer_id" in meta.numeric_columns
    assert isinstance(meta.numeric_columns, list)


# Test 2: Schema Introspector — date detection
def test_introspector_date_detection(timeseries_df):
    introspector = SchemaIntrospector()
    meta = introspector.introspect(timeseries_df)
    assert "date" in meta.date_columns


# Test 3: Simple physical KPI computability
def test_computability_simple_physical(telco_df):
    kpi = _make_simple_kpi("churn_status")
    result = kpi_validate(telco_df, kpi)
    assert result.valid is True
    assert result.missing_columns == []


# Test 4: Missing column rejection (hallucination guard)
def test_computability_missing_column(telco_df):
    kpi = _make_simple_kpi("hallucinated_col")
    result = kpi_validate(telco_df, kpi)
    assert result.valid is False
    assert "hallucinated_col" in result.missing_columns[0]
    assert result.error is not None


# Test 5: Ratio KPI computability
def test_computability_ratio_kpi(ecommerce_df):
    kpi = _make_ratio_kpi("revenue", "cost")
    result = kpi_validate(ecommerce_df, kpi)
    assert result.valid is True


# Test 6: KPI executor — simple sum
def test_executor_simple(ecommerce_df):
    kpi = _make_simple_kpi("revenue")
    series, col_name = compute_kpi_series(ecommerce_df, kpi)
    assert len(series) == len(ecommerce_df)
    assert "revenue" in col_name


# Test 7: KPI executor — ratio (row-wise)
def test_executor_ratio(ecommerce_df):
    kpi = _make_ratio_kpi("revenue", "cost")
    series, col_name = compute_kpi_series(ecommerce_df, kpi)
    assert len(series) == len(ecommerce_df)
    assert (series > 0).any()


# Test 8: KPI executor — derived/conditional count (churn rate)
def test_executor_conditional_count(telco_df):
    kpi = EnhancedKPI(
        name="Customer Churn Rate",
        type="ratio",
        definition="churned_customers / total_customers",
        formula="COUNT(churn_status==1) / COUNT(customer_id)",
        source_columns=["churn_status", "customer_id"],
        numerator=KPIAggregation(
            method="conditional_count",
            condition=KPICondition(column="churn_status", operator="==", value=1)
        ),
        denominator=KPIAggregation(method="count_distinct", column="customer_id"),
        drivers=[KPIDriver(column="MonthlyCharges", reason="Price sensitivity", priority=1)],
        confidence=0.9,
    )
    result = kpi_validate(telco_df, kpi)
    assert result.valid is True


# Test 9: Zero denominator raises ZeroDivisionError
def test_executor_zero_denominator():
    df = pd.DataFrame({"numerator": [10, 20, 30], "denominator": [0, 0, 0]})
    kpi = _make_ratio_kpi("numerator", "denominator")
    # Row-wise division — zeros become NaN, filled to 0 (not an error).
    series, _ = compute_kpi_series(df, kpi)
    assert all(series == 0)


# Test 10: Date-based time-series KPI detection
def test_computability_timeseries_detection(timeseries_df):
    kpi = _make_simple_kpi("total_quantity")
    kpi.time_column = "date"
    kpi.source_columns = ["total_quantity"]
    result = kpi_validate(timeseries_df, kpi)
    assert result.valid is True
    assert result.is_time_series is True


# Test 11: Different organisations — KPI catalogs do not leak
def test_knowledge_base_org_isolation(tmp_path, monkeypatch):
    import src.kpi.knowledge_base as kb_module
    monkeypatch.setattr(kb_module, "_BASE_DIR", str(tmp_path / "kpi_knowledge"))
    monkeypatch.setattr(kb_module, "_FEEDBACK_DIR", str(tmp_path / "kpi_knowledge/feedback"))

    kb = KPIKnowledgeBase()
    kpi_a = _make_simple_kpi("revenue")
    kpi_a.organization_id = "org_a"
    kpi_a.name = "Org A Revenue"
    kb.store_kpi("org_a", kpi_a)

    kpi_b = _make_simple_kpi("churn_status")
    kpi_b.organization_id = "org_b"
    kpi_b.name = "Org B Churn"
    kb.store_kpi("org_b", kpi_b)

    catalog_a = kb.get_catalog("org_a")
    catalog_b = kb.get_catalog("org_b")

    names_a = {k["name"] for k in catalog_a}
    names_b = {k["name"] for k in catalog_b}
    assert "Org A Revenue" in names_a
    assert "Org B Churn" not in names_a
    assert "Org A Revenue" not in names_b


# Test 12: RBAC-restricted column — introspector respects allowed list
def test_introspector_rbac_filtering(telco_df):
    introspector = SchemaIntrospector()
    allowed = ["tenure_months", "churn_status"]
    meta = introspector.introspect(telco_df, allowed_columns=allowed)
    col_names = [c.name for c in meta.columns]
    assert "MonthlyCharges" not in col_names
    assert "tenure_months" in col_names


# Test 13: Hallucinated driver column — warning, not error
def test_hallucinated_driver_warning(telco_df):
    kpi = _make_simple_kpi("churn_status", drivers=[
        KPIDriver(column="real_col", reason="real", priority=1),
        KPIDriver(column="invented_col_xyz", reason="hallucinated", priority=2),
    ])
    # churn_status exists; invented_col_xyz does not → should be warned, not failed
    result = kpi_validate(telco_df, kpi)
    assert result.valid is True
    assert any("invented_col_xyz" in w for w in result.warnings)


# Test 14: Feedback is org-scoped and persisted
def test_feedback_org_scoped(tmp_path, monkeypatch):
    import src.kpi.knowledge_base as kb_module
    monkeypatch.setattr(kb_module, "_BASE_DIR", str(tmp_path / "kpi_knowledge"))
    monkeypatch.setattr(kb_module, "_FEEDBACK_DIR", str(tmp_path / "kpi_knowledge/feedback"))

    kb = KPIKnowledgeBase()
    kb.store_feedback("org_a", "Revenue", "Use net_revenue not gross_revenue", "user_1")
    kb.store_feedback("org_b", "Churn", "Churn window is 90 days not 30", "user_2")

    corrections_a = kb.get_feedback("org_a")
    corrections_b = kb.get_feedback("org_b")

    assert len(corrections_a) == 1
    assert corrections_a[0]["correction"] == "Use net_revenue not gross_revenue"
    assert len(corrections_b) == 1
    assert "org_a" not in [c.get("kpi_name") for c in corrections_b]


# Test 15: Schema Mapper with mocked Groq response
def test_schema_mapper_mocked_groq():
    mock_response = MagicMock()
    mock_response.content = json.dumps({
        "name": "Customer Churn Rate",
        "type": "ratio",
        "definition": "Percentage of customers who churned",
        "formula": "churned / total_customers",
        "numerator": {"method": "conditional_count", "column": None, "condition": {"column": "churn_status", "operator": "==", "value": 1}},
        "denominator": {"method": "count", "column": "churn_status", "condition": None},
        "source_columns": ["churn_status"],
        "aggregation": None,
        "dimensions": [],
        "drivers": [{"column": "MonthlyCharges", "reason": "price", "expected_relationship": "positive", "priority": 1}],
        "time_column": None,
        "unit": "%",
        "direction": "lower_is_better",
        "thresholds": {"material_drop_percentage": 0.05, "alert_threshold": None, "target_value": None},
        "confidence": 0.92,
        "assumptions": [],
        "clarification_required": False,
        "clarification_question": None
    })

    from src.kpi.schema_mapper import SchemaMapper
    from src.kpi.models import SchemaColumnMeta, SchemaMetadata

    meta = SchemaMetadata(
        total_rows=500,
        total_columns=2,
        columns=[
            SchemaColumnMeta(name="churn_status", dtype="int64", is_binary=True),
            SchemaColumnMeta(name="MonthlyCharges", dtype="float64", is_numeric=True),
        ],
    )

    with patch("src.kpi.schema_mapper.ChatGroq") as MockGroq:
        instance = MockGroq.return_value
        instance.invoke.return_value = mock_response
        mapper = SchemaMapper(api_key="fake_key", model_name="test-model")
        kpi = mapper.map("Why is churn increasing?", meta)

    assert kpi.name == "Customer Churn Rate"
    assert kpi.type == "ratio"
    assert kpi.confidence == 0.92
    assert len(kpi.drivers) == 1
    assert kpi.drivers[0].column == "MonthlyCharges"
