"""
Upgraded KPI Computability Check.

Validates an EnhancedKPI spec against the actual DataFrame before execution.
Handles derived KPIs, ratio KPIs, date-based KPIs, and hallucinated columns.
Returns a structured ComputabilityResult instead of raising bare exceptions.
"""
from __future__ import annotations
import pandas as pd
from typing import List

from src.kpi.models import EnhancedKPI, ComputabilityResult, KPIAggregation


def _check_aggregation(agg: KPIAggregation, df: pd.DataFrame, prefix: str) -> List[str]:
    """Validate a single aggregation spec against df columns. Returns list of missing cols."""
    missing = []
    if agg.column and agg.column not in df.columns:
        missing.append(f"{prefix}.column='{agg.column}'")
    if agg.condition and agg.condition.column not in df.columns:
        missing.append(f"{prefix}.condition.column='{agg.condition.column}'")
    return missing


def validate(df: pd.DataFrame, kpi: EnhancedKPI) -> ComputabilityResult:
    """
    Full deterministic validation of an EnhancedKPI against a DataFrame.
    """
    missing_cols: List[str] = []
    warnings: List[str] = []
    formula_valid = True
    is_time_series = False
    is_classification = False

    # 1. Validate required source columns
    for col in kpi.source_columns:
        if col not in df.columns:
            missing_cols.append(col)

    # 2. Validate driver columns (hallucination guard)
    for driver in kpi.drivers:
        if driver.column not in df.columns:
            warnings.append(
                f"Driver column '{driver.column}' not found in dataset — will be ignored by ML engine."
            )

    # 3. Ratio KPI — validate numerator and denominator
    if kpi.type == "ratio":
        if kpi.numerator:
            missing_cols += _check_aggregation(kpi.numerator, df, "numerator")
        if kpi.denominator:
            missing_cols += _check_aggregation(kpi.denominator, df, "denominator")

    # 4. Simple aggregation KPI
    elif kpi.aggregation:
        if kpi.aggregation.column and kpi.aggregation.column not in df.columns:
            missing_cols.append(f"aggregation.column='{kpi.aggregation.column}'")

    # 5. Time-series detection via time_column
    if kpi.time_column:
        if kpi.time_column not in df.columns:
            missing_cols.append(f"time_column='{kpi.time_column}'")
        else:
            is_time_series = True

    # 6. Fallback: if there's a literal 'date' column (legacy support)
    if not kpi.time_column and "date" in df.columns:
        is_time_series = True

    # 7. Classification detection (binary KPI col)
    if kpi.source_columns:
        first_col = kpi.source_columns[0]
        if first_col in df.columns and df[first_col].nunique() == 2:
            is_classification = True

    # 8. Data quality checks (only if source columns exist)
    data_quality: dict = {}
    for col in kpi.source_columns:
        if col in df.columns:
            null_pct = float(df[col].isna().mean())
            unique_count = int(df[col].nunique())
            data_quality[col] = {
                "null_pct": round(null_pct, 4),
                "unique_count": unique_count,
            }
            if null_pct > 0.5:
                warnings.append(
                    f"Column '{col}' has {null_pct:.0%} missing values — results may be unreliable."
                )
            if unique_count <= 1:
                warnings.append(
                    f"Column '{col}' has no variance (all values identical)."
                )

    # 9. Denominator zero-check is deferred to KPIExecutor

    # 10. Formula validity
    if missing_cols:
        formula_valid = False

    is_valid = len(missing_cols) == 0

    error_msg = None
    if not is_valid:
        error_msg = (
            f"KPI '{kpi.name}' is not computable. Missing columns: {missing_cols}. "
            f"Tip: The LLM may have hallucinated column names not present in this dataset."
        )

    return ComputabilityResult(
        valid=is_valid,
        kpi_name=kpi.name,
        formula_valid=formula_valid,
        required_columns=kpi.source_columns,
        missing_columns=missing_cols,
        data_quality=data_quality,
        warnings=warnings,
        error=error_msg,
        is_time_series=is_time_series,
        is_classification=is_classification,
    )
