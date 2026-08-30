"""
Safe KPI Execution Engine.

Executes an EnhancedKPI spec deterministically against a DataFrame.
NEVER uses eval() or exec() on LLM output.
The LLM produces a structured spec; this module evaluates it safely.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Tuple

from src.kpi.models import EnhancedKPI, KPIAggregation, KPICondition


_SAFE_OPS = {
    "==": lambda s, v: s == v,
    "!=": lambda s, v: s != v,
    ">": lambda s, v: s > v,
    ">=": lambda s, v: s >= v,
    "<": lambda s, v: s < v,
    "<=": lambda s, v: s <= v,
    "isin": lambda s, v: s.isin(v if isinstance(v, list) else [v]),
    "notnull": lambda s, _: s.notna(),
}


def _apply_condition(df: pd.DataFrame, cond: KPICondition) -> pd.Series:
    """Return a boolean mask series from a KPICondition."""
    series = df[cond.column]
    op_fn = _SAFE_OPS.get(cond.operator)
    if op_fn is None:
        raise ValueError(f"Unsupported operator: {cond.operator}")
    return op_fn(series, cond.value)


def _execute_aggregation(df: pd.DataFrame, agg: KPIAggregation) -> float:
    """Execute a single aggregation spec and return a scalar."""
    method = agg.method

    if method == "conditional_count":
        if agg.condition is None:
            raise ValueError("conditional_count requires a condition.")
        mask = _apply_condition(df, agg.condition)
        return float(mask.sum())

    if method == "count":
        col = agg.column or df.columns[0]
        return float(df[col].count())

    if method == "count_distinct":
        col = agg.column
        if col is None:
            raise ValueError("count_distinct requires a column.")
        return float(df[col].nunique())

    col = agg.column
    if col is None or col not in df.columns:
        raise ValueError(f"Column '{col}' not found for aggregation '{method}'.")

    series = df[col].dropna()

    if method == "sum":
        return float(series.sum())
    if method == "mean":
        return float(series.mean())
    if method == "median":
        return float(series.median())
    if method == "min":
        return float(series.min())
    if method == "max":
        return float(series.max())

    raise ValueError(f"Unknown aggregation method: {method}")


def compute_kpi_series(df: pd.DataFrame, kpi: EnhancedKPI) -> Tuple[pd.Series, str]:
    """
    Compute the KPI and return:
      - A pandas Series of KPI values (one per row or one per time bucket).
      - The column name to pass to the ML engine.

    For simple column-based KPIs this just returns the column.
    For ratio / derived KPIs this evaluates the formula row-wise where possible,
    or returns the primary source column with a global scalar applied as metadata.

    The ML engine needs a numeric Series indexed identically to df.
    """
    kpi_type = kpi.type

    # --- Simple physical column ---
    if kpi_type in ("sum", "count", "mean", "median") and kpi.aggregation:
        col = kpi.aggregation.column
        if col and col in df.columns:
            return df[col], col

    # --- Ratio / rate KPI ---
    if kpi_type in ("ratio", "rate") and kpi.numerator and kpi.denominator:
        num_col = kpi.numerator.column
        den_col = kpi.denominator.column

        # For time-series ratio we need a per-row or per-period value.
        if num_col and num_col in df.columns:
            numerator_series = df[num_col]
        elif kpi.numerator.condition:
            mask = _apply_condition(df, kpi.numerator.condition)
            numerator_series = mask.astype(float)
        else:
            raise ValueError("Cannot determine numerator series for ratio KPI.")

        if den_col and den_col in df.columns:
            denominator_series = df[den_col].replace(0, np.nan)
            ratio_series = numerator_series / denominator_series
            return ratio_series.fillna(0), f"{num_col}_ratio"

        # Scalar denominator (global count)
        global_den = _execute_aggregation(df, kpi.denominator)
        if global_den == 0:
            raise ZeroDivisionError(f"Denominator for KPI '{kpi.name}' is zero.")
        ratio_series = numerator_series / global_den
        return ratio_series, f"{kpi.numerator.column or 'num'}_rate"

    # --- Fallback: use first available source column ---
    for col in kpi.source_columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            return df[col], col

    raise ValueError(
        f"KPI Executor: cannot compute KPI '{kpi.name}'. "
        f"Source columns {kpi.source_columns} are not numeric or not present."
    )
