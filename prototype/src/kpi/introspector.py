"""
Schema Introspector: Produces rich, safe schema metadata from a DataFrame.

Does NOT send raw row data to any LLM.
Respects RBAC — only introspects the columns that have already been allowed.
"""
from __future__ import annotations
import re
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional

from src.kpi.models import SchemaColumnMeta, SchemaMetadata

# Patterns that suggest a column is an ID / structural key
_ID_PATTERNS = re.compile(
    r"(\bid\b|_id$|^id_|uuid|pk|key|index|code)", re.IGNORECASE
)

# Patterns that suggest a column holds dates / times
_DATE_PATTERNS = re.compile(
    r"(date|time|timestamp|_at$|_on$|month|year|week|day|period)", re.IGNORECASE
)

# Patterns for columns that likely carry a target / outcome signal
_TARGET_PATTERNS = re.compile(
    r"(churn|default|fraud|conversion|status|label|target|outcome|flag|score)",
    re.IGNORECASE,
)

_MAX_SAMPLE_VALUES = 5
_MAX_CATEGORICAL_CARDINALITY = 30


def _safe_scalar(value: Any) -> Any:
    """Convert numpy scalars to plain Python for JSON safety."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


class SchemaIntrospector:
    """
    Inspects a pandas DataFrame and returns rich per-column metadata
    that is safe to send to the LLM (no raw row data, only statistics).
    """

    def introspect(
        self, df: pd.DataFrame, allowed_columns: Optional[List[str]] = None
    ) -> SchemaMetadata:
        """
        Parameters
        ----------
        df              : The (already filtered by RBAC) DataFrame.
        allowed_columns : If provided, only these columns are introspected.
                          This is the second RBAC gate — if the caller already
                          filtered df.columns, pass None.
        Returns
        -------
        SchemaMetadata Pydantic object.
        """
        if allowed_columns:
            df = df[[c for c in allowed_columns if c in df.columns]]

        columns_meta: List[SchemaColumnMeta] = []
        date_cols: List[str] = []
        id_cols: List[str] = []
        cat_cols: List[str] = []
        num_cols: List[str] = []
        bin_cols: List[str] = []
        target_cols: List[str] = []

        for col in df.columns:
            series = df[col]
            dtype_str = str(series.dtype)
            n_total = len(series)
            n_null = int(series.isna().sum())
            null_pct = round(n_null / n_total, 4) if n_total else 0.0
            n_unique = int(series.nunique(dropna=True))

            # Classify column
            is_datetime = False
            is_numeric = False
            is_binary = False
            is_categorical = False
            is_id_like = bool(_ID_PATTERNS.search(col))

            # Try datetime parsing
            if "datetime" in dtype_str or "date" in dtype_str or _DATE_PATTERNS.search(col):
                is_datetime = True
                date_cols.append(col)
            elif "int" in dtype_str or "float" in dtype_str:
                is_numeric = True
                if n_unique == 2:
                    is_binary = True
                    bin_cols.append(col)
                else:
                    num_cols.append(col)
            elif "bool" in dtype_str:
                is_binary = True
                bin_cols.append(col)
            elif "object" in dtype_str or "category" in dtype_str or "string" in dtype_str:
                if n_unique <= _MAX_CATEGORICAL_CARDINALITY:
                    is_categorical = True
                    cat_cols.append(col)
                else:
                    is_id_like = True  # High-cardinality strings are likely IDs

            if is_id_like and col not in id_cols:
                id_cols.append(col)

            if _TARGET_PATTERNS.search(col):
                target_cols.append(col)

            # Safe sample values (no PII, just representative)
            sample_raw = series.dropna().head(_MAX_SAMPLE_VALUES).tolist()
            sample_values = [_safe_scalar(v) for v in sample_raw]

            # Numerical stats
            min_v = max_v = mean_v = median_v = None
            if is_numeric:
                try:
                    min_v = _safe_scalar(series.min())
                    max_v = _safe_scalar(series.max())
                    mean_v = _safe_scalar(series.mean())
                    median_v = _safe_scalar(series.median())
                except Exception:
                    pass

            columns_meta.append(
                SchemaColumnMeta(
                    name=col,
                    dtype=dtype_str,
                    nullable=bool(n_null > 0),
                    unique_count=n_unique,
                    null_pct=null_pct,
                    sample_values=sample_values,
                    min_value=min_v,
                    max_value=max_v,
                    mean_value=mean_v,
                    median_value=median_v,
                    is_datetime=is_datetime,
                    is_id_like=is_id_like,
                    is_binary=is_binary,
                    is_categorical=is_categorical,
                    is_numeric=is_numeric,
                )
            )

        return SchemaMetadata(
            total_rows=len(df),
            total_columns=len(df.columns),
            columns=columns_meta,
            date_columns=date_cols,
            id_columns=id_cols,
            categorical_columns=cat_cols,
            numeric_columns=num_cols,
            binary_columns=bin_cols,
            target_like_columns=target_cols,
        )
