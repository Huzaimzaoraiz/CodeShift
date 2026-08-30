"""
Canonical Pydantic models for the Enhanced KPI Intelligence Layer.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field, model_validator
import datetime


class KPICondition(BaseModel):
    """A filter condition: column operator value."""
    column: str
    operator: Literal["==", "!=", ">", ">=", "<", "<=", "isin", "notnull"]
    value: Optional[Any] = None


class KPIAggregation(BaseModel):
    """Describes how to aggregate a set of rows into a scalar."""
    method: Literal["sum", "count", "count_distinct", "mean", "median", "min", "max", "conditional_count"]
    column: Optional[str] = None
    condition: Optional[KPICondition] = None


class KPIDriver(BaseModel):
    """A candidate causal driver column."""
    column: str
    reason: str
    expected_relationship: Optional[str] = None
    priority: int = 1


class KPIThreshold(BaseModel):
    material_drop_percentage: float = 0.10
    alert_threshold: Optional[float] = None
    target_value: Optional[float] = None


class KPILineage(BaseModel):
    source_tables: List[str] = Field(default_factory=list)
    transformation: Optional[str] = None
    created_by: str = "auto"
    last_updated: str = Field(default_factory=lambda: datetime.datetime.utcnow().isoformat())


class EnhancedKPI(BaseModel):
    """
    Canonical KPI specification produced by the Schema Mapper and validated
    by the Computability Check before being executed by the KPI Executor.
    """
    name: str
    type: Literal["sum", "count", "count_distinct", "mean", "median", "ratio", "derived", "rate"]
    definition: str
    formula: str

    # Numerator / denominator for ratio KPIs
    numerator: Optional[KPIAggregation] = None
    denominator: Optional[KPIAggregation] = None

    # Physical columns that must exist in the dataframe
    source_columns: List[str] = Field(default_factory=list)

    # Simple aggregation for non-ratio KPIs
    aggregation: Optional[KPIAggregation] = None

    # Dimension columns (categorical breakdowns)
    dimensions: List[str] = Field(default_factory=list)

    # Drivers for the ML causal analysis
    drivers: List[KPIDriver] = Field(default_factory=list)

    # The column name holding time information (can be any name, not just "date")
    time_column: Optional[str] = None

    unit: str = ""
    direction: Literal["higher_is_better", "lower_is_better", "neutral"] = "neutral"
    thresholds: KPIThreshold = Field(default_factory=KPIThreshold)
    lineage: KPILineage = Field(default_factory=KPILineage)
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    organization_id: str = ""
    schema_version: str = "1.0"

    # LLM-provided context
    assumptions: List[str] = Field(default_factory=list)
    clarification_required: bool = False
    clarification_question: Optional[str] = None

    @model_validator(mode="after")
    def validate_ratio_has_parts(self) -> "EnhancedKPI":
        if self.type == "ratio":
            if self.numerator is None or self.denominator is None:
                raise ValueError("KPI type 'ratio' requires both numerator and denominator.")
        return self

    def driver_columns(self) -> List[str]:
        """Convenience: return just the column names from drivers."""
        return [d.column for d in self.drivers]


class ComputabilityResult(BaseModel):
    """Structured result from the KPI Computability Check."""
    valid: bool
    kpi_name: str
    formula_valid: bool = True
    required_columns: List[str] = Field(default_factory=list)
    missing_columns: List[str] = Field(default_factory=list)
    data_quality: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    error: Optional[str] = None
    is_time_series: bool = False
    is_classification: bool = False


class SchemaColumnMeta(BaseModel):
    """Rich metadata for one column produced by the SchemaIntrospector."""
    name: str
    dtype: str
    nullable: bool = True
    unique_count: int = 0
    null_pct: float = 0.0
    sample_values: List[Any] = Field(default_factory=list)
    min_value: Optional[Any] = None
    max_value: Optional[Any] = None
    mean_value: Optional[float] = None
    median_value: Optional[float] = None
    is_datetime: bool = False
    is_id_like: bool = False
    is_binary: bool = False
    is_categorical: bool = False
    is_numeric: bool = False


class SchemaMetadata(BaseModel):
    """Full schema introspection result for a DataFrame."""
    total_rows: int
    total_columns: int
    columns: List[SchemaColumnMeta]
    date_columns: List[str] = Field(default_factory=list)
    id_columns: List[str] = Field(default_factory=list)
    categorical_columns: List[str] = Field(default_factory=list)
    numeric_columns: List[str] = Field(default_factory=list)
    binary_columns: List[str] = Field(default_factory=list)
    target_like_columns: List[str] = Field(default_factory=list)
