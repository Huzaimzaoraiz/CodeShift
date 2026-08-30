"""
Thin backward-compatibility wrapper.

The real logic now lives in:
  src/kpi/introspector.py  → SchemaIntrospector
  src/kpi/schema_mapper.py → SchemaMapper

This module exposes the old SchemaInferenceLLM interface so that any
remaining callers do not break.  main.py now uses SchemaMapper directly.
"""
from __future__ import annotations
from typing import Dict, Any, Tuple, Optional, List
import json, re

# pyrefly: ignore [missing-import]
from langchain_groq import ChatGroq

from src.kpi.schema_mapper import SchemaMapper
from src.kpi.introspector import SchemaIntrospector
from src.kpi.models import EnhancedKPI, SchemaMetadata
import pandas as pd


class SchemaInferenceLLM:
    """
    Backward-compatible wrapper around SchemaMapper + SchemaIntrospector.
    When main.py calls schema_mapper(query, db_schema_dict), this converts
    the old dict format and returns (legacy_dict, {}).
    """
    def __init__(self, api_key: str, model_name: str):
        self._mapper = SchemaMapper(api_key, model_name)
        self.active_model = model_name

    def schema_mapper(
        self, user_query: str, db_schema: Dict[str, str]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Legacy interface — kept for any remaining callers."""
        # Convert simple dtype dict back to a minimal SchemaMetadata
        from src.kpi.models import SchemaColumnMeta, SchemaMetadata
        cols = []
        for col_name, dtype in db_schema.items():
            is_numeric = any(t in dtype for t in ["int", "float"])
            is_datetime = "datetime" in dtype or "date" in dtype
            cols.append(SchemaColumnMeta(
                name=col_name,
                dtype=dtype,
                is_numeric=is_numeric,
                is_datetime=is_datetime,
            ))
        meta = SchemaMetadata(
            total_rows=0,
            total_columns=len(cols),
            columns=cols,
        )
        try:
            kpi = self._mapper.map(user_query, meta)
            legacy = {
                "kpi": kpi.source_columns[0] if kpi.source_columns else "",
                "drivers": kpi.driver_columns(),
            }
            return legacy, {}
        except Exception as e:
            raise ValueError(str(e))
