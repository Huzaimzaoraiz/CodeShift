"""
Enhanced Schema Mapper (LLM layer).

Sends rich schema metadata (not raw rows) to Groq and returns a fully-validated
EnhancedKPI Pydantic object.

Key design decisions:
- Only schema metadata is sent to LLM (no raw data).
- LLM output is parsed and validated with Pydantic before use.
- Hallucinated columns are rejected by the Computability Check (downstream).
- If LLM cannot determine KPI confidently, clarification_required=True is returned.
"""
from __future__ import annotations
import json
import re
from typing import Any, Dict, List, Optional, Tuple

# pyrefly: ignore [missing-import]
from langchain_groq import ChatGroq
from pydantic import ValidationError

from src.kpi.models import (
    EnhancedKPI, KPIAggregation, KPICondition, KPIDriver,
    KPIThreshold, KPILineage, SchemaMetadata
)


_SYSTEM_PROMPT = """You are an expert Business Intelligence Architect.
Your role is to analyse a database schema and a user's natural-language question,
then produce a precise, machine-executable KPI specification.

Rules:
1. Never invent columns that do not appear in the schema.
2. A KPI may be derived (computed from multiple columns); say so explicitly.
3. If you are not confident, set clarification_required to true.
4. Respond with ONLY a single valid JSON object — no markdown fences.
"""

_USER_TEMPLATE = """User Question: {query}

Schema Metadata:
{schema_summary}

Existing KPI Catalog for this org (may be empty):
{existing_catalog}

User Feedback / Corrections (apply these):
{feedback_corrections}

Produce a JSON object with this EXACT structure:
{{
  "name": "Human-readable KPI name",
  "type": "sum|count|count_distinct|mean|median|ratio|derived|rate",
  "definition": "Plain-English definition of what this KPI measures",
  "formula": "Mathematical or pseudo-code formula, e.g. SUM(revenue) or churned / total",
  "numerator": {{                            // null if not a ratio
    "method": "sum|count|count_distinct|mean|median|min|max|conditional_count",
    "column": "column_name",               // null if method needs a condition
    "condition": {{                          // null if no filter needed
      "column": "col",
      "operator": "==|!=|>|>=|<|<=|isin|notnull",
      "value": <value_or_null>
    }}
  }},
  "denominator": {{                          // null if not a ratio
    "method": "count|count_distinct|sum",
    "column": "column_name",
    "condition": null
  }},
  "source_columns": ["col1", "col2"],       // ALL physical columns needed
  "aggregation": {{                          // null if ratio
    "method": "sum|count|count_distinct|mean|median",
    "column": "column_name"
  }},
  "dimensions": ["cat_col1"],               // breakdown dimensions
  "drivers": [                              // causal candidate columns (no IDs, no dates, no KPI col itself)
    {{
      "column": "col_name",
      "reason": "Why this drives the KPI",
      "expected_relationship": "positive|negative|unknown",
      "priority": 1
    }}
  ],
  "time_column": "column_name_or_null",     // date/time column if available
  "unit": "%|$|units|etc",
  "direction": "higher_is_better|lower_is_better|neutral",
  "thresholds": {{
    "material_drop_percentage": 0.10,
    "alert_threshold": null,
    "target_value": null
  }},
  "confidence": 0.0,                        // 0.0-1.0, your own confidence
  "assumptions": ["assumption1"],
  "clarification_required": false,
  "clarification_question": null
}}"""


def _clean_json(raw: str) -> str:
    """Strip markdown fences and thinking blocks, then extract the JSON object."""
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    raw = re.sub(r"```[a-z]*", "", raw)
    raw = raw.replace("```", "")
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        return raw[start : end + 1]
    return raw.strip()


def _schema_summary(meta: SchemaMetadata) -> str:
    """Produce a concise, token-efficient schema summary for the LLM."""
    lines = [f"Rows: {meta.total_rows}  Columns: {meta.total_columns}", ""]
    lines.append("Columns (name | dtype | unique | null% | sample):")
    for col in meta.columns:
        sample = ", ".join(str(v) for v in col.sample_values[:3])
        flags = []
        if col.is_datetime:
            flags.append("DATETIME")
        if col.is_id_like:
            flags.append("ID-LIKE")
        if col.is_binary:
            flags.append("BINARY")
        if col.is_categorical:
            flags.append("CATEGORICAL")
        if col.is_numeric:
            num_info = f"min={col.min_value} max={col.max_value} mean={col.mean_value}"
            flags.append(f"NUMERIC({num_info})")
        flag_str = " [" + ", ".join(flags) + "]" if flags else ""
        lines.append(
            f"  {col.name} | {col.dtype} | {col.unique_count} unique | "
            f"{col.null_pct:.1%} null | [{sample}]{flag_str}"
        )
    lines.append(f"\nDate columns: {meta.date_columns or 'none'}")
    lines.append(f"Numeric columns: {meta.numeric_columns}")
    lines.append(f"Categorical columns: {meta.categorical_columns}")
    lines.append(f"Target-like columns: {meta.target_like_columns}")
    return "\n".join(lines)


class SchemaMapper:
    """
    Calls Groq with rich schema metadata and parses the response into
    a validated EnhancedKPI object.
    """

    def __init__(self, api_key: str, model_name: str):
        self.active_model = model_name
        
        # Override for JSON tasks to prevent DeepSeek <think> blocks breaking JSON 
        # and to save on strict TPM limits.
        if "deepseek" in model_name.lower() or "r1" in model_name.lower():
            model_name = "llama-3.3-70b-versatile"
            
        self.llm = ChatGroq(
            api_key=api_key, 
            model_name=model_name, 
            temperature=0.1,
            max_tokens=2000,
            max_retries=5
        )

    def map(
        self,
        user_query: str,
        schema_meta: SchemaMetadata,
        existing_catalog: Optional[List[Dict[str, Any]]] = None,
        feedback_corrections: Optional[List[Dict[str, Any]]] = None,
        org_id: str = "",
    ) -> Tuple[EnhancedKPI, int, float]:
        """
        Returns a validated EnhancedKPI along with tokens and cost. Raises ValueError if the LLM
        output cannot be parsed or validated.
        """
        schema_str = _schema_summary(schema_meta)
        catalog_str = json.dumps(existing_catalog or [], indent=2)
        feedback_str = json.dumps(feedback_corrections or [], indent=2)

        user_msg = _USER_TEMPLATE.format(
            query=user_query,
            schema_summary=schema_str,
            existing_catalog=catalog_str,
            feedback_corrections=feedback_str,
        )

        from langchain_core.messages import SystemMessage, HumanMessage
        messages = [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=user_msg)]

        response = self.llm.invoke(messages)
        raw = response.content

        tokens = 0
        cost = 0.0
        try:
            tokens = response.response_metadata["token_usage"]["total_tokens"]
            cost = tokens * 0.0000001
        except Exception:
            tokens = len(raw.split()) * 1.3
            cost = 0.0001

        cleaned = _clean_json(raw)
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"LLM did not return valid JSON. Raw output: {raw[:300]}"
            ) from exc

        # Inject org_id and convert to Pydantic
        data["organization_id"] = org_id
        try:
            kpi = EnhancedKPI(**data)
        except (ValidationError, TypeError) as exc:
            raise ValueError(f"LLM output failed Pydantic validation: {exc}") from exc

        return kpi, int(tokens), cost
