# BusinessIntelligence.ai — KPI Intelligence Layer

An enterprise AI analyst platform that connects to any SQL database and autonomously discovers, validates, and reasons about business KPIs using a production-grade LangGraph reasoning engine.

---

## Architecture

```
POST /api/analyze
    │
    ▼
DBConnector (read-only SQL)
    │
    ▼
RBAC / JWT Entitlements Filter   ← src/api/auth.py
    │
    ▼
Schema Introspector               ← src/kpi/introspector.py
    │  (rich metadata: dtypes, nulls, ranges, samples, date/categorical/binary detection)
    │  (NO raw rows sent to LLM)
    ▼
KPI Knowledge Base (org-scoped)   ← src/kpi/knowledge_base.py
    │  (existing KPI defs + user feedback corrections for THIS org)
    ▼
Schema Mapper / Groq LLM          ← src/kpi/schema_mapper.py
    │  (zero-shot inference; returns EnhancedKPI Pydantic object)
    │  (hallucinated columns rejected downstream)
    ▼
KPI Computability Check           ← src/kpi/computability.py
    │  (validates source columns, numerator/denominator, date col, variance)
    │  (returns structured ComputabilityResult — not a bare exception)
    ▼
KPI Execution Engine (safe)       ← src/kpi/executor.py
    │  (deterministic; NO eval(); evaluates structured KPI spec)
    │  (supports: sum, count, ratio, conditional_count, count_distinct, etc.)
    ▼
Quantitative Logic Ensemble       ← src/engine/ml_models.py
    │  (IsolationForest anomaly detection + RandomForest causal drivers)
    │  (dynamic time column from EnhancedKPI.time_column)
    ▼
Statistical Rules                 ← src/engine/rules.py
    │  (materiality threshold check)
    ▼
Evidence Graph
    │
    ▼
Reasoning Model (LangGraph)       ← src/llm/reasoning_model.py
    │  (2-node CoT: analyze_evidence → format_output)
    │  (Confidence Scorer → Abstention if score < 0.2)
    ▼
Story Generator (narrative)
    │
    ▼
Telemetry Logger                  ← src/utils/telemetry.py
    │
    ▼
API Response (enhanced)
```

---

## Key Concepts

### LLM Inferred KPI vs. Validated/Computable KPI

| Concept | Description |
|---------|-------------|
| **LLM Inferred KPI** | The Groq model reads schema metadata and proposes a KPI spec (name, formula, source columns, drivers). This is a *hypothesis* — the LLM may hallucinate column names. |
| **Validated KPI** | The `KPIComputabilityCheck` deterministically verifies that every column in the spec exists in the actual DataFrame and that the formula is computable. |
| **Computable/Executed KPI** | The `KPIExecutor` safely evaluates the validated spec against real data, producing a pandas Series. **No `eval()` is used.** |

---

## Schema Introspector

**File:** `src/kpi/introspector.py`

Inspects the DataFrame and produces a `SchemaMetadata` object with per-column:
- dtype, unique count, null %, sample values (≤5)
- min/max/mean/median for numeric columns
- Automatic classification: datetime, ID-like, binary, categorical, numeric, target-like

Only schema metadata (no raw rows) is sent to the LLM.

---

## Schema Mapper (LLM)

**File:** `src/kpi/schema_mapper.py`

Sends a structured schema summary + user query to Groq and returns a validated `EnhancedKPI` Pydantic object containing:
- KPI name, type, definition, formula
- Numerator / denominator for ratio KPIs
- Source columns, drivers, time column
- Confidence score, assumptions, clarification_required flag

---

## Enhanced KPI

**File:** `src/kpi/models.py`

```python
class EnhancedKPI(BaseModel):
    name: str
    type: Literal["sum","count","count_distinct","mean","median","ratio","derived","rate"]
    definition: str
    formula: str
    numerator: Optional[KPIAggregation]
    denominator: Optional[KPIAggregation]
    source_columns: List[str]
    aggregation: Optional[KPIAggregation]
    dimensions: List[str]
    drivers: List[KPIDriver]
    time_column: Optional[str]
    unit: str
    direction: Literal["higher_is_better","lower_is_better","neutral"]
    thresholds: KPIThreshold
    lineage: KPILineage
    confidence: float
    organization_id: str
    schema_version: str
    assumptions: List[str]
    clarification_required: bool
    clarification_question: Optional[str]
```

---

## Org-Scoped KPI Knowledge Base

**File:** `src/kpi/knowledge_base.py`
**Storage:** `config/kpi_knowledge/<org_slug>.json`

Each organisation has its own KPI catalog. KPI definitions **do not leak between organisations**. The `store_kpi()` method upserts definitions by name.

---

## Human Feedback Loop

**Endpoint:** `POST /api/feedback`

Now **actually persists** corrections:
```json
{
  "kpi": "Revenue",
  "correction": "Use net_revenue not gross_revenue",
  "user_id": "user_123",
  "org_id": "Org C (Global Superstore)"
}
```

Stored in `config/kpi_knowledge/feedback/<org_slug>.json`.
On the next `POST /api/analyze` call for the same org, these corrections are injected into the Schema Mapper prompt.

---

## KPI Execution Engine

**File:** `src/kpi/executor.py`

Safe, deterministic evaluation. Supported operations:
- `sum`, `count`, `count_distinct`, `mean`, `median`, `min`, `max`
- `conditional_count` (e.g., `COUNT WHERE churn_status == 1`)
- Row-wise ratio (`revenue / cost` per row)
- Global-scalar denominator (`churned / TOTAL_CUSTOMERS`)

**Never uses `eval()` or `exec()`.**

---

## Environment Variables

```bash
GROQ_API_KEY=your_groq_api_key_here
```

---

## Running

```bash
# Backend
cd prototype
.venv/bin/uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# Frontend
cd prototype/frontend
npm run dev
```

---

## Running Tests

```bash
cd prototype
.venv/bin/pytest tests/test_kpi_layer.py -v
```

No real Groq API key required — Groq is mocked in all tests.

---

## Example Request

```json
POST /api/analyze
{
  "query": "Why is customer churn increasing?",
  "org_id": "Org A (Telecommunications)",
  "model": "llama-3.1-70b-versatile"
}
```

## Example Enhanced KPI (LLM output → Pydantic validated)

```json
{
  "name": "Customer Churn Rate",
  "type": "ratio",
  "definition": "Percentage of customers who churned in the period",
  "formula": "COUNT(churn_status==1) / COUNT(DISTINCT customer_id)",
  "numerator": {"method": "conditional_count", "condition": {"column":"churn_status","operator":"==","value":1}},
  "denominator": {"method": "count_distinct", "column": "customer_id"},
  "source_columns": ["churn_status", "customer_id"],
  "drivers": [
    {"column": "MonthlyCharges", "reason": "Price sensitivity drives churn", "priority": 1},
    {"column": "Contract", "reason": "Month-to-month customers churn more", "priority": 2}
  ],
  "time_column": null,
  "unit": "%",
  "direction": "lower_is_better",
  "confidence": 0.91
}
```

## Example Computability Result

```json
{
  "valid": true,
  "kpi_name": "Customer Churn Rate",
  "formula_valid": true,
  "required_columns": ["churn_status", "customer_id"],
  "missing_columns": [],
  "warnings": [],
  "is_time_series": false,
  "is_classification": true
}
```

## Example Final Response

```json
{
  "mapped_kpi": "churn_status",
  "enhanced_kpi": { ... },
  "mapped_drivers": ["MonthlyCharges", "TotalCharges", "Contract"],
  "computability": {"valid": true, "warnings": [], ...},
  "evidence_graph": {"status": "anomaly_detected", "drop_percentage": 0.266, ...},
  "story": {
    "status": "Complete",
    "change_detection": "...",
    "contributing_factors": "...",
    "recommended_actions": "..."
  },
  "telemetry": {"latency_ms": 8420, "tokens": 3200, "cost_usd": 0.00032}
}
```
