from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import os
import sys
from dotenv import load_dotenv

load_dotenv()

# Ensure src/ is in the python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.orchestration.pipeline import OrchestratorPipeline
from src.database.connectors import DBConnector
from src.utils.telemetry import TelemetryLogger
from src.api.auth import create_access_token, get_current_user

# New KPI Intelligence Layer
from src.kpi.introspector import SchemaIntrospector
from src.kpi.schema_mapper import SchemaMapper
from src.kpi.computability import validate as kpi_validate
from src.kpi.executor import compute_kpi_series
from src.kpi.knowledge_base import KPIKnowledgeBase
from src.llm.reasoning_model import ReasoningModel

app = FastAPI(title="BusinessIntelligence.ai API — KPI Intelligence v2")
db_connector = DBConnector()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request / Response Models ────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    query: str
    org_id: str
    model: str

class LoginRequest(BaseModel):
    role: str

class FeedbackRequest(BaseModel):
    kpi: str
    correction: str
    user_id: str
    org_id: str = ""   # now required for org-scoping

class DatabaseRequest(BaseModel):
    org_name: str
    connection_string: str
    query: str


# ─── Helpers ──────────────────────────────────────────────────────────────────

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        import numpy as np
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.bool_): return bool(obj)
        return super().default(obj)


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/api/models")
async def get_models():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not found in environment.")
    try:
        import groq
        client = groq.Groq(api_key=api_key)
        active_models = client.models.list().data
        valid_models = [
            m.id for m in active_models
            if "whisper" not in m.id.lower() and "guard" not in m.id.lower()
        ]
        return {"models": valid_models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/login")
async def login(req: LoginRequest):
    try:
        token = create_access_token(req.role)
        return {"access_token": token, "token_type": "bearer"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/databases")
async def get_databases():
    return {"databases": db_connector.list_databases()}


@app.post("/api/databases")
async def add_database(req: DatabaseRequest):
    try:
        db_connector.add_database(req.org_name, req.connection_string, req.query)
        return {"status": "success", "message": f"Database '{req.org_name}' registered successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analyze")
async def analyze_data(req: AnalyzeRequest, current_user: dict = Depends(get_current_user)):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not found in environment")

    telemetry_logger = TelemetryLogger()

    try:
        # ── 1. Fetch Data ────────────────────────────────────────────────────
        try:
            df = db_connector.fetch_org_data(req.org_id)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Database Connection Error: {str(e)}")

        # ── 2. RBAC — filter columns by user entitlements ────────────────────
        user_entitlements = current_user.get("entitlements", [])
        if user_entitlements:
            allowed_cols = [c for c in df.columns if c in user_entitlements]
            df = df[allowed_cols]

        # ── 3. Schema Introspection ──────────────────────────────────────────
        introspector = SchemaIntrospector()
        schema_meta = introspector.introspect(df)

        # ── 4. KPI Knowledge Base — fetch existing catalog + feedback ────────
        kb = KPIKnowledgeBase()
        existing_catalog = kb.get_catalog(req.org_id)
        feedback_corrections = kb.get_feedback(req.org_id)

        # ── 5. Schema Mapper (Groq) → EnhancedKPI ───────────────────────────
        mapper = SchemaMapper(api_key, req.model)
        try:
            enhanced_kpi, mapper_tokens, mapper_cost = mapper.map(
                user_query=req.query,
                schema_meta=schema_meta,
                existing_catalog=existing_catalog,
                feedback_corrections=feedback_corrections,
                org_id=req.org_id,
            )
            telemetry_logger.add_metric(mapper_tokens, mapper_cost)
        except ValueError as e:
            raise HTTPException(status_code=500, detail=f"Schema Mapping failed: {str(e)}")

        # Handle clarification required
        if enhanced_kpi.clarification_required:
            return {
                "status": "clarification_required",
                "clarification_question": enhanced_kpi.clarification_question,
                "mapped_kpi": None,
                "enhanced_kpi": json.loads(enhanced_kpi.model_dump_json()),
                "mapped_drivers": [],
                "computability": None,
                "evidence_graph": None,
                "story": None,
                "telemetry": telemetry_logger.get_telemetry(),
                "rows_analyzed": len(df),
                "active_model": req.model,
                "retrieved_rag_context": {},
            }

        # ── 6. KPI Computability Check ───────────────────────────────────────
        computability = kpi_validate(df, enhanced_kpi)
        if not computability.valid:
            raise HTTPException(
                status_code=400,
                detail=f"KPI Computability Check Failed: {computability.error}",
            )

        # ── 7. KPI Execution (safe, deterministic) ───────────────────────────
        try:
            kpi_series, kpi_col_name = compute_kpi_series(df, enhanced_kpi)
        except (ValueError, ZeroDivisionError) as e:
            raise HTTPException(status_code=400, detail=f"KPI Execution Error: {str(e)}")

        # ── 8. Persist KPI definition to Knowledge Base ──────────────────────
        try:
            kb.store_kpi(req.org_id, enhanced_kpi)
        except Exception:
            pass  # Non-fatal

        # ── 9. ML Pipeline (Causal + Time-Series) ───────────────────────────
        pipeline = OrchestratorPipeline()
        driver_cols = [
            d.column for d in enhanced_kpi.drivers
            if d.column in df.columns
        ]
        try:
            evidence_graph = pipeline.run_analysis(
                df=df,
                kpi_name=kpi_col_name,
                drivers=driver_cols,
                enhanced_kpi=enhanced_kpi,
                kpi_series=kpi_series,
                computability=computability,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"ML Engine failed: {str(e)}")

        # ── 10. Reasoning Model (LangGraph) ─────────────────────────────────
        reasoning_model = ReasoningModel(api_key, req.model)
        if evidence_graph.get("status") == "anomaly_detected":
            status, change, factors, actions, llm_telemetry = reasoning_model.generate_story(
                evidence_graph,
                persona=current_user.get("persona", "Executive Team"),
                db_connector=db_connector,
                org_id=req.org_id,
            )
            telemetry_logger.add_metric(
                llm_telemetry.get("tokens", 0), llm_telemetry.get("cost_usd", 0)
            )
            story_dict = {
                "status": status,
                "change_detection": change,
                "contributing_factors": factors,
                "recommended_actions": actions,
                "db_insights": llm_telemetry.get("db_insights", [])
            }
        else:
            story_dict = {
                "status": "No Anomalies",
                "change_detection": evidence_graph.get("message", "No anomalies detected."),
                "contributing_factors": "N/A",
                "recommended_actions": "N/A",
            }

        # ── 11. Build full response ──────────────────────────────────────────
        result_dict = {
            # Legacy fields (frontend compatibility)
            "mapped_kpi": kpi_col_name,
            "mapped_drivers": driver_cols,
            "retrieved_rag_context": {},
            "active_model": req.model,
            "rows_analyzed": len(df),

            # New fields
            "enhanced_kpi": json.loads(enhanced_kpi.model_dump_json()),
            "computability": computability.model_dump(),
            "schema_summary": {
                "total_rows": schema_meta.total_rows,
                "total_columns": schema_meta.total_columns,
                "numeric_columns": schema_meta.numeric_columns,
                "date_columns": schema_meta.date_columns,
                "categorical_columns": schema_meta.categorical_columns,
            },

            # Existing response structure
            "evidence_graph": evidence_graph,
            "story": story_dict,
            "telemetry": telemetry_logger.get_telemetry(),
        }

        return json.loads(json.dumps(result_dict, cls=NumpyEncoder))

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/feedback")
async def submit_feedback(req: FeedbackRequest):
    """
    Real feedback persistence — org-scoped.
    Stores the correction in config/kpi_knowledge/feedback/<org_slug>.json
    """
    if not req.org_id:
        return {"status": "warning", "message": "No org_id provided. Feedback not persisted."}
    try:
        kb = KPIKnowledgeBase()
        kb.store_feedback(
            org_id=req.org_id,
            kpi_name=req.kpi,
            correction=req.correction,
            user_id=req.user_id,
        )
        return {
            "status": "success",
            "message": f"Feedback for '{req.kpi}' persisted. Will influence future KPI mapping for '{req.org_id}'.",
            "correction": req.correction,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/kpi-catalog/{org_id}")
async def get_kpi_catalog(org_id: str):
    """Return the stored KPI catalog for an organisation."""
    kb = KPIKnowledgeBase()
    return {"org_id": org_id, "kpis": kb.get_catalog(org_id)}
