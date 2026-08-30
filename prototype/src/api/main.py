from fastapi import FastAPI, HTTPException
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
from src.llm.schema_inference import SchemaInferenceLLM
from src.llm.reasoning_model import ReasoningModel
from src.database.connectors import DBConnector
from src.utils.computability import KPIComputabilityCheck
from src.utils.telemetry import TelemetryLogger
from src.api.auth import create_access_token, get_current_user
from fastapi import Depends

app = FastAPI(title="BusinessIntelligence.ai API")
db_connector = DBConnector()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

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

class DatabaseRequest(BaseModel):
    org_name: str
    connection_string: str
    query: str

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        import numpy as np
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.bool_): return bool(obj)
        return super(NumpyEncoder, self).default(obj)


@app.get("/api/models")
async def get_models():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not found in environment.")
    try:
        import groq
        client = groq.Groq(api_key=api_key)
        active_models = client.models.list().data
        valid_models = [m.id for m in active_models if 'whisper' not in m.id.lower() and 'guard' not in m.id.lower()]
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
        return {"status": "success", "message": f"Database {req.org_name} registered successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze")
async def analyze_data(req: AnalyzeRequest, current_user: dict = Depends(get_current_user)):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not found in environment")
        
    telemetry_logger = TelemetryLogger()
        
    try:
        schema_inference = SchemaInferenceLLM(api_key, req.model)
        reasoning_model = ReasoningModel(api_key, req.model)
        pipeline = OrchestratorPipeline()
        
        # 1. Fetch Data
        try:
            df = db_connector.fetch_org_data(req.org_id)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Database Connection Error: {str(e)}")

        # 2. Extract Schema & Apply Data Entitlements (RBAC)
        db_schema = {col: str(dtype) for col, dtype in df.dtypes.items()}
        user_entitlements = current_user.get("entitlements", [])
        if user_entitlements:
            db_schema = {col: dtype for col, dtype in db_schema.items() if col in user_entitlements}

        # 3. Schema Inference (LLM)
        try:
            semantic_mapping, retrieved_context = schema_inference.schema_mapper(req.query, db_schema)
            mapped_kpi = semantic_mapping.get("kpi")
            mapped_drivers = semantic_mapping.get("drivers", [])
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"LLM Schema Inference failed: {str(e)}")

        # 4. KPI Computability Check
        try:
            KPIComputabilityCheck.verify(df, mapped_kpi, mapped_drivers)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Computability Check Failed: {str(e)}")

        # 5. Run Quantitative Logic Ensemble
        try:
            evidence_graph = pipeline.run_analysis(df, kpi_name=mapped_kpi, drivers=mapped_drivers)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Machine Learning Engine failed: {str(e)}")

        # 6. Reasoning Model (LLM)
        if evidence_graph.get('status') == 'anomaly_detected':
            status, change, factors, actions, llm_telemetry = reasoning_model.generate_story(evidence_graph, persona=current_user.get("persona", "Executive Team"))
            telemetry_logger.add_metric(llm_telemetry.get('tokens', 0), llm_telemetry.get('cost_usd', 0))
            story_dict = {
                "status": status,
                "change_detection": change,
                "contributing_factors": factors,
                "recommended_actions": actions
            }
        else:
            story_dict = {
                "status": "No Anomalies",
                "change_detection": evidence_graph.get('message', 'No anomalies detected.'),
                "contributing_factors": "N/A",
                "recommended_actions": "N/A"
            }

        result_dict = {
            "evidence_graph": evidence_graph,
            "story": story_dict,
            "telemetry": telemetry_logger.get_telemetry(),
            "mapped_kpi": mapped_kpi,
            "mapped_drivers": mapped_drivers,
            "retrieved_rag_context": retrieved_context,
            "active_model": schema_inference.active_model,
            "rows_analyzed": len(df)
        }
        
        return json.loads(json.dumps(result_dict, cls=NumpyEncoder))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/feedback")
async def submit_feedback(req: FeedbackRequest):
    # This is a stub for the Human Feedback Loop
    # In a full deployment, this would write to config/few_shot_tuning.json
    return {"status": "Feedback received and queued for tuning.", "correction": req.correction}
