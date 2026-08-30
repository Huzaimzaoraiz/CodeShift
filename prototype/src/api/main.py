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
from src.llm.storyteller import LLMStoryteller
from src.database.connectors import DBConnector

app = FastAPI(title="BusinessIntelligence.ai API")

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

def load_kb():
    # Go up three levels: api -> src -> prototype, then into config
    kb_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'config', 'kpi_knowledge_base.json')
    with open(kb_path, "r") as f:
        return json.load(f)

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

@app.post("/api/analyze")
async def analyze_data(req: AnalyzeRequest):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not found in environment")
        
    try:
        kpi_kb = load_kb()
        llm_layer = LLMStoryteller(api_key, kpi_kb, req.model)
        pipeline = OrchestratorPipeline(kpi_kb)
        connector = DBConnector()
        
        # 1. Fetch Data
        try:
            df = connector.fetch_org_data(req.org_id)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Database Connection Error: {str(e)}")

        # 2. Extract Schema & Run LLM Mapper (RAG)
        db_schema = {col: str(dtype) for col, dtype in df.dtypes.items()}
        try:
            semantic_mapping, retrieved_context = llm_layer.schema_mapper(req.query, db_schema)
            mapped_kpi = semantic_mapping.get("kpi")
            mapped_drivers = semantic_mapping.get("drivers", [])
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"LLM Schema Inference failed: {str(e)}")

        # 3. Run Pipeline Engine
        try:
            evidence_graph = pipeline.run_analysis(df, kpi_name=mapped_kpi, drivers=mapped_drivers)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Machine Learning Engine failed: {str(e)}")

        # 4. Generate Narrative
        if evidence_graph.get('status') == 'anomaly_detected':
            story, telemetry = llm_layer.generate_story(evidence_graph, persona="Executive Team")
        else:
            story = evidence_graph.get('message', 'No anomalies detected.')
            telemetry = {"latency_ms": 0, "tokens": 0, "cost_usd": 0}

        return {
            "evidence_graph": evidence_graph,
            "story": story,
            "telemetry": telemetry,
            "mapped_kpi": mapped_kpi,
            "mapped_drivers": mapped_drivers,
            "retrieved_rag_context": retrieved_context,
            "active_model": llm_layer.active_model,
            "rows_analyzed": len(df)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
