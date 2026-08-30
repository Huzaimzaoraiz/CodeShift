import streamlit as st
import json
import os
import sys

# Ensure src/ is in the python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.orchestration.pipeline import OrchestratorPipeline
from src.llm.storyteller import LLMStoryteller
from src.database.connectors import DBConnector

st.set_page_config(page_title="BusinessIntelligence.ai", layout="wide")

st.sidebar.title("⚙️ Control Panel")
api_key = st.sidebar.text_input("Groq API Key", type="password")
if api_key:
    os.environ["GROQ_API_KEY"] = api_key

st.sidebar.markdown("---")
org_id = st.sidebar.selectbox("🏢 Select Organization", ["Org A (Telecommunications)", "Org B (Supply Chain)"])

@st.cache_data
def load_kb():
    # Load from the new config/ directory
    kb_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config', 'kpi_knowledge_base.json')
    with open(kb_path, "r") as f:
        return json.load(f)

try:
    kpi_kb = load_kb()
except FileNotFoundError:
    st.error("KPI Knowledge Base not found. Please check config/ directory.")
    st.stop()

st.title("📈 BusinessIntelligence.ai")
st.markdown("### 🔍 Enterprise AI Analyst")

query = st.text_input("What would you like to analyze?", "Analyze our core business health and find anomalies.")

if st.button("Run Pipeline"):
    if not os.environ.get("GROQ_API_KEY"):
        st.error("Please enter a Groq API Key.")
        st.stop()
        
    llm_layer = LLMStoryteller(os.environ["GROQ_API_KEY"], kpi_kb)
    pipeline = OrchestratorPipeline(kpi_kb)
    connector = DBConnector()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### ⚙️ Pipeline Execution")
        with st.status("Running Engine...", expanded=True) as status:
            try:
                df = connector.fetch_org_data(org_id)
                st.write(f"✅ Data fetched ({len(df)} rows).")
            except Exception as e:
                st.error(str(e))
                st.stop()
                
            st.write("🤖 **Schema Mapper (LLM)** interpreting intent & database schema...")
            
            # Extract database schema dynamically
            db_schema = {col: str(dtype) for col, dtype in df.dtypes.items()}
            
            # Use the LLM to dynamically generate the KPI and drivers!
            try:
                semantic_mapping = llm_layer.schema_mapper(query, db_schema)
                mapped_kpi = semantic_mapping.get("kpi")
                mapped_drivers = semantic_mapping.get("drivers", [])
                st.write(f"✅ LLM Generated Target KPI: `{mapped_kpi}`")
                st.write(f"✅ LLM Discovered Drivers: `{mapped_drivers}`")
            except Exception as e:
                st.error(f"LLM Schema Inference failed: {e}")
                st.stop()
                
            st.write("🧠 **Orchestrator Pipeline** running Machine Learning...")
            evidence_graph = pipeline.run_analysis(df, kpi_name=mapped_kpi, drivers=mapped_drivers)
            
            status.update(label="Complete", state="complete")
            
        with st.expander("Evidence Graph (JSON)"):
            st.json(evidence_graph)
            
    with col2:
        st.markdown(f"#### 📄 Narrative")
        if evidence_graph.get('status') != 'anomaly_detected':
            st.info(evidence_graph.get('message', "No anomalies."))
        else:
            with st.spinner("Story Generator drafting..."):
                story, telemetry = llm_layer.generate_story(evidence_graph, persona="Executive Team")
            st.markdown(story)
            st.code(f"Latency: {telemetry.get('latency_ms')} ms | Cost: ${telemetry.get('cost_usd', 0):.5f}")
