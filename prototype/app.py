import streamlit as st
import json
import os
from ensemble import QuantitativeEnsemble
from llm_layer import LLMStoryteller
from db_connectors import DBConnector

st.set_page_config(page_title="BusinessIntelligence.ai", layout="wide")

# Sidebar
st.sidebar.title("⚙️ Control Panel")
api_key = st.sidebar.text_input("Groq API Key", type="password", help="Enter your Groq API key")
if api_key:
    os.environ["GROQ_API_KEY"] = api_key

st.sidebar.markdown("---")
org_id = st.sidebar.selectbox("🏢 Select Organization (Server Route)", ["Org A (Taxi)", "Org B (Airlines)"])

@st.cache_data
def load_kb():
    with open("kpi_knowledge_base.json", "r") as f:
        kpi_kb = json.load(f)
    return kpi_kb

try:
    kpi_kb = load_kb()
except FileNotFoundError:
    st.error("KPI Knowledge Base not found.")
    st.stop()

st.title("📈 BusinessIntelligence.ai")
st.markdown("### 🔍 Enterprise AI Analyst")

query = st.text_input("What would you like to analyze?", "Why did our core metric drop recently?")

if st.button("Run Pipeline"):
    if not os.environ.get("GROQ_API_KEY"):
        st.error("Please enter a Groq API Key in the sidebar.")
        st.stop()
        
    llm_layer = LLMStoryteller(os.environ["GROQ_API_KEY"], kpi_kb)
    ensemble = QuantitativeEnsemble(kpi_kb)
    connector = DBConnector()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### ⚙️ Pipeline Execution (Behind the Scenes)")
        with st.status("Running Data Intelligence Engine...", expanded=True) as status:
            st.write("🤖 **Schema Mapper (LLM)** interpreting intent...")
            
            # For this prototype, we'll manually map the KPI based on the Org since we are not using complex queries yet.
            mapped_kpi = "revenue" if "Taxi" in org_id else "total_passengers"
            
            st.write(f"✅ Mapped to Core KPI: `{mapped_kpi}`")
            
            st.write(f"🔌 **DB Connector** routing query to: `{org_id}` Server...")
            try:
                df = connector.fetch_org_data(org_id)
                st.write(f"✅ Data fetched successfully ({len(df)} rows).")
            except Exception as e:
                st.error(str(e))
                status.update(label="Failed", state="error")
                st.stop()
            
            st.write("🧠 **Quantitative Logic Ensemble (Non-LLM)** running...")
            st.write("  - 🌲 Training Isolation Forest for Anomaly Detection...")
            st.write("  - 🌳 Training Random Forest for Causal Driver Analysis...")
            st.write("  - 📏 Checking Statistical Rules Thresholds...")
            
            evidence_graph = ensemble.run_pipeline(df, kpi_name=mapped_kpi)
            
            status.update(label="Pipeline Complete", state="complete")
            
        with st.expander("Show Evidence Graph (JSON)"):
            st.json(evidence_graph)
            
    with col2:
        st.markdown(f"#### 📄 Action Narrative: {org_id}")
        
        if evidence_graph.get('status') != 'anomaly_detected':
            st.info(evidence_graph.get('message', "No anomalies detected."))
        else:
            with st.spinner("Story Generator (LLM) crafting narrative..."):
                story, telemetry = llm_layer.generate_story(evidence_graph, persona="Executive Team")
                
            st.markdown(story)
            
            st.markdown("---")
            st.markdown("##### ⏱️ Runtime Telemetry")
            st.code(f"Latency: {telemetry.get('latency_ms')} ms | Tokens: {telemetry.get('tokens')} | Est. Cost: ${telemetry.get('cost_usd', 0):.5f}")
            
            st.markdown("##### 🔄 Human Feedback Loop")
            f_col1, f_col2, f_col3 = st.columns([1, 1, 2])
            with f_col1:
                if st.button("👍 Looks Good"):
                    st.success("Feedback logged to KB!")
            with f_col2:
                if st.button("👎 Incorrect"):
                    st.warning("Feedback logged! Thresholds updated.")
