import time
import json
from typing import Dict, Any, Tuple
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

class LLMStoryteller:
    def __init__(self, api_key: str, kpi_kb: Dict):
        self.llm = ChatGroq(
            api_key=api_key,
            model_name="llama3-8b-8192", 
            temperature=0.2
        )
        self.kpi_kb = kpi_kb
        
    def schema_mapper(self, user_query: str, db_schema: Dict[str, str]) -> Dict[str, Any]:
        """
        Dynamically generates the KPI and drivers by inspecting the database schema.
        """
        prompt = f"""
        You are a highly intelligent Data Architect AI.
        A user has asked: "{user_query}"
        
        The database returned the following schema (columns and data types):
        {json.dumps(db_schema, indent=2)}
        
        Analyze the schema and the user's intent. Determine:
        1. Which column is the target KPI to analyze?
        2. Which columns are the causal drivers (features) that impact this KPI? Do not include the KPI or 'date'/'id' columns.
        
        Output ONLY a valid JSON object in this exact format:
        {{
            "kpi": "column_name",
            "drivers": ["driver1", "driver2"]
        }}
        """
        response = self.llm.invoke(prompt)
        
        # Clean the response to parse JSON
        content = response.content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].strip()
            
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            raise ValueError(f"LLM failed to return valid JSON. Output: {content}")

    def confidence_scorer(self, evidence_graph: Dict[str, Any]) -> Tuple[bool, float]:
        if evidence_graph['status'] != 'anomaly_detected':
            return False, 0.0
            
        importances = evidence_graph.get('driver_importances', {})
        if not importances:
            return False, 0.0
            
        top_importance = list(importances.values())[0]
        
        if top_importance < 0.2:
            return False, top_importance
            
        return True, top_importance

    def generate_story(self, evidence_graph: Dict[str, Any], persona: str) -> Tuple[str, Dict]:
        start_time = time.time()
        
        is_confident, conf_score = self.confidence_scorer(evidence_graph)
        
        telemetry = {
            "latency_ms": 0,
            "tokens": 0,
            "cost_usd": 0
        }
        
        if not is_confident:
            return (
                "🚨 **ABSTENTION / CLARIFICATION REQUEST**\n\n"
                f"I detected a drop of {evidence_graph.get('drop_percentage', 0):.1%} in {evidence_graph.get('kpi')}, "
                f"but the statistical confidence in the root cause is too low (Random Forest feature importance: {conf_score:.2f}). "
                "I will not generate a hallucinated explanation. Please request a deeper manual analysis by the Data Science team.",
                telemetry
            )

        top_driver = evidence_graph['top_driver']
        lever_data = self.kpi_kb['business_levers'].get(top_driver, {})

        template = """
        You are an Enterprise AI BI Analyst speaking to a {persona}.
        
        Evidence Graph:
        - KPI: {kpi}
        - Anomaly Date: {anomaly_date}
        - Severity: {drop_percentage} drop
        - Root Cause (ML Ranked): {top_driver} (Importance Score: {conf_score})
        - Recommended Action: {action}
        - Expected Impact: {impact}
        
        Write a concise, professional executive summary. 
        Start with the finding, explain the mathematical root cause, and prescribe the action.
        """
        prompt = PromptTemplate(
            input_variables=["persona", "kpi", "anomaly_date", "drop_percentage", "top_driver", "conf_score", "action", "impact"],
            template=template
        )
        
        formatted_prompt = prompt.format(
            persona=persona,
            kpi=evidence_graph['kpi'],
            anomaly_date=evidence_graph['anomaly_date'],
            drop_percentage=f"{evidence_graph['drop_percentage']:.1%}",
            top_driver=top_driver,
            conf_score=f"{conf_score:.2f}",
            action=lever_data.get('action', 'N/A'),
            impact=lever_data.get('expected_impact', 'N/A')
        )
        
        response = self.llm.invoke(formatted_prompt)
        
        end_time = time.time()
        
        try:
            tokens = response.response_metadata['token_usage']['total_tokens']
            cost = tokens * 0.0000001
        except:
            tokens = len(response.content.split()) * 1.3
            cost = 0.0001
            
        telemetry["latency_ms"] = int((end_time - start_time) * 1000)
        telemetry["tokens"] = int(tokens)
        telemetry["cost_usd"] = cost

        return response.content, telemetry
