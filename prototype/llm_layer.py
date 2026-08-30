import os
import json
import time
from typing import Dict, Any, Tuple
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

class LLMStoryteller:
    def __init__(self, api_key: str, kpi_kb: Dict):
        self.llm = ChatGroq(
            groq_api_key=api_key,
            model_name="llama3-8b-8192", # Fast and capable
            temperature=0.2
        )
        self.kpi_kb = kpi_kb
        
    def schema_mapper(self, user_query: str) -> str:
        """
        Simulates the Schema Mapper (LLM) determining which KPI to query.
        """
        prompt = f"""
        You are a Data Schema Mapper for an enterprise database.
        Available KPIs in our Knowledge Base: {list(self.kpi_kb['kpis'].keys())}
        
        User Query: "{user_query}"
        
        Which exact KPI is the user asking about? Return ONLY the exact KPI name as a string, with no other text.
        """
        response = self.llm.invoke(prompt)
        return response.content.strip()

    def confidence_scorer(self, evidence_graph: Dict[str, Any]) -> Tuple[bool, float]:
        """
        Checks confidence before calling the LLM Story Generator.
        Returns (is_confident, confidence_score)
        """
        if evidence_graph['status'] != 'anomaly_detected':
            return False, 0.0
            
        importances = evidence_graph.get('driver_importances', {})
        if not importances:
            return False, 0.0
            
        # Top driver feature importance
        top_importance = list(importances.values())[0]
        
        # If the highest feature importance is less than 0.2, the model isn't confident in any single driver
        if top_importance < 0.2:
            return False, top_importance
            
        return True, top_importance

    def generate_story(self, evidence_graph: Dict[str, Any], persona: str) -> Tuple[str, Dict]:
        """
        Persona-Aware Story Generator.
        Also tracks telemetry (latency, tokens - simulated).
        """
        start_time = time.time()
        
        is_confident, conf_score = self.confidence_scorer(evidence_graph)
        
        if not is_confident:
            telemetry = {"latency_ms": int((time.time() - start_time) * 1000), "status": "abstained"}
            return (
                "🚨 **ABSTENTION / CLARIFICATION REQUEST**\n\n"
                f"I detected a drop of {evidence_graph.get('drop_percentage', 0):.1%} in {evidence_graph.get('kpi')}, "
                f"but the statistical confidence in the root cause is too low (Random Forest feature importance: {conf_score:.2f}). "
                "I will not generate a hallucinated explanation. Please request a deeper manual analysis by the Data Science team.",
                telemetry
            )
            
        # We are confident, generate story
        top_driver = evidence_graph['top_driver']
        lever_info = self.kpi_kb.get('business_levers', {}).get(top_driver, {})
        
        prompt = f"""
        You are a Business Intelligence AI.
        Write a concise, data-backed narrative explaining a KPI movement to a {persona}.
        
        EVIDENCE GRAPH (DO NOT INVENT NUMBERS, ONLY USE THESE):
        - KPI: {evidence_graph['kpi']}
        - Drop: {evidence_graph['drop_percentage']:.1%}
        - Date: {evidence_graph['anomaly_date']}
        - Region: {evidence_graph['region_filtered'] or 'Global'}
        - Top Causal Driver: {top_driver}
        
        RECOMMENDED ACTION (Based on Knowledge Base):
        - Lever: {lever_info.get('lever', 'N/A')}
        - Action: {lever_info.get('action', 'Investigate further')}
        - Owner: {lever_info.get('owner', 'Operations')}
        - Expected Impact: {lever_info.get('expected_impact', 'Unknown')}
        
        Format your response nicely with markdown headers:
        ### What Changed
        ### Contributing Factors
        ### Recommended Actions (Levers & Impact)
        
        Tailor the tone specifically for a {persona}.
        """
        
        response = self.llm.invoke(prompt)
        end_time = time.time()
        
        # Simulate token tracking (Groq API returns token usage in response.response_metadata)
        tokens = response.response_metadata.get('token_usage', {}).get('total_tokens', 150)
        cost = tokens * 0.00005 # rough estimate
        
        telemetry = {
            "latency_ms": int((end_time - start_time) * 1000),
            "tokens": tokens,
            "cost_usd": cost,
            "status": "success"
        }
        
        return response.content, telemetry
