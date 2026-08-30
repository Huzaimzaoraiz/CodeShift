import json
from typing import Dict, Any, Tuple, TypedDict
# pyrefly: ignore [missing-import]
from langchain_groq import ChatGroq
# pyrefly: ignore [missing-import]
from langchain_core.prompts import PromptTemplate
# pyrefly: ignore [missing-import]
from langgraph.graph import StateGraph, START, END

class ReasoningState(TypedDict):
    evidence_graph: Dict[str, Any]
    persona: str
    conf_score: float
    
    # Internal CoT
    reasoning_trace: str
    
    # Final Outputs
    change_detection: str
    contributing_factors: str
    recommended_actions: str
    
    # Telemetry
    total_tokens: int
    total_cost: float

class ReasoningModel:
    def __init__(self, api_key: str, model_name: str):
        self.llm = ChatGroq(
            api_key=api_key,
            model_name=model_name, 
            temperature=0.4 # slightly higher for better reasoning
        )
        self.json_llm = ChatGroq(
            api_key=api_key,
            model_name=model_name, 
            temperature=0.1 # strict for json
        )
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(ReasoningState)
        
        workflow.add_node("analyze_evidence", self.analyze_evidence_node)
        workflow.add_node("format_output", self.format_output_node)
        
        workflow.add_edge(START, "analyze_evidence")
        workflow.add_edge("analyze_evidence", "format_output")
        workflow.add_edge("format_output", END)
        
        return workflow.compile()

    def analyze_evidence_node(self, state: ReasoningState):
        ev = state["evidence_graph"]
        prompt = f"""
        You are an elite Data Scientist and Business Strategist.
        You must perform a deep Chain-of-Thought reasoning based on this data:
        
        KPI: {ev.get('kpi', 'Unknown')}
        Anomaly Date: {ev.get('anomaly_date', 'Unknown')}
        Severity: {ev.get('drop_percentage', 0):.1%} drop
        Root Cause Feature (ML Ranked): {ev.get('top_driver', 'Unknown')} (Score: {state['conf_score']:.2f})
        
        Write a detailed, step-by-step internal monologue analyzing:
        1. Why this severity matters.
        2. How the root cause mathematically connects to the KPI.
        3. Brainstorm and propose an optimal, data-driven recommended action to address this root cause.
        Do not output JSON. Just output your raw reasoning.
        """
        response = self.llm.invoke(prompt)
        
        try:
            tokens = response.response_metadata['token_usage']['total_tokens']
            cost = tokens * 0.0000001
        except:
            tokens = len(response.content.split()) * 1.3
            cost = 0.0001
            
        return {
            "reasoning_trace": response.content,
            "total_tokens": state.get("total_tokens", 0) + int(tokens),
            "total_cost": state.get("total_cost", 0.0) + cost
        }

    def format_output_node(self, state: ReasoningState):
        prompt = f"""
        You are an Enterprise AI BI Analyst speaking to a {state['persona']}.
        
        Based on the following internal reasoning trace:
        ---
        {state['reasoning_trace']}
        ---
        
        Synthesize the final narrative.
        Output ONLY a valid JSON object in this exact format, with NO markdown formatting around it:
        {{
            "change_detection": "A concise summary of the anomaly finding, tailored to a {state['persona']}.",
            "contributing_factors": "Explain the mathematical root cause from the trace.",
            "recommended_actions": "Prescribe the action based on the evidence."
        }}
        """
        response = self.json_llm.invoke(prompt)
        
        try:
            tokens = response.response_metadata['token_usage']['total_tokens']
            cost = tokens * 0.0000001
        except:
            tokens = len(response.content.split()) * 1.3
            cost = 0.0001
            
        content = response.content.strip()
        import re
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
        start_idx = content.find('{')
        end_idx = content.rfind('}')
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            content = content[start_idx:end_idx+1]
            
        try:
            mapping = json.loads(content)
        except json.JSONDecodeError:
            mapping = {
                "change_detection": "Anomaly Detected.",
                "contributing_factors": f"Error parsing factors. Trace was: {state['reasoning_trace'][:100]}...",
                "recommended_actions": "N/A"
            }
            
        return {
            "change_detection": mapping.get("change_detection", "N/A"),
            "contributing_factors": mapping.get("contributing_factors", "N/A"),
            "recommended_actions": mapping.get("recommended_actions", "N/A"),
            "total_tokens": state.get("total_tokens", 0) + int(tokens),
            "total_cost": state.get("total_cost", 0.0) + cost
        }

    def confidence_scorer(self, evidence_graph: Dict[str, Any]) -> Tuple[bool, float]:
        if evidence_graph.get('status') != 'anomaly_detected':
            return False, 0.0
            
        importances = evidence_graph.get('driver_importances', {})
        if not importances:
            return False, 0.0
            
        top_importance = list(importances.values())[0]
        
        if top_importance < 0.2:
            return False, top_importance
            
        return True, top_importance

    def generate_story(self, evidence_graph: Dict[str, Any], persona: str = "Executive Team") -> Tuple[str, str, str, str, Dict[str, Any]]:
        is_confident, conf_score = self.confidence_scorer(evidence_graph)
        
        telemetry_raw = {"tokens": 0, "cost_usd": 0.0}
        
        if not is_confident:
            return (
                "🚨 ABSTENTION / CLARIFICATION REQUEST",
                f"I detected a drop of {evidence_graph.get('drop_percentage', 0):.1%} in {evidence_graph.get('kpi', 'Unknown KPI')}.",
                f"Statistical confidence in root cause is too low (Score: {conf_score:.2f}).",
                "Please request a deeper manual analysis by the Data Science team.",
                telemetry_raw
            )

        top_driver = evidence_graph.get('top_driver', 'Unknown Driver')

        initial_state = {
            "evidence_graph": evidence_graph,
            "persona": persona,
            "conf_score": conf_score,
            "total_tokens": 0,
            "total_cost": 0.0
        }
        
        print(f"--- Starting LangGraph Reasoning Run for {persona} ---")
        
        # Run the LangGraph
        final_state = self.graph.invoke(initial_state)
        
        print("\n--- Internal Reasoning Trace (LangGraph) ---")
        print(final_state["reasoning_trace"])
        print("--------------------------------------------\n")
        
        telemetry_raw["tokens"] = final_state["total_tokens"]
        telemetry_raw["cost_usd"] = final_state["total_cost"]
            
        return (
            "Complete",
            final_state.get("change_detection", "N/A"),
            final_state.get("contributing_factors", "N/A"),
            final_state.get("recommended_actions", "N/A"),
            telemetry_raw
        )
