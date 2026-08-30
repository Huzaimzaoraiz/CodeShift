import time
import json
from typing import Dict, Any, Tuple
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class LLMStoryteller:
    def __init__(self, api_key: str, kpi_kb: Dict, model_name: str):
        self.llm = ChatGroq(
            api_key=api_key,
            model_name=model_name, 
            temperature=0.2
        )
        self.active_model = model_name
        self.kpi_kb = kpi_kb
        
        # Build RAG Vector Store
        self.rag_corpus = []
        self.rag_keys = []
        for kpi, data in self.kpi_kb.items():
            if kpi == 'business_levers': continue
            doc = f"KPI: {kpi}. Description: {data.get('description', '')}. Drivers: {', '.join(data.get('causal_drivers', []))}."
            self.rag_corpus.append(doc)
            self.rag_keys.append(kpi)
            
        self.vectorizer = TfidfVectorizer()
        if self.rag_corpus:
            self.rag_vectors = self.vectorizer.fit_transform(self.rag_corpus)
            
    def retrieve_context(self, query: str) -> Dict[str, Any]:
        """Performs RAG: Semantic search over the knowledge base."""
        if not self.rag_corpus:
            return {}
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.rag_vectors).flatten()
        best_idx = similarities.argmax()
        
        # Return the most relevant KPI context if similarity is > 0
        if similarities[best_idx] > 0.0:
            best_kpi = self.rag_keys[best_idx]
            return {best_kpi: self.kpi_kb[best_kpi]}
        return self.kpi_kb # Fallback to all if no match

    def schema_mapper(self, user_query: str, db_schema: Dict[str, str]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Maps user intent and raw schema to KPI and drivers using LLM + RAG."""
        
        # [RAG STEP] Retrieve context
        retrieved_context = self.retrieve_context(user_query)
        
        prompt = PromptTemplate(
            input_variables=["query", "schema", "kpi_context"],
            template='''
            You are an expert Data Architect.
            User Query: {query}
            
            Retrieved Business Context (RAG):
            {kpi_context}
            
            The database returned the following schema (columns and data types):
            {schema}
            
            Analyze the schema and the user's intent. Determine:
            1. Which column is the target KPI to analyze?
            2. Which columns are the causal drivers (features) that impact this KPI? Do not include the KPI or 'date'/'id' columns.
            
            Output ONLY a valid JSON object in this exact format:
            {{
                "kpi": "column_name",
                "drivers": ["driver1", "driver2"]
            }}
            '''
        )
        
        chain = prompt | self.llm
        response = chain.invoke({
            "query": user_query,
            "schema": json.dumps(db_schema, indent=2),
            "kpi_context": json.dumps(retrieved_context, indent=2)
        })
        
        # Models like DeepSeek-R1 output <think> blocks which ruin JSON parsers
        import re
        content = response.content.strip()
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
        
        # Extract everything from the first { to the last }
        start_idx = content.find('{')
        end_idx = content.rfind('}')
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            content = content[start_idx:end_idx+1]
            
        try:
            mapping = json.loads(content)
        except json.JSONDecodeError:
            raise ValueError(f"LLM failed to return valid JSON. Output: {content}")
            
        return mapping, retrieved_context

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
        lever_data = self.kpi_kb.get('business_levers', {}).get(top_driver, {})

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
