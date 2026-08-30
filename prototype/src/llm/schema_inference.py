import time
import json
import re
from typing import Dict, Any, Tuple
# pyrefly: ignore [missing-import]
from langchain_groq import ChatGroq
# pyrefly: ignore [missing-import]
from langchain_core.prompts import PromptTemplate

class SchemaInferenceLLM:
    def __init__(self, api_key: str, model_name: str):
        self.llm = ChatGroq(
            api_key=api_key,
            model_name=model_name, 
            temperature=0.2
        )
        self.active_model = model_name

    def schema_mapper(self, user_query: str, db_schema: Dict[str, str]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Maps user intent and raw schema to KPI and drivers dynamically using Zero-Shot LLM reasoning."""
        
        prompt = PromptTemplate(
            input_variables=["query", "schema"],
            template='''
            You are an expert Data Architect and Business Intelligence Analyst.
            User Query: {query}
            
            The database returned the following schema (columns and data types):
            {schema}
            
            Analyze the schema and the user's intent. Determine:
            1. Which column represents the primary KPI the user wants to analyze?
            2. Which columns are the causal drivers (features) that impact this KPI? Do not include the KPI itself or any 'date', 'id', or purely structural columns.
            
            Output ONLY a valid JSON object in this exact format:
            {{
                "kpi": "column_name",
                "drivers": ["driver1", "driver2", "driver3"]
            }}
            '''
        )
        
        chain = prompt | self.llm
        response = chain.invoke({
            "query": user_query,
            "schema": json.dumps(db_schema, indent=2)
        })
        
        content = response.content.strip()
        # Remove thinking blocks from models like DeepSeek-R1
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
            
        # Return empty dict for retrieved context since we no longer use RAG
        return mapping, {}
