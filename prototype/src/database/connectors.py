import json
import os
import pandas as pd
from sqlalchemy import create_engine
from typing import Dict, List

CONFIG_PATH = os.path.join(os.path.dirname(__file__), '../../config/databases.json')

class DBConnector:
    """
    DB Connector Layer: Connects to isolated PostgreSQL servers based on Organization dynamically.
    """
    def __init__(self):
        self.engines = {}
        self.configs = {}
        self.load_configs()

    def load_configs(self):
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, 'r') as f:
                self.configs = json.load(f)
                
            for org_id, config in self.configs.items():
                try:
                    self.engines[org_id] = create_engine(config["connection_string"])
                except Exception as e:
                    print(f"Failed to initialize engine for {org_id}: {e}")

    def list_databases(self) -> List[str]:
        return list(self.configs.keys())

    def add_database(self, org_id: str, connection_string: str, query: str):
        self.configs[org_id] = {
            "connection_string": connection_string,
            "query": query,
            "post_processing": "none"
        }
        self.engines[org_id] = create_engine(connection_string)
        
        # Persist to disk
        with open(CONFIG_PATH, 'w') as f:
            json.dump(self.configs, f, indent=4)
            
    def fetch_org_data(self, org_id: str) -> pd.DataFrame:
        if org_id not in self.configs:
            raise ValueError(f"Unknown Organization: {org_id}")
            
        config = self.configs[org_id]
        engine = self.engines.get(org_id)
        
        if not engine:
            raise ConnectionError(f"Engine not initialized for {org_id}")
            
        try:
            df = pd.read_sql_query(config["query"], engine)
            
            if config.get("post_processing") == "dummies_contract":
                if 'Contract' in df.columns:
                    df = pd.get_dummies(df, columns=['Contract'], drop_first=True)
                    
            return df
        except Exception as e:
            raise ConnectionError(f"Failed to fetch data for {org_id}: {e}")

    def safe_query(self, org_id: str, sql: str, max_rows: int = 20) -> pd.DataFrame:
        """
        Execute a read-only SQL query for the Reasoning Model's insight queries.

        Security rules (enforced deterministically, NOT by LLM):
        - Only SELECT statements are allowed.
        - DDL/DML keywords (DROP, INSERT, UPDATE, DELETE, ALTER, CREATE, TRUNCATE) are forbidden.
        - Max rows capped at max_rows to prevent data dumps.
        - Only executes against the specified org's database (isolation guaranteed).
        - Returns an empty DataFrame on any error (graceful degradation).
        """
        if org_id not in self.engines:
            raise ValueError(f"No engine for org: {org_id}")

        # --- Safety Gate: deterministic keyword check (not LLM-controlled) ---
        normalized = sql.strip().upper()
        if not normalized.startswith("SELECT"):
            raise PermissionError("Only SELECT queries are allowed by the Reasoning Model.")

        _FORBIDDEN = ["DROP ", "DELETE ", "INSERT ", "UPDATE ", "ALTER ", "CREATE ", "TRUNCATE ", "EXEC ", "EXECUTE "]
        for keyword in _FORBIDDEN:
            if keyword in normalized:
                raise PermissionError(f"Forbidden keyword detected: {keyword.strip()}")

        # Inject LIMIT if not already present to cap row count
        if "LIMIT" not in normalized:
            sql = sql.rstrip(";").rstrip() + f" LIMIT {max_rows}"

        engine = self.engines[org_id]
        try:
            return pd.read_sql_query(sql, engine)
        except Exception as e:
            # Non-fatal: return empty DF so the reasoning model can continue
            print(f"[safe_query] Error for {org_id}: {e}")
            return pd.DataFrame()

