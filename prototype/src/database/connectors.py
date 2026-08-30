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
