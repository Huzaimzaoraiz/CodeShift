import pandas as pd
from sqlalchemy import create_engine

class DBConnector:
    """
    DB Connector Layer: Connects to isolated PostgreSQL servers based on Organization.
    """
    def __init__(self):
        self.engine_a = create_engine('postgresql://admin:password@localhost:5432/org_a_telco')
        self.engine_b = create_engine('postgresql://admin:password@localhost:5433/org_b_supply')
        
    def fetch_org_data(self, org_id: str) -> pd.DataFrame:
        if org_id == "Org A (Telecommunications)":
            query = """
            SELECT 
                "MonthlyCharges",
                "TotalCharges",
                "tenure" as tenure_months,
                "Churn" as churn_status,
                "Contract"
            FROM customer_churn
            WHERE "TotalCharges" IS NOT NULL
            """
            try:
                df = pd.read_sql_query(query, self.engine_a)
                df = pd.get_dummies(df, columns=['Contract'], drop_first=True)
                return df
            except Exception as e:
                raise ConnectionError(f"Failed to connect to Org A (Telco) Server: {e}")
            
        elif org_id == "Org B (Supply Chain)":
            query = """
            SELECT 
                date,
                SUM(quantity) as total_quantity,
                AVG("priceMin") as avg_price_min,
                AVG("priceMax") as avg_price_max,
                AVG("priceMod") as modal_price
            FROM wholesale_supply
            WHERE date IS NOT NULL
            GROUP BY date
            ORDER BY date
            """
            try:
                df = pd.read_sql_query(query, self.engine_b)
                return df
            except Exception as e:
                raise ConnectionError(f"Failed to connect to Org B (Supply Chain) Server: {e}")
            
        else:
            raise ValueError("Unknown Organization")
