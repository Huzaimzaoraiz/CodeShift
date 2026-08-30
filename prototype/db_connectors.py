import pandas as pd
from sqlalchemy import create_engine

class DBConnector:
    """
    Simulates the DB Connector Layer from the architecture diagram.
    Connects to physically separated PostgreSQL Docker containers based on Organization.
    """
    
    def __init__(self):
        # Postgres connection strings mapped to Docker ports
        self.engine_a = create_engine('postgresql://admin:password@localhost:5432/org_a_taxi')
        self.engine_b = create_engine('postgresql://admin:password@localhost:5433/org_b_airlines')
        
    def fetch_org_data(self, org_id: str) -> pd.DataFrame:
        if org_id == "Org A (Taxi)":
            query = """
            SELECT 
                date,
                SUM(total) as revenue,
                SUM(distance) as total_distance,
                SUM(passengers) as total_passengers,
                AVG(tip) as average_tip
            FROM taxi_trips
            WHERE date IS NOT NULL
            GROUP BY date
            ORDER BY date
            """
            try:
                df = pd.read_sql_query(query, self.engine_a)
                return df
            except Exception as e:
                raise ConnectionError(f"Failed to connect to Org A PostgreSQL Server: {e}")
            
        elif org_id == "Org B (Airlines)":
            query = """
            SELECT 
                date,
                passengers as total_passengers,
                year,
                month_num
            FROM flight_records
            WHERE date IS NOT NULL
            ORDER BY date
            """
            try:
                df = pd.read_sql_query(query, self.engine_b)
                return df
            except Exception as e:
                raise ConnectionError(f"Failed to connect to Org B PostgreSQL Server: {e}")
            
        else:
            raise ValueError("Unknown Organization")
