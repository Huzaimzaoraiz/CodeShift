import pandas as pd
from sqlalchemy import create_engine
import time

def load_data():
    url = "https://raw.githubusercontent.com/yajasarora/Superstore-Sales-Analysis-with-Tableau/master/Sample%20-%20Superstore.csv"
    print(f"Downloading data from {url}...")
    try:
        df = pd.read_csv(url, encoding='latin1')
    except Exception as e:
        # Fallback to another popular repo if that one is gone
        print("Fallback URL...")
        url = "https://raw.githubusercontent.com/praveen-kumar-maurya/Superstore-Sales-Dashboard-Power-BI/main/Sample%20-%20Superstore.csv"
        df = pd.read_csv(url, encoding='latin1')
        
    print(f"Loaded {len(df)} rows. Connecting to Postgres...")
    
    # Wait for DB to be ready
    time.sleep(3)
    
    engine = create_engine('postgresql://admin:password@localhost:5435/org_c_superstore')
    
    # Clean column names for easier querying
    df.columns = [c.replace(' ', '_').replace('-', '_') for c in df.columns]
    
    # Push to Postgres
    df.to_sql('superstore_sales', engine, if_exists='replace', index=False)
    print("Successfully seeded org_c_superstore database!")

if __name__ == "__main__":
    load_data()
