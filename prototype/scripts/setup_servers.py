import pandas as pd
from sqlalchemy import create_engine
import time

def setup_servers():
    print("Waiting for PostgreSQL Docker containers to initialize...")
    time.sleep(5) 
    
    # 1. Org A: Telecommunications (IBM Telco Customer Churn)
    print("Downloading Org A (Telco) Enterprise dataset...")
    url_telco = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    try:
        df_telco = pd.read_csv(url_telco)
        df_telco['TotalCharges'] = pd.to_numeric(df_telco['TotalCharges'].replace(' ', ''), errors='coerce')
        df_telco['Churn'] = df_telco['Churn'].map({'Yes': 1, 'No': 0})
        
        print("Pushing data to Org A PostgreSQL Server...")
        engine_a = create_engine('postgresql://admin:password@localhost:5432/org_a_telco')
        df_telco.to_sql('customer_churn', engine_a, if_exists='replace', index=False)
        print("✅ Org A PostgreSQL database populated successfully.")
    except Exception as e:
        print(f"Error setting up Org A: {e}")

    # 2. Org B: Supply Chain (Wholesale Market Arrivals)
    print("Downloading Org B (Supply Chain) Enterprise dataset...")
    url_supply = "https://raw.githubusercontent.com/selva86/datasets/master/MarketArrivals.csv"
    try:
        df_supply = pd.read_csv(url_supply)
        df_supply['date'] = pd.to_datetime(df_supply['date'])
        
        print("Pushing data to Org B PostgreSQL Server...")
        engine_b = create_engine('postgresql://admin:password@localhost:5433/org_b_supply')
        df_supply.to_sql('wholesale_supply', engine_b, if_exists='replace', index=False)
        print("✅ Org B PostgreSQL database populated successfully.")
    except Exception as e:
        print(f"Error setting up Org B: {e}")

if __name__ == "__main__":
    setup_servers()
