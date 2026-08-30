import pandas as pd
from sqlalchemy import create_engine
import time

def setup_servers():
    print("Waiting for PostgreSQL Docker containers to initialize...")
    time.sleep(5) # Give docker containers a few seconds to boot up
    
    # 1. Org A: Taxi Company (Real NYC Taxi Data) -> postgres_org_a (Port 5432)
    print("Downloading Org A (Taxi) dataset...")
    url_taxi = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/taxis.csv"
    try:
        df_taxi = pd.read_csv(url_taxi)
        df_taxi['pickup'] = pd.to_datetime(df_taxi['pickup'])
        df_taxi['date'] = df_taxi['pickup'].dt.date
        
        print("Pushing data to Org A PostgreSQL Server...")
        engine_a = create_engine('postgresql://admin:password@localhost:5432/org_a_taxi')
        df_taxi.to_sql('taxi_trips', engine_a, if_exists='replace', index=False)
        print("✅ Org A PostgreSQL database populated successfully.")
    except Exception as e:
        print(f"Error setting up Org A: {e}")

    # 2. Org B: Airlines (Real Flight Data) -> postgres_org_b (Port 5433)
    print("Downloading Org B (Airlines) dataset...")
    url_flights = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/flights.csv"
    try:
        df_flights = pd.read_csv(url_flights)
        month_map = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 
                     'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
        df_flights['month_num'] = df_flights['month'].map(month_map)
        df_flights['date'] = pd.to_datetime(df_flights['year'].astype(str) + '-' + df_flights['month_num'].astype(str) + '-01')
        
        print("Pushing data to Org B PostgreSQL Server...")
        engine_b = create_engine('postgresql://admin:password@localhost:5433/org_b_airlines')
        df_flights.to_sql('flight_records', engine_b, if_exists='replace', index=False)
        print("✅ Org B PostgreSQL database populated successfully.")
    except Exception as e:
        print(f"Error setting up Org B: {e}")

if __name__ == "__main__":
    setup_servers()
