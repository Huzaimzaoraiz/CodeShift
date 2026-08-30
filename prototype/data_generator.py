import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_large_mock_data():
    np.random.seed(42)
    
    # 3 years of data (approx 1095 days)
    start_date = datetime(2021, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(1095)]
    regions = ['North America', 'EMEA']
    products = ['Legacy Product', 'New Product'] 
    
    daily_data = []
    
    for date in dates:
        # Add seasonality (sin wave based on day of year to simulate peak seasons)
        day_of_year = date.timetuple().tm_yday
        seasonality = 1.0 + 0.3 * np.sin(2 * np.pi * day_of_year / 365)
        
        # Add a slow upward trend over the 3 years
        trend = 1.0 + (date - start_date).days / 1095 * 0.5
        
        for region in regions:
            for product in products:
                # Sparse history for New Product (launched exactly 1 year ago)
                if product == 'New Product' and date < datetime(2022, 10, 1):
                    continue
                    
                # Baseline metrics
                base_traffic = 5000 if region == 'North America' else 3000
                conversion_rate = 0.05
                base_price = 100 if product == 'Legacy Product' else 150
                discount = np.random.uniform(0.05, 0.25)
                
                # Apply trend and seasonality
                current_traffic = base_traffic * seasonality * trend
                
                # Introduce the anomaly: A massive drop in NA in Oct 2023
                if datetime(2023, 10, 10) <= date <= datetime(2023, 10, 24) and region == 'North America':
                    current_traffic *= 0.4 # 60% drop in traffic
                    
                # Add heavy random noise to simulate real-world data (harder for ML to fit perfectly)
                daily_traffic = int(np.random.normal(current_traffic, current_traffic * 0.15))
                units_sold = int(daily_traffic * conversion_rate * np.random.normal(1, 0.12))
                revenue = units_sold * base_price * (1 - discount)
                
                daily_data.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'region': region,
                    'product_category': product,
                    'website_traffic': daily_traffic,
                    'discount_percentage': round(discount, 2),
                    'units_sold': units_sold,
                    'revenue': round(revenue, 2)
                })
                
    df_daily = pd.DataFrame(daily_data)
    
    # Weekly Marketing Data (External System)
    weekly_dates = pd.date_range(start='2021-01-04', end='2023-12-31', freq='W-MON')
    
    weekly_data = []
    for date in weekly_dates:
        # Same seasonality and trend
        day_of_year = date.timetuple().tm_yday
        seasonality = 1.0 + 0.3 * np.sin(2 * np.pi * day_of_year / 365)
        trend = 1.0 + (date - start_date).days / 1095 * 0.5
        
        for region in regions:
            base_spend = 15000 if region == 'North America' else 8000
            current_spend = base_spend * seasonality * trend
            
            # Anomaly: Marketing budget slashed in NA in mid-October 2023
            if datetime(2023, 10, 9) <= date <= datetime(2023, 10, 23) and region == 'North America':
                current_spend *= 0.2 # 80% cut in ad spend
                
            spend = int(np.random.normal(current_spend, current_spend * 0.1))
            ad_clicks = int(spend * np.random.uniform(0.8, 1.2))
            
            weekly_data.append({
                'week_start': date.strftime('%Y-%m-%d'),
                'region': region,
                'marketing_spend': spend,
                'ad_clicks': ad_clicks,
                'campaign_status': 'Active' if spend > (base_spend * 0.5) else 'Paused'
            })
            
    df_weekly = pd.DataFrame(weekly_data)
    
    os.makedirs('data', exist_ok=True)
    df_daily.to_csv('data/sales_daily.csv', index=False)
    df_weekly.to_csv('data/marketing_weekly.csv', index=False)
    print(f"Generated large mock dataset: {len(df_daily)} daily rows, {len(df_weekly)} weekly rows.")

if __name__ == '__main__':
    generate_large_mock_data()
