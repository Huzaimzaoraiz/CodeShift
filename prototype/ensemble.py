import pandas as pd
import numpy as np
from typing import Dict, List, Any
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.preprocessing import StandardScaler

class QuantitativeEnsemble:
    def __init__(self, kpi_kb: Dict):
        self.kpi_kb = kpi_kb
        self.scaler = StandardScaler()
        # Set contamination based on how many anomalies we expect
        self.anomaly_model = IsolationForest(contamination=0.05, random_state=42) 
        self.causal_model = RandomForestRegressor(n_estimators=100, random_state=42)
        
    def time_series_anomaly_detection(self, df: pd.DataFrame, kpi_col: str) -> pd.DataFrame:
        df_sorted = df.sort_values('date').copy()
        
        rolling_mean = df_sorted[kpi_col].rolling(window=3, min_periods=1).mean()
        
        X = pd.DataFrame({
            'kpi_value': df_sorted[kpi_col],
            'rolling_mean': rolling_mean,
            'delta': df_sorted[kpi_col] - rolling_mean
        }).fillna(0)
        
        X_scaled = self.scaler.fit_transform(X)
        predictions = self.anomaly_model.fit_predict(X_scaled)
        
        df_sorted['is_anomaly'] = predictions == -1
        df_sorted['drop_percentage'] = (rolling_mean - df_sorted[kpi_col]) / (rolling_mean + 1e-9)
        
        return df_sorted
        
    def causal_driver_analysis(self, df: pd.DataFrame, kpi_col: str, driver_cols: List[str]) -> Dict[str, float]:
        df_clean = df.dropna(subset=[kpi_col] + driver_cols).copy()
        
        if df_clean.empty:
            return {}
            
        X = df_clean[driver_cols]
        y = df_clean[kpi_col]
        
        self.causal_model.fit(X, y)
        importances = self.causal_model.feature_importances_
        driver_importance = dict(zip(driver_cols, importances))
        
        return dict(sorted(driver_importance.items(), key=lambda item: item[1], reverse=True))
        
    def statistical_rules_check(self, drop_percentage: float, kpi_name: str) -> bool:
        threshold = self.kpi_kb['kpis'].get(kpi_name, {}).get('thresholds', {}).get('material_drop_percentage', 0.1)
        return drop_percentage >= threshold

    def run_pipeline(self, df: pd.DataFrame, kpi_name: str) -> Dict[str, Any]:
        df['date'] = pd.to_datetime(df['date'])
        
        # Anomaly Detection
        anomaly_df = self.time_series_anomaly_detection(df, kpi_name)
        
        # Get worst anomaly in the dataset
        anomalies = anomaly_df[anomaly_df['is_anomaly']]
        if anomalies.empty:
            return {"status": "normal", "message": "No material anomalies detected by the Isolation Forest."}
            
        worst_anomaly = anomalies.sort_values('drop_percentage', ascending=False).iloc[0]
        drop_pct = worst_anomaly['drop_percentage']
        
        # Rules Check
        is_material = self.statistical_rules_check(drop_pct, kpi_name)
        if not is_material:
            return {"status": "normal", "message": f"Isolation Forest flagged anomaly, but severity ({drop_pct:.1%}) is below materiality threshold."}
            
        # Causal Driver Analysis
        drivers = self.kpi_kb['kpis'].get(kpi_name, {}).get('drivers', [])
        importances = self.causal_driver_analysis(df, kpi_name, drivers)
        
        evidence_graph = {
            "status": "anomaly_detected",
            "kpi": kpi_name,
            "anomaly_date": worst_anomaly['date'].strftime('%Y-%m-%d'),
            "drop_percentage": float(drop_pct),
            "is_material": bool(is_material),
            "driver_importances": importances,
            "top_driver": list(importances.keys())[0] if importances else None
        }
        
        return evidence_graph
