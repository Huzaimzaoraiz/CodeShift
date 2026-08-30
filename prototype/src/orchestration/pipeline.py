import pandas as pd
from typing import Dict, Any
from src.engine.ml_models import MLEngine
from src.engine.rules import StatisticalRules

class OrchestratorPipeline:
    def __init__(self):
        self.ml_engine = MLEngine()
        self.rules = StatisticalRules()
        
    def run_analysis(self, df: pd.DataFrame, kpi_name: str, drivers: list) -> Dict[str, Any]:
        
        if 'date' in df.columns:
            # Time-Series Pipeline (Supply Chain)
            df['date'] = pd.to_datetime(df['date'])
            anomaly_df = self.ml_engine.time_series_anomaly_detection(df, kpi_name)
            
            anomalies = anomaly_df[anomaly_df['is_anomaly']]
            if anomalies.empty:
                return {"status": "normal", "message": "No material anomalies detected by the Isolation Forest."}
                
            worst_anomaly = anomalies.sort_values('drop_percentage', ascending=False).iloc[0]
            drop_pct = worst_anomaly['drop_percentage']
            
            is_material = self.rules.check_materiality(drop_pct, kpi_name)
            if not is_material:
                return {"status": "normal", "message": f"Isolation Forest flagged anomaly, but severity ({drop_pct:.1%}) is below materiality threshold."}
                
            importances = self.ml_engine.causal_driver_analysis(df, kpi_name, drivers, is_classification=False)
            
            return {
                "status": "anomaly_detected",
                "kpi": kpi_name,
                "anomaly_date": worst_anomaly['date'].strftime('%Y-%m-%d'),
                "drop_percentage": float(drop_pct),
                "is_material": bool(is_material),
                "driver_importances": importances,
                "top_driver": list(importances.keys())[0] if importances else None
            }
            
        else:
            # Cross-Sectional Pipeline (IBM Telco Churn)
            churn_rate = df[kpi_name].mean()
            importances = self.ml_engine.causal_driver_analysis(df, kpi_name, drivers, is_classification=True)
            
            return {
                "status": "anomaly_detected",
                "kpi": "High Customer Churn Rate",
                "anomaly_date": "Current Snapshot",
                "drop_percentage": float(churn_rate),
                "is_material": churn_rate > 0.15,
                "driver_importances": importances,
                "top_driver": list(importances.keys())[0] if importances else None
            }
