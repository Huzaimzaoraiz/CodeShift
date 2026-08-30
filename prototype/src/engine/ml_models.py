import pandas as pd
from typing import List, Dict
from sklearn.ensemble import IsolationForest, RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler

class MLEngine:
    def __init__(self):
        self.scaler = StandardScaler()
        self.anomaly_model = IsolationForest(contamination=0.05, random_state=42) 
        self.causal_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        self.causal_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        
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
        
    def causal_driver_analysis(self, df: pd.DataFrame, kpi_col: str, driver_cols: List[str], is_classification: bool = False) -> Dict[str, float]:
        available_drivers = [col for col in driver_cols if col in df.columns]
        df_clean = df.dropna(subset=[kpi_col] + available_drivers).copy()
        
        if df_clean.empty or not available_drivers:
            return {}
            
        X = df_clean[available_drivers]
        y = df_clean[kpi_col]
        
        model = self.causal_classifier if is_classification else self.causal_regressor
        model.fit(X, y)
        
        importances = model.feature_importances_
        driver_importance = dict(zip(available_drivers, importances))
        
        return dict(sorted(driver_importance.items(), key=lambda item: item[1], reverse=True))
