import pandas as pd
from typing import Dict, Any, Optional, List
from src.engine.ml_models import (
    MLEngine, score_kpi_quality, decompose_series, compute_anomaly_confidence
)
from src.engine.rules import StatisticalRules
from src.kpi.models import EnhancedKPI, ComputabilityResult


class OrchestratorPipeline:
    def __init__(self):
        self.ml_engine = MLEngine()
        self.rules = StatisticalRules()

    def run_analysis(
        self,
        df: pd.DataFrame,
        kpi_name: str,
        drivers: list,
        enhanced_kpi: Optional[EnhancedKPI] = None,
        kpi_series: Optional[pd.Series] = None,
        computability: Optional[ComputabilityResult] = None,
    ) -> Dict[str, Any]:
        """
        Full quantitative analysis pipeline.
        Returns an enriched Evidence Graph with:
          - KPI quality score
          - Ensemble anomaly detection consensus
          - STL decomposition summary
          - SHAP + correlation + Granger driver analysis
          - Confidence intervals on anomaly magnitude
        """
        # --- Dynamic time column resolution ---
        time_col: Optional[str] = None
        if enhanced_kpi and enhanced_kpi.time_column and enhanced_kpi.time_column in df.columns:
            time_col = enhanced_kpi.time_column
        elif "date" in df.columns:
            time_col = "date"

        is_classification = computability.is_classification if computability else False

        # --- Inject pre-computed KPI series into working DataFrame ---
        df_work = df.copy()
        if kpi_series is not None and kpi_name not in df_work.columns:
            df_work[kpi_name] = kpi_series.values

        # Threshold from enhanced KPI
        threshold = 0.10
        if enhanced_kpi:
            threshold = enhanced_kpi.thresholds.material_drop_percentage

        # === TIME-SERIES PIPELINE ===
        if time_col:
            df_work[time_col] = pd.to_datetime(df_work[time_col])
            if time_col != "date":
                df_work = df_work.rename(columns={time_col: "date"})

            # 1. KPI Quality Score
            kpi_quality = score_kpi_quality(df_work[kpi_name])

            # 2. STL Decomposition
            df_sorted_for_stl = df_work.sort_values("date")
            n_pts = len(df_sorted_for_stl)
            period = 12 if n_pts >= 24 else (7 if n_pts >= 14 else 4)
            decomp = decompose_series(df_sorted_for_stl[kpi_name].reset_index(drop=True), period=period)

            # 3. Ensemble Anomaly Detection
            anomaly_df = self.ml_engine.time_series_anomaly_detection(df_work, kpi_name)

            anomalies = anomaly_df[anomaly_df["is_anomaly"]]
            if anomalies.empty:
                return {
                    "status": "normal",
                    "message": "No material anomalies detected (ensemble: IsolationForest + Z-Score + CUSUM all agree).",
                    "kpi": kpi_name,
                    "enhanced_kpi_name": enhanced_kpi.name if enhanced_kpi else kpi_name,
                    "kpi_quality": kpi_quality,
                    "decomposition": decomp,
                }

            worst_anomaly = anomalies.sort_values("drop_percentage", ascending=False).iloc[0]
            drop_pct = float(worst_anomaly["drop_percentage"])
            anomaly_idx = anomaly_df.index.get_loc(worst_anomaly.name)

            is_material = self.rules.check_materiality(drop_pct, kpi_name)
            if not is_material:
                return {
                    "status": "normal",
                    "message": f"Ensemble flagged anomaly but severity ({drop_pct:.1%}) < threshold ({threshold:.0%}).",
                    "kpi": kpi_name,
                    "kpi_quality": kpi_quality,
                    "decomposition": decomp,
                }

            # 4. Confidence Interval on the drop
            anomaly_ci = compute_anomaly_confidence(
                anomaly_df[kpi_name].reset_index(drop=True), anomaly_idx
            )

            # 5. Full Causal Driver Analysis (SHAP + Corr + Granger)
            rich_result = self.ml_engine.causal_driver_analysis(
                anomaly_df, kpi_name, drivers, is_classification=False
            )
            importances = rich_result.get("importances", {})
            driver_analysis = rich_result.get("driver_analysis", {})

            # Ensemble vote breakdown
            ensemble_summary = {
                "isolation_forest_vote": int(worst_anomaly.get("if_flag", False)),
                "zscore_vote": int(worst_anomaly.get("zscore_flag", False)),
                "cusum_vote": int(worst_anomaly.get("cusum_flag", False)),
                "total_votes": int(worst_anomaly.get("ensemble_votes", 2)),
                "consensus": True,
            }

            return {
                "status": "anomaly_detected",
                "kpi": kpi_name,
                "enhanced_kpi_name": enhanced_kpi.name if enhanced_kpi else kpi_name,
                "kpi_formula": enhanced_kpi.formula if enhanced_kpi else kpi_name,
                "anomaly_date": worst_anomaly["date"].strftime("%Y-%m-%d"),
                "drop_percentage": round(drop_pct, 4),
                "is_material": bool(is_material),

                # New enriched fields
                "kpi_quality": kpi_quality,
                "decomposition": decomp,
                "anomaly_ensemble": ensemble_summary,
                "anomaly_significance": anomaly_ci,

                # Driver analysis (flat + rich)
                "driver_importances": importances,
                "driver_analysis": driver_analysis,
                "shap_available": rich_result.get("shap_available", False),
                "top_driver": list(importances.keys())[0] if importances else None,
            }

        # === CROSS-SECTIONAL PIPELINE ===
        else:
            if kpi_name not in df_work.columns:
                if enhanced_kpi and enhanced_kpi.source_columns:
                    for col in enhanced_kpi.source_columns:
                        if col in df_work.columns:
                            kpi_name = col
                            break

            kpi_quality = score_kpi_quality(df_work[kpi_name])
            kpi_rate = float(df_work[kpi_name].mean())

            rich_result = self.ml_engine.causal_driver_analysis(
                df_work, kpi_name, drivers, is_classification=is_classification
            )
            importances = rich_result.get("importances", {})
            driver_analysis = rich_result.get("driver_analysis", {})

            return {
                "status": "anomaly_detected",
                "kpi": kpi_name,
                "enhanced_kpi_name": enhanced_kpi.name if enhanced_kpi else kpi_name,
                "kpi_formula": enhanced_kpi.formula if enhanced_kpi else kpi_name,
                "anomaly_date": "Current Snapshot",
                "drop_percentage": round(kpi_rate, 4),
                "is_material": kpi_rate > threshold,

                # Quality
                "kpi_quality": kpi_quality,

                # Driver analysis
                "driver_importances": importances,
                "driver_analysis": driver_analysis,
                "shap_available": rich_result.get("shap_available", False),
                "top_driver": list(importances.keys())[0] if importances else None,
            }
