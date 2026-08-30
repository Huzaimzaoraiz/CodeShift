"""
Advanced Quantitative ML Engine.

Implements a full ensemble of statistical and ML methods:
- KPI Quality Scoring (ADF stationarity, variance, autocorrelation)
- 3-model Anomaly Ensemble (IsolationForest + Z-Score + CUSUM)
- STL Decomposition (trend / seasonality / residual separation)
- SHAP Feature Importance (directional, game-theory-based)
- Pearson + Spearman Correlation
- Granger Causality Test (statistical causal evidence)
- Confidence Intervals on anomaly magnitude
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple

from sklearn.ensemble import IsolationForest, RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from scipy import stats

# Graceful optional imports
try:
    import shap
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False

try:
    from statsmodels.tsa.stattools import adfuller, grangercausalitytests, acf
    from statsmodels.tsa.seasonal import STL
    _STATSMODELS_AVAILABLE = True
except ImportError:
    _STATSMODELS_AVAILABLE = False

warnings.filterwarnings("ignore")


# ─── KPI Quality Scorer ────────────────────────────────────────────────────────

def score_kpi_quality(series: pd.Series) -> Dict[str, Any]:
    """
    Score how 'analysis-ready' a KPI series is.
    Returns stationarity, variance score, autocorrelation.
    """
    result: Dict[str, Any] = {
        "n_observations": int(len(series)),
        "variance_score": 0.0,
        "stationarity": "unknown",
        "adf_p_value": None,
        "autocorrelation_lag1": None,
        "is_constant": False,
    }

    clean = series.dropna()
    if len(clean) < 4:
        result["stationarity"] = "insufficient_data"
        return result

    # Variance
    std = float(clean.std())
    mean = float(clean.mean())
    cv = (std / (abs(mean) + 1e-9))  # Coefficient of variation
    result["variance_score"] = round(min(cv, 1.0), 4)
    result["is_constant"] = bool(std < 1e-6)

    # Stationarity (ADF test)
    if _STATSMODELS_AVAILABLE and len(clean) >= 10:
        try:
            adf_result = adfuller(clean, autolag="AIC")
            p = float(adf_result[1])
            result["adf_p_value"] = round(p, 4)
            result["stationarity"] = "stationary" if p < 0.05 else "non_stationary"
        except Exception:
            pass

    # Lag-1 Autocorrelation
    if _STATSMODELS_AVAILABLE and len(clean) >= 4:
        try:
            ac = acf(clean, nlags=1, fft=False)
            result["autocorrelation_lag1"] = round(float(ac[1]), 4)
        except Exception:
            pass

    return result


# ─── Z-Score Anomaly ──────────────────────────────────────────────────────────

def zscore_anomaly(series: pd.Series, threshold: float = 2.5) -> pd.Series:
    """Return a boolean Series: True = anomaly by z-score."""
    z = np.abs(stats.zscore(series.fillna(series.mean())))
    return pd.Series(z > threshold, index=series.index)


# ─── CUSUM Anomaly ────────────────────────────────────────────────────────────

def cusum_anomaly(series: pd.Series, threshold_sigma: float = 4.0) -> pd.Series:
    """
    CUSUM (Cumulative Sum) control chart anomaly detection.
    Detects persistent drifts away from the mean.
    """
    mean = series.mean()
    std = series.std() + 1e-9
    threshold = threshold_sigma * std

    cusum_pos = np.zeros(len(series))
    cusum_neg = np.zeros(len(series))
    flags = np.zeros(len(series), dtype=bool)

    for i in range(1, len(series)):
        val = series.iloc[i]
        cusum_pos[i] = max(0, cusum_pos[i - 1] + (val - mean) - 0.5 * std)
        cusum_neg[i] = min(0, cusum_neg[i - 1] + (val - mean) + 0.5 * std)
        flags[i] = cusum_pos[i] > threshold or cusum_neg[i] < -threshold

    return pd.Series(flags, index=series.index)


# ─── STL Decomposition ────────────────────────────────────────────────────────

def decompose_series(series: pd.Series, period: int = 12) -> Dict[str, Any]:
    """
    Decompose a time series into trend/seasonal/residual using STL.
    Returns a summary dict (not the full arrays).
    """
    result = {
        "trend_direction": "unknown",
        "seasonality_detected": False,
        "seasonality_strength": 0.0,
        "trend_strength": 0.0,
        "decomposition_available": False,
    }
    if not _STATSMODELS_AVAILABLE or len(series) < period * 2:
        return result

    try:
        stl = STL(series.fillna(series.mean()), period=period, robust=True)
        fit = stl.fit()

        trend = pd.Series(fit.trend)
        seasonal = pd.Series(fit.seasonal)
        residual = pd.Series(fit.resid)

        # Trend strength: 1 - Var(residual) / Var(trend + residual)
        var_res = float(residual.var())
        var_trend = float((trend + residual).var()) + 1e-9
        result["trend_strength"] = round(max(0.0, 1 - var_res / var_trend), 4)

        # Seasonality strength
        var_seas = float((seasonal + residual).var()) + 1e-9
        result["seasonality_strength"] = round(max(0.0, 1 - var_res / var_seas), 4)
        result["seasonality_detected"] = result["seasonality_strength"] > 0.3

        # Trend direction from linear regression on trend component
        x = np.arange(len(trend))
        slope, _, _, _, _ = stats.linregress(x, trend.fillna(trend.mean()))
        result["trend_direction"] = "declining" if slope < -1e-6 else ("rising" if slope > 1e-6 else "flat")
        result["decomposition_available"] = True

    except Exception:
        pass

    return result


# ─── Confidence Interval on Drop ──────────────────────────────────────────────

def compute_anomaly_confidence(
    series: pd.Series, anomaly_idx: int, baseline_window: int = 6
) -> Dict[str, Any]:
    """
    Compute a 95% confidence interval on the drop magnitude.
    Compares the anomaly point against the rolling baseline mean.
    """
    result = {"ci_lower": None, "ci_upper": None, "p_value": None, "is_significant": False}
    try:
        start = max(0, anomaly_idx - baseline_window)
        baseline = series.iloc[start:anomaly_idx].dropna()
        anomaly_val = float(series.iloc[anomaly_idx])

        if len(baseline) < 3:
            return result

        t_stat, p_value = stats.ttest_1samp(baseline, anomaly_val)
        sem = stats.sem(baseline)
        ci = stats.t.interval(0.95, len(baseline) - 1, loc=baseline.mean(), scale=sem)
        drop_pct = float((baseline.mean() - anomaly_val) / (baseline.mean() + 1e-9))

        result["ci_lower"] = round(float(ci[0]), 4)
        result["ci_upper"] = round(float(ci[1]), 4)
        result["p_value"] = round(float(p_value), 4)
        result["is_significant"] = bool(p_value < 0.05)
        result["baseline_mean"] = round(float(baseline.mean()), 4)
        result["anomaly_value"] = round(anomaly_val, 4)

    except Exception:
        pass

    return result


# ─── Granger Causality ────────────────────────────────────────────────────────

def granger_test(kpi_series: pd.Series, driver_series: pd.Series, max_lag: int = 3) -> Dict[str, Any]:
    """
    Granger causality test: does driver_series *predict* kpi_series?
    Returns best p-value across lags.
    """
    result = {"p_value": None, "granger_causes": False, "best_lag": None}
    if not _STATSMODELS_AVAILABLE:
        return result
    if len(kpi_series) < max_lag * 4:
        return result

    try:
        data = pd.DataFrame({"kpi": kpi_series.values, "driver": driver_series.values}).dropna()
        if len(data) < max_lag * 4:
            return result
        test_results = grangercausalitytests(data[["kpi", "driver"]], maxlag=max_lag, verbose=False)
        # Extract best (minimum) p-value across lags
        p_values = {lag: res[0]["ssr_ftest"][1] for lag, res in test_results.items()}
        best_lag = min(p_values, key=p_values.get)
        best_p = float(p_values[best_lag])
        result["p_value"] = round(best_p, 4)
        result["granger_causes"] = bool(best_p < 0.05)
        result["best_lag"] = best_lag
    except Exception:
        pass

    return result


# ─── Main ML Engine ────────────────────────────────────────────────────────────

class MLEngine:
    def __init__(self):
        self.scaler = StandardScaler()
        self.anomaly_model = IsolationForest(contamination=0.05, random_state=42)
        self.causal_regressor = RandomForestRegressor(n_estimators=150, random_state=42)
        self.causal_classifier = RandomForestClassifier(n_estimators=150, random_state=42)

    # ── Time-Series Anomaly Detection (Ensemble) ──────────────────────────────

    def time_series_anomaly_detection(
        self, df: pd.DataFrame, kpi_col: str
    ) -> pd.DataFrame:
        """
        Ensemble anomaly detection: IsolationForest + Z-Score + CUSUM.
        An anomaly is flagged only if ≥ 2 of 3 models agree (majority vote).
        """
        df_sorted = df.sort_values("date").copy()
        rolling_mean = df_sorted[kpi_col].rolling(window=3, min_periods=1).mean()

        X = pd.DataFrame({
            "kpi_value": df_sorted[kpi_col],
            "rolling_mean": rolling_mean,
            "delta": df_sorted[kpi_col] - rolling_mean,
        }).fillna(0)

        X_scaled = self.scaler.fit_transform(X)

        # Model 1: Isolation Forest
        if_preds = self.anomaly_model.fit_predict(X_scaled)
        if_flags = pd.Series(if_preds == -1, index=df_sorted.index)

        # Model 2: Z-Score
        zs_flags = zscore_anomaly(df_sorted[kpi_col])

        # Model 3: CUSUM
        cs_flags = cusum_anomaly(df_sorted[kpi_col])

        # Majority vote (≥ 2 of 3)
        votes = if_flags.astype(int) + zs_flags.astype(int) + cs_flags.astype(int)
        consensus = votes >= 2

        df_sorted["is_anomaly"] = consensus
        df_sorted["if_flag"] = if_flags
        df_sorted["zscore_flag"] = zs_flags
        df_sorted["cusum_flag"] = cs_flags
        df_sorted["ensemble_votes"] = votes
        df_sorted["drop_percentage"] = (
            rolling_mean - df_sorted[kpi_col]
        ) / (rolling_mean + 1e-9)

        return df_sorted

    # ── Causal Driver Analysis (SHAP + Correlation + Granger) ─────────────────

    def causal_driver_analysis(
        self,
        df: pd.DataFrame,
        kpi_col: str,
        driver_cols: List[str],
        is_classification: bool = False,
    ) -> Dict[str, Any]:
        """
        Full causal analysis:
        - RandomForest feature importance
        - SHAP values (if available)
        - Pearson + Spearman correlation
        - Granger causality test (time-series only)

        Returns a rich dict instead of a bare importance dict,
        but still provides the legacy flat importance dict for
        backward compatibility.
        """
        available_drivers = [col for col in driver_cols if col in df.columns]
        df_clean = df.dropna(subset=[kpi_col] + available_drivers).copy()

        if df_clean.empty or not available_drivers:
            return {}

        X = df_clean[available_drivers]
        y = df_clean[kpi_col]

        # 1. RandomForest
        model = self.causal_classifier if is_classification else self.causal_regressor
        model.fit(X, y)
        rf_importances = dict(zip(available_drivers, model.feature_importances_))
        rf_importances = dict(sorted(rf_importances.items(), key=lambda x: x[1], reverse=True))

        # Build rich per-driver analysis
        driver_analysis: Dict[str, Dict[str, Any]] = {}

        for col in available_drivers:
            entry: Dict[str, Any] = {
                "rf_importance": round(float(rf_importances.get(col, 0.0)), 4),
            }

            # 2. Pearson correlation
            try:
                pearson_r, pearson_p = stats.pearsonr(df_clean[col].fillna(0), y.fillna(0))
                entry["pearson_r"] = round(float(pearson_r), 4)
                entry["pearson_p"] = round(float(pearson_p), 4)
            except Exception:
                entry["pearson_r"] = None
                entry["pearson_p"] = None

            # 3. Spearman correlation
            try:
                spearman_r, spearman_p = stats.spearmanr(df_clean[col].fillna(0), y.fillna(0))
                entry["spearman_r"] = round(float(spearman_r), 4)
                entry["spearman_p"] = round(float(spearman_p), 4)
            except Exception:
                entry["spearman_r"] = None
                entry["spearman_p"] = None

            # 4. Direction of effect
            if entry.get("pearson_r") is not None:
                r = entry["pearson_r"]
                entry["direction"] = "positive" if r > 0.05 else ("negative" if r < -0.05 else "neutral")
            else:
                entry["direction"] = "unknown"

            # 5. Granger causality (only if 'date' exists, so it's time-series)
            if "date" in df_clean.columns:
                gc = granger_test(y.reset_index(drop=True), df_clean[col].reset_index(drop=True))
                entry["granger_p_value"] = gc.get("p_value")
                entry["granger_causes"] = gc.get("granger_causes", False)
                entry["granger_lag"] = gc.get("best_lag")

            driver_analysis[col] = entry

        # 6. SHAP values
        shap_values_dict: Dict[str, float] = {}
        if _SHAP_AVAILABLE and not is_classification:
            try:
                explainer = shap.TreeExplainer(model)
                shap_vals = explainer.shap_values(X)
                mean_abs_shap = np.abs(shap_vals).mean(axis=0)
                for i, col in enumerate(available_drivers):
                    shap_values_dict[col] = round(float(mean_abs_shap[i]), 4)
                    driver_analysis[col]["shap_importance"] = shap_values_dict[col]
            except Exception:
                pass

        # Return the full rich dict; backward-compat simple importance dict is kept
        # under "driver_importances" key in the caller (pipeline.py)
        return {
            "importances": rf_importances,          # flat dict for ML engine compat
            "driver_analysis": driver_analysis,      # rich per-driver dict
            "shap_available": _SHAP_AVAILABLE and bool(shap_values_dict),
        }
