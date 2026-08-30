"""
Backward-compatible shim.
The real implementation is in src/kpi/computability.py.
This module keeps the old import path working for any legacy callers.
"""
from typing import Dict, Any, List
from src.kpi import computability as _new_computability
from src.kpi.models import EnhancedKPI, ComputabilityResult


class KPIComputabilityCheck:
    @staticmethod
    def verify(df, mapped_kpi: str, mapped_drivers: List[str]) -> bool:
        """Legacy interface — checks physical columns only."""
        if mapped_kpi not in df.columns:
            raise ValueError(f"KPI '{mapped_kpi}' not found in the dataset.")
        for driver in mapped_drivers:
            if driver not in df.columns:
                raise ValueError(f"Driver '{driver}' not found in the dataset.")
        if df[mapped_kpi].nunique() <= 1:
            raise ValueError(f"KPI '{mapped_kpi}' has no variance. Cannot run ML analysis.")
        return True

    @staticmethod
    def validate_enhanced(df, kpi: EnhancedKPI) -> ComputabilityResult:
        """New interface — full validation of EnhancedKPI."""
        return _new_computability.validate(df, kpi)
