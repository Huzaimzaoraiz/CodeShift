from typing import Dict, Any, List

class KPIComputabilityCheck:
    @staticmethod
    def verify(df, mapped_kpi: str, mapped_drivers: List[str]) -> bool:
        if mapped_kpi not in df.columns:
            raise ValueError(f"KPI '{mapped_kpi}' not found in the dataset.")
            
        for driver in mapped_drivers:
            if driver not in df.columns:
                raise ValueError(f"Driver '{driver}' not found in the dataset.")
                
        # Check variance
        if df[mapped_kpi].nunique() <= 1:
            raise ValueError(f"KPI '{mapped_kpi}' has no variance (constant value). Cannot run ML analysis.")
            
        return True
