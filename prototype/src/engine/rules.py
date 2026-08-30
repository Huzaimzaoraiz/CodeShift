from typing import Dict

class StatisticalRules:
    def __init__(self, kpi_kb: Dict):
        self.kpi_kb = kpi_kb
        
    def check_materiality(self, drop_percentage: float, kpi_name: str) -> bool:
        """
        Checks if the drop percentage crosses the materiality threshold defined in the KB.
        """
        threshold = self.kpi_kb['kpis'].get(kpi_name, {}).get('thresholds', {}).get('material_drop_percentage', 0.1)
        return drop_percentage >= threshold
