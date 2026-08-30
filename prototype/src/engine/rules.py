from typing import Dict

class StatisticalRules:
    def __init__(self):
        pass
        
    def check_materiality(self, drop_percentage: float, kpi_name: str) -> bool:
        """
        Checks if the drop percentage crosses a generic materiality threshold (10%).
        """
        threshold = 0.10
        return drop_percentage >= threshold
