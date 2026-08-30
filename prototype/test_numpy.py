import json
# pyrefly: ignore [missing-import]
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.bool_): return bool(obj)
        return super(NumpyEncoder, self).default(obj)

data = {"is_anomaly": np.bool_(True)}
print(json.loads(json.dumps(data, cls=NumpyEncoder)))
