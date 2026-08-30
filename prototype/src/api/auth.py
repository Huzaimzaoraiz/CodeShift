import jwt
from fastapi import HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Dict, Any, List

# In a real app, this is in an environment variable!
SECRET_KEY = "enterprise_super_secret_key"
ALGORITHM = "HS256"

security = HTTPBearer()

# Mock Database of Roles and Entitlements
ROLE_DB = {
    "cmo": {
        "persona": "Chief Marketing Officer",
        "entitlements": []  # Empty means access to ALL columns
    },
    "analyst": {
        "persona": "Junior Data Analyst",
        # Restricted from seeing financial data like MonthlyCharges, TotalCharges
        "entitlements": ["customerID", "gender", "SeniorCitizen", "Partner", "Dependents", 
                         "tenure_months", "PhoneService", "MultipleLines", "InternetService", 
                         "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", 
                         "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", 
                         "PaymentMethod", "churn_status"]
    }
}

def create_access_token(role: str) -> str:
    role_lower = role.lower()
    if role_lower not in ROLE_DB:
        raise ValueError(f"Role {role} not found in Role DB.")
        
    payload = {
        "role": role_lower,
        "persona": ROLE_DB[role_lower]["persona"],
        "entitlements": ROLE_DB[role_lower]["entitlements"]
    }
    
    encoded_jwt = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_current_user(credentials: HTTPAuthorizationCredentials = Security(security)) -> Dict[str, Any]:
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Could not validate credentials or token expired")
