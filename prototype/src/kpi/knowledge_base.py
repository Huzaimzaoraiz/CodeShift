"""
Org-scoped KPI Knowledge Base.

Each organisation has its own JSON file under config/kpi_knowledge/<org_slug>.json.
Feedback (user corrections) is stored under config/kpi_knowledge/feedback/<org_slug>.json.

KPI definitions do NOT leak across organisations.
"""
from __future__ import annotations
import json
import os
import re
import datetime
from typing import Any, Dict, List, Optional

from src.kpi.models import EnhancedKPI

# Resolve the config directory relative to this file:
# src/kpi/knowledge_base.py → ../../config/kpi_knowledge
_BASE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "config",
    "kpi_knowledge",
)
_FEEDBACK_DIR = os.path.join(_BASE_DIR, "feedback")


def _org_slug(org_id: str) -> str:
    """Convert org name to a safe filename slug."""
    return re.sub(r"[^a-zA-Z0-9_-]", "_", org_id).lower()


def _catalog_path(org_id: str) -> str:
    return os.path.join(_BASE_DIR, f"{_org_slug(org_id)}.json")


def _feedback_path(org_id: str) -> str:
    return os.path.join(_FEEDBACK_DIR, f"{_org_slug(org_id)}.json")


class KPIKnowledgeBase:
    """
    Reads and writes org-scoped KPI catalogs and feedback corrections.
    """

    # ---------- Read ----------

    def get_catalog(self, org_id: str) -> List[Dict[str, Any]]:
        """Return existing KPI definitions for this organisation (may be empty)."""
        path = _catalog_path(org_id)
        if not os.path.exists(path):
            return []
        with open(path) as f:
            data = json.load(f)
        return data.get("kpis", [])

    def get_feedback(self, org_id: str) -> List[Dict[str, Any]]:
        """Return user corrections for this organisation."""
        path = _feedback_path(org_id)
        if not os.path.exists(path):
            return []
        with open(path) as f:
            data = json.load(f)
        return data.get("corrections", [])

    # ---------- Write ----------

    def store_kpi(self, org_id: str, kpi: EnhancedKPI) -> None:
        """Persist an EnhancedKPI definition for this organisation."""
        os.makedirs(_BASE_DIR, exist_ok=True)
        path = _catalog_path(org_id)

        if os.path.exists(path):
            with open(path) as f:
                catalog = json.load(f)
        else:
            catalog = {"org_id": org_id, "kpis": [], "schema_version": "1.0"}

        kpi_dict = json.loads(kpi.model_dump_json())
        kpi_dict["last_updated"] = datetime.datetime.utcnow().isoformat()

        # Upsert by name
        existing = [k for k in catalog["kpis"] if k.get("name") != kpi.name]
        existing.append(kpi_dict)
        catalog["kpis"] = existing

        with open(path, "w") as f:
            json.dump(catalog, f, indent=2)

    def store_feedback(
        self, org_id: str, kpi_name: str, correction: str, user_id: str
    ) -> None:
        """Store a human feedback correction — org-scoped."""
        os.makedirs(_FEEDBACK_DIR, exist_ok=True)
        path = _feedback_path(org_id)

        if os.path.exists(path):
            with open(path) as f:
                store = json.load(f)
        else:
            store = {"org_id": org_id, "corrections": []}

        store["corrections"].append(
            {
                "kpi_name": kpi_name,
                "correction": correction,
                "user_id": user_id,
                "timestamp": datetime.datetime.utcnow().isoformat(),
            }
        )

        with open(path, "w") as f:
            json.dump(store, f, indent=2)

    def invalidate_kpi(self, org_id: str, kpi_name: str) -> None:
        """Mark a KPI as invalid (e.g., after schema change)."""
        path = _catalog_path(org_id)
        if not os.path.exists(path):
            return
        with open(path) as f:
            catalog = json.load(f)
        for kpi in catalog.get("kpis", []):
            if kpi.get("name") == kpi_name:
                kpi["schema_version"] = "INVALIDATED"
        with open(path, "w") as f:
            json.dump(catalog, f, indent=2)
