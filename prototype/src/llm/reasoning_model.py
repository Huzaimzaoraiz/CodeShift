"""
Reasoning Model with DB-Access LangGraph Node.

Pipeline: fetch_additional_insights → analyze_evidence → format_output

New node `fetch_additional_insights`:
  - LLM inspects the evidence graph and decides which additional SQL queries
    would deepen the analysis.
  - Safety gate: only SELECT, no DDL/DML, max 20 rows, org-scoped.
  - Results are serialised into the state and injected into the reasoning prompt.
"""
import json
import re
from typing import Any, Dict, List, Optional, Tuple, TypedDict

# pyrefly: ignore [missing-import]
from langchain_groq import ChatGroq
# pyrefly: ignore [missing-import]
from langgraph.graph import StateGraph, START, END


class ReasoningState(TypedDict):
    evidence_graph: Dict[str, Any]
    persona: str
    conf_score: float
    org_id: str                         # NEW — needed by DB insight node

    # DB insight results (populated by fetch_additional_insights node)
    db_insights: List[Dict[str, Any]]   # [{query, description, result_table}]

    # Internal CoT
    reasoning_trace: str

    # Final Outputs
    change_detection: str
    contributing_factors: str
    recommended_actions: str

    # Telemetry
    total_tokens: int
    total_cost: float


def _clean_json(raw: str) -> str:
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    raw = re.sub(r"```[a-z]*", "", raw).replace("```", "")
    start = raw.find("[")
    end = raw.rfind("]")
    if start != -1 and end != -1:
        return raw[start: end + 1]
    # Try single object fallback
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1:
        return "[" + raw[start: end + 1] + "]"
    return "[]"


_INSIGHT_SYSTEM = """You are a database analyst. You are given:
- An evidence graph describing a business KPI anomaly.
- A list of available SQL table columns.

Your job: propose up to 3 targeted SELECT-only SQL queries that would reveal
deeper insight into WHY this anomaly happened.

Rules:
- Only SELECT statements.
- Use only the columns listed in the schema.
- Keep queries simple and fast (no complex joins unless absolutely needed).
- Return ONLY a valid JSON array — no markdown, no extra text.

Format:
[
  {
    "description": "What this query will reveal",
    "sql": "SELECT ... FROM ... WHERE ... LIMIT 20"
  }
]
"""


class ReasoningModel:
    def __init__(self, api_key: str, model_name: str):
        # Override any DeepSeek model with Llama-3 to make the pipeline FAST
        if "deepseek" in model_name.lower() or "r1" in model_name.lower():
            model_name = "llama-3.3-70b-versatile"
            
        self.llm = ChatGroq(
            api_key=api_key,
            model_name=model_name,
            temperature=0.4,
            max_tokens=3000,
            max_retries=5
        )
        
        self.json_llm = ChatGroq(
            api_key=api_key,
            model_name=model_name,
            temperature=0.1,
            max_tokens=2000,
            max_retries=5
        )
        self._db_connector = None   # injected before graph run
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(ReasoningState)

        workflow.add_node("analyze_evidence", self.analyze_evidence_node)
        workflow.add_node("format_output", self.format_output_node)

        workflow.add_edge(START, "analyze_evidence")
        workflow.add_edge("analyze_evidence", "format_output")
        workflow.add_edge("format_output", END)

        return workflow.compile()

    # ── Node 1: DB Insight Queries ─────────────────────────────────────────────

    def fetch_additional_insights_node(self, state: ReasoningState) -> Dict[str, Any]:
        """
        LLM proposes targeted SQL queries; we execute them safely and return
        summarised results as a list of dicts.
        """
        ev = state["evidence_graph"]
        org_id = state.get("org_id", "")

        if not self._db_connector or not org_id:
            print("[Insight Node] No DB connector or org_id — skipping DB queries.")
            return {"db_insights": [], "total_tokens": state.get("total_tokens", 0),
                    "total_cost": state.get("total_cost", 0.0)}

        # Build a compact schema hint from the evidence graph
        known_cols = list(ev.get("driver_importances", {}).keys())
        kpi_col = ev.get("kpi", "")
        if kpi_col:
            known_cols = [kpi_col] + known_cols

        # Fetch actual table columns from DB to avoid hallucination
        try:
            sample_df = self._db_connector.safe_query(
                org_id,
                f"SELECT * FROM ({self._db_connector.configs[org_id]['query']}) AS t LIMIT 1"
            )
            all_cols = list(sample_df.columns)
        except Exception:
            all_cols = known_cols

        context = f"""
Anomaly KPI: {ev.get('enhanced_kpi_name', ev.get('kpi', 'Unknown'))}
Anomaly Date: {ev.get('anomaly_date', 'N/A')}
Drop: {ev.get('drop_percentage', 0):.1%}
Top Driver: {ev.get('top_driver', 'N/A')}
Available columns: {all_cols}
Base query the org uses: {self._db_connector.configs[org_id].get('query', 'SELECT *')}
"""
        from langchain_core.messages import SystemMessage, HumanMessage
        msgs = [
            SystemMessage(content=_INSIGHT_SYSTEM),
            HumanMessage(content=context),
        ]

        try:
            resp = self.json_llm.invoke(msgs)
            tokens = 0
            cost = 0.0
            try:
                tokens = resp.response_metadata["token_usage"]["total_tokens"]
                cost = tokens * 0.0000001
            except Exception:
                tokens = len(resp.content.split()) * 1.3
                cost = 0.0001

            cleaned = _clean_json(resp.content)
            proposed_queries = json.loads(cleaned)
        except Exception as exc:
            print(f"[Insight Node] LLM query proposal failed: {exc}")
            return {"db_insights": [], "total_tokens": state.get("total_tokens", 0),
                    "total_cost": state.get("total_cost", 0.0)}

        # Execute each proposed query through the safety gate
        insights = []
        for proposal in proposed_queries[:3]:
            sql = proposal.get("sql", "").strip()
            desc = proposal.get("description", "")
            if not sql:
                continue
            try:
                result_df = self._db_connector.safe_query(org_id, sql, max_rows=10)
                if result_df.empty:
                    result_str = "No rows returned."
                else:
                    result_str = result_df.to_string(index=False, max_rows=10)
                insights.append({
                    "description": desc,
                    "sql": sql,
                    "result": result_str,
                    "rows": len(result_df),
                })
                print(f"[Insight Node] [SUCCESS] Query executed: {desc}")
            except PermissionError as pe:
                print(f"[Insight Node] [BLOCKED]: {pe}")
                insights.append({"description": desc, "sql": sql,
                                  "result": f"BLOCKED: {pe}", "rows": 0})
            except Exception as exc:
                print(f"[Insight Node] [ERROR]: {exc}")
                insights.append({"description": desc, "sql": sql,
                                  "result": f"Error: {exc}", "rows": 0})

        return {
            "db_insights": insights,
            "total_tokens": state.get("total_tokens", 0) + int(tokens),
            "total_cost": state.get("total_cost", 0.0) + cost,
        }

    # ── Node 2: Evidence Analysis ──────────────────────────────────────────────

    def analyze_evidence_node(self, state: ReasoningState) -> Dict[str, Any]:
        ev = state["evidence_graph"]
        kpi_quality = ev.get("kpi_quality", {})
        decomp = ev.get("decomposition", {})
        ensemble = ev.get("anomaly_ensemble", {})
        ci = ev.get("anomaly_significance", {})
        driver_analysis = ev.get("driver_analysis", {})
        top_driver = ev.get("top_driver", "Unknown")

        # Format driver evidence
        driver_lines = []
        for col, dstats in (driver_analysis or {}).items():
            parts = [f"  * {col}:"]
            if dstats.get("rf_importance") is not None:
                parts.append(f"RF={dstats['rf_importance']:.3f}")
            if dstats.get("shap_importance") is not None:
                parts.append(f"SHAP={dstats['shap_importance']:.3f}")
            if dstats.get("pearson_r") is not None:
                parts.append(f"Pearson_r={dstats['pearson_r']:.3f}(p={dstats.get('pearson_p','?')})")
            if dstats.get("granger_causes") is True:
                parts.append(f"Granger_causal(p={dstats.get('granger_p_value','?')})")
            parts.append(f"direction={dstats.get('direction','?')}")
            driver_lines.append(" ".join(parts))
        driver_evidence = "\n".join(driver_lines) if driver_lines else "  No driver data available."

        # Format DB insight results
        db_insights = state.get("db_insights", [])
        if db_insights:
            insight_lines = []
            for ins in db_insights:
                insight_lines.append(f"\n  Query: {ins['description']}")
                insight_lines.append(f"  SQL: {ins['sql']}")
                insight_lines.append(f"  Result ({ins['rows']} rows):\n{ins['result']}")
            db_section = "\n".join(insight_lines)
        else:
            db_section = "  No additional DB queries executed."

        prompt = f"""
        You are an elite Quantitative Data Scientist and Business Strategist.
        Perform rigorous Chain-of-Thought analysis using ALL available statistical evidence
        AND the additional database query results retrieved for deeper insight:

        === KPI OVERVIEW ===
        KPI: {ev.get('enhanced_kpi_name', ev.get('kpi', 'Unknown'))}
        Formula: {ev.get('kpi_formula', 'N/A')}
        Anomaly Date: {ev.get('anomaly_date', 'Unknown')}
        Drop Magnitude: {ev.get('drop_percentage', 0):.1%}

        === STATISTICAL EVIDENCE ===
        KPI Quality:
          - Stationarity: {kpi_quality.get('stationarity', 'unknown')} (ADF p={kpi_quality.get('adf_p_value', 'N/A')})
          - Variance Score: {kpi_quality.get('variance_score', 'N/A')}
          - Autocorrelation (lag-1): {kpi_quality.get('autocorrelation_lag1', 'N/A')}

        STL Decomposition:
          - Trend: {decomp.get('trend_direction', 'unknown')} (strength={decomp.get('trend_strength', 0):.2f})
          - Seasonality Detected: {decomp.get('seasonality_detected', False)} (strength={decomp.get('seasonality_strength', 0):.2f})

        Anomaly Ensemble (3-model majority vote):
          - IsolationForest: {ensemble.get('isolation_forest_vote', '?')} | Z-Score: {ensemble.get('zscore_vote', '?')} | CUSUM: {ensemble.get('cusum_vote', '?')}
          - Consensus: {ensemble.get('total_votes', '?')}/3 models agree

        Statistical Significance:
          - p-value: {ci.get('p_value', 'N/A')} | Significant: {ci.get('is_significant', False)}
          - 95% CI: [{ci.get('ci_lower', 'N/A')}, {ci.get('ci_upper', 'N/A')}]
          - Baseline mean: {ci.get('baseline_mean', 'N/A')} -> Anomaly value: {ci.get('anomaly_value', 'N/A')}

        === CAUSAL DRIVER EVIDENCE (SHAP + Correlation + Granger) ===
{driver_evidence}
        Top Driver: {top_driver} (ML Confidence Score: {state['conf_score']:.2f})

        === ADDITIONAL DATABASE INSIGHTS (Live Queries) ===
{db_section}

        Write step-by-step reasoning:
        1. Assess the statistical robustness (significance, ensemble consensus, CI).
        2. Explain trend and seasonality context.
        3. Analyse each driver quantitatively using Pearson, SHAP, and Granger evidence.
        4. Integrate the ADDITIONAL DATABASE INSIGHTS to provide deeper, data-backed conclusions.
        5. Propose a specific, data-driven corrective action grounded in ALL the evidence above.
        Do NOT output JSON. Write raw analytical reasoning only.
        """
        response = self.llm.invoke(prompt)

        try:
            tokens = response.response_metadata["token_usage"]["total_tokens"]
            cost = tokens * 0.0000001
        except Exception:
            tokens = len(response.content.split()) * 1.3
            cost = 0.0001

        return {
            "reasoning_trace": response.content,
            "total_tokens": state.get("total_tokens", 0) + int(tokens),
            "total_cost": state.get("total_cost", 0.0) + cost,
        }

    # ── Node 3: Format Output ──────────────────────────────────────────────────

    def format_output_node(self, state: ReasoningState) -> Dict[str, Any]:
        prompt = f"""
        You are an Enterprise AI BI Analyst speaking to a {state['persona']}.

        Based on the following internal reasoning trace:
        ---
        {state['reasoning_trace']}
        ---

        Synthesize the final narrative.
        Output ONLY a valid JSON object in this exact format, with NO markdown formatting around it:
        {{
            "change_detection": "A concise summary of the anomaly finding, tailored to a {state['persona']}.",
            "contributing_factors": "Explain the mathematical root cause from the trace.",
            "recommended_actions": "Prescribe the specific action based on the evidence and DB insights."
        }}
        """
        response = self.json_llm.invoke(prompt)

        try:
            tokens = response.response_metadata["token_usage"]["total_tokens"]
            cost = tokens * 0.0000001
        except Exception:
            tokens = len(response.content.split()) * 1.3
            cost = 0.0001

        content = response.content.strip()
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        start_idx = content.find("{")
        end_idx = content.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            content = content[start_idx: end_idx + 1]

        try:
            mapping = json.loads(content)
        except json.JSONDecodeError:
            mapping = {
                "change_detection": "Anomaly Detected.",
                "contributing_factors": f"Error parsing. Trace: {state['reasoning_trace'][:100]}...",
                "recommended_actions": "N/A",
            }

        return {
            "change_detection": mapping.get("change_detection", "N/A"),
            "contributing_factors": mapping.get("contributing_factors", "N/A"),
            "recommended_actions": mapping.get("recommended_actions", "N/A"),
            "total_tokens": state.get("total_tokens", 0) + int(tokens),
            "total_cost": state.get("total_cost", 0.0) + cost,
        }

    # ── Confidence Scorer ──────────────────────────────────────────────────────

    def confidence_scorer(self, evidence_graph: Dict[str, Any]) -> Tuple[bool, float]:
        if evidence_graph.get("status") != "anomaly_detected":
            return False, 0.0
        importances = evidence_graph.get("driver_importances", {})
        if not importances:
            return False, 0.0
        top_importance = list(importances.values())[0]
        if top_importance < 0.2:
            return False, top_importance
        return True, top_importance

    # ── Public Entry Point ─────────────────────────────────────────────────────

    def generate_story(
        self,
        evidence_graph: Dict[str, Any],
        persona: str = "Executive Team",
        db_connector=None,
        org_id: str = "",
    ) -> Tuple[str, str, str, str, Dict[str, Any]]:
        """
        Generate the full narrative story.

        Parameters
        ----------
        db_connector : DBConnector instance (optional) — enables live DB queries.
        org_id       : Organisation ID for scoped DB access.
        """
        is_confident, conf_score = self.confidence_scorer(evidence_graph)
        telemetry_raw = {"tokens": 0, "cost_usd": 0.0}

        if not is_confident:
            return (
                "ABSTENTION / CLARIFICATION REQUEST",
                f"I detected a drop of {evidence_graph.get('drop_percentage', 0):.1%} in {evidence_graph.get('kpi', 'Unknown KPI')}.",
                f"Statistical confidence in root cause is too low (Score: {conf_score:.2f}).",
                "Please request a deeper manual analysis by the Data Science team.",
                telemetry_raw,
            )

        # Inject DB connector into self for the node to use
        self._db_connector = db_connector

        initial_state: ReasoningState = {
            "evidence_graph": evidence_graph,
            "persona": persona,
            "conf_score": conf_score,
            "org_id": org_id,
            "db_insights": [],
            "reasoning_trace": "",
            "change_detection": "",
            "contributing_factors": "",
            "recommended_actions": "",
            "total_tokens": 0,
            "total_cost": 0.0,
        }

        print(f"--- Starting LangGraph Reasoning Run for {persona} (org={org_id}) ---")

        final_state = self.graph.invoke(initial_state)

        print("\n--- DB Insights Fetched ---")
        for ins in final_state.get("db_insights", []):
            print(f"  [{ins['rows']} rows] {ins['description']}")
        print("\n--- Internal Reasoning Trace (LangGraph) ---")
        print(final_state["reasoning_trace"])
        print("--------------------------------------------\n")

        telemetry_raw["tokens"] = final_state["total_tokens"]
        telemetry_raw["cost_usd"] = final_state["total_cost"]
        telemetry_raw["db_insights"] = final_state.get("db_insights", [])

        return (
            "Complete",
            final_state.get("change_detection", "N/A"),
            final_state.get("contributing_factors", "N/A"),
            final_state.get("recommended_actions", "N/A"),
            telemetry_raw,
        )
