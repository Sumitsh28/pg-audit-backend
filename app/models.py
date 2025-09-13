from pydantic import BaseModel
from typing import List, Optional, Any

# --- Request Models ---

class AnalyzeRequest(BaseModel):
    """
    The request body for the main /analyze endpoint.
    Now includes a flag for the automated health check.
    """
    db_uri: str
    query_file_content: Optional[str] = None # For user-uploaded queries
    run_automated_check: Optional[bool] = False # For AI-generated benchmark


class DDLRequest(BaseModel):
    """
    The request body for /simulate and /apply endpoints.
    It includes the DDL statement and the original query for simulation.
    """
    db_uri: str
    ddl_statement: str
    original_query: str


class ApplyConfirmationRequest(DDLRequest):
    """
    Extends DDLRequest to require write credentials for the /apply endpoint.
    This is a critical security measure.
    """
    write_db_uri: str


# --- Response Models ---

class CostSlayerInfo(BaseModel):
    """
    Holds the cost analysis data.
    """
    estimated_daily_cost: float
    potential_savings_percentage: float
    cost_before: float
    cost_after: float


class AISuggestion(BaseModel):
    """
    The AI's generated fix and explanation.
    """
    rewritten_query: Optional[str]
    new_index_suggestion: Optional[str]
    explanation: str


class PerformanceIssue(BaseModel):
    """
    A single identified performance issue.
    Now includes execution time metrics.
    """
    id: int
    query: str
    # avg_execution_time_ms is from pg_stat_statements
    avg_execution_time_ms: Optional[float] = None
    # actual_execution_time_ms is from a direct EXPLAIN ANALYZE run
    actual_execution_time_ms: Optional[float] = None
    estimated_execution_time_after_ms: Optional[float] = None
    calls: int
    cost_slayer: CostSlayerInfo
    ai_suggestion: AISuggestion
    query_plan_before: Any # Could be JSON or text


class AnalysisResult(BaseModel):
    """
    The full response for the /analyze endpoint.
    """
    performance_issues: List[PerformanceIssue]


class SimulationResult(BaseModel):
    """
    The response for the /simulate endpoint.
    """
    query_plan_before: Any
    query_plan_after: Any
    estimated_improvement_factor: float


class ApplyResult(BaseModel):
    """
    The response for the /apply endpoint.
    """
    success: bool
    message: str