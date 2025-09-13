from pydantic import BaseModel
from typing import List, Optional, Any

# --- Shared Models ---

class AISuggestion(BaseModel):
    rewritten_query: Optional[str]
    new_index_suggestion: Optional[str]
    explanation: str

class CostSlayerInfo(BaseModel):
    estimated_daily_cost: float
    potential_savings_percentage: float
    cost_before: float
    cost_after: float

# --- Request Models ---

class AnalysisRequest(BaseModel):
    """
    A unified request to start an analysis session.
    """
    db_uri: str
    mode: str  # "auto", "benchmark", or "file"
    file_content: Optional[str] = None

class OptimizationRequest(BaseModel):
    """
    Contains all the "before" data for a single query to be optimized.
    """
    db_uri: str
    query: str
    query_plan_before: Any
    # This can be optional as we may not have it for pg_stat_statements queries initially
    execution_time_ms: Optional[float] = None

class ApplyConfirmationRequest(BaseModel):
    write_db_uri: str
    ddl_statement: str

# --- Response Models ---

class Problem(BaseModel):
    """
    Represents a single identified problem (a slow query).
    """
    query: str
    # This will be the average time from pg_stat or the actual time from a benchmark
    execution_time_ms: Optional[float] = None
    query_plan_before: Optional[Any] = None
    calls: Optional[int] = None # Only available for pg_stat_statements
    error: Optional[str] = None

class AnalysisSessionResult(BaseModel):
    """
    The result of starting an analysis session.
    """
    source: str # "pg_stat_statements", "automated_benchmark", "user_file", "empty"
    problems: List[Problem]

class OptimizationResult(BaseModel):
    """
    The optimization details for a single query (the "solution").
    """
    ai_suggestion: AISuggestion
    cost_slayer: CostSlayerInfo
    estimated_execution_time_after_ms: Optional[float] = None

class ApplyResult(BaseModel):
    success: bool
    message: str