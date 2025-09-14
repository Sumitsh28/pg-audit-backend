from pydantic import BaseModel,  Field
from typing import List, Optional, Any, Dict

# --- Shared Models ---

class AISuggestion(BaseModel):
    rewritten_query: Optional[str]
    new_index_suggestion: Optional[str]
    explanation: str
    estimated_execution_time_after_ms: Optional[float]
    estimated_data_scanned_before_mb: Optional[float]
    estimated_data_scanned_after_mb: Optional[float]

class ProviderCostSavings(BaseModel):
    provider_name: str
    potential_savings_inr_per_call: float

class CostSlayerInfo(BaseModel):
    potential_savings: List[ProviderCostSavings]

# --- Request Models ---

class AnalysisRequest(BaseModel):
    """
    A unified request to start an analysis session.
    """
    db_uri: str
    mode: str  # "auto", "benchmark", or "file"
    file_content: Optional[str] = None

class OptimizationRequest(BaseModel):
    db_uri: str
    query: str
    execution_time_ms: float
    query_plan_before: Optional[Dict[str, Any]]
    # ADD THIS LINE:
    calls: Optional[int] = None

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
    ai_suggestion: AISuggestion
    cost_slayer: CostSlayerInfo
    estimated_execution_time_after_ms: Optional[float]
    

class ApplyResult(BaseModel):
    success: bool
    message: str
    
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatOnQueryRequest(BaseModel):
    db_uri: str
    query_context: "Problem"  
    optimization_context: Optional["OptimizationResult"] = None
    chat_history: List[ChatMessage]
    user_question: str

class ChatResponse(BaseModel):
    response: str
    
class SandboxRequest(BaseModel):
    """Request to the /verify-queries endpoint."""
    db_uri: str
    original_query: str
    optimized_query: Optional[str] = None

class SandboxResult(BaseModel):
    """Response from the /verify-queries endpoint."""
    match: bool
    original_query_results: List[Dict[str, Any]]
    optimized_query_results: List[Dict[str, Any]]
    error: Optional[str] = None