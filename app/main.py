import logging
import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import json
from pydantic import BaseModel, Field

from . import db_inspector, ai_analyzer
from .models import (
    AnalysisRequest, OptimizationRequest, ApplyConfirmationRequest,
    AnalysisSessionResult, Problem, OptimizationResult, ApplyResult,
    ChatMessage, ChatOnQueryRequest, ChatResponse,
    SandboxRequest, SandboxResult
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Database AI Optimizer API",
    description="API for analyzing and optimizing SQL databases.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

def parse_queries_from_file(content: str) -> list[str]:
    content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
    content = re.sub(r'--.*?\n', '', content)
    queries = [query.strip() for query in content.split(';') if query.strip()]
    return queries

@app.post("/start-analysis-session", response_model=AnalysisSessionResult)
async def start_analysis_session(request: AnalysisRequest):
    """
    Starts an analysis session based on the specified mode.
    - "auto": Checks for slow queries using database-specific tools (e.g., pg_stat_statements or Performance Schema).
    - "benchmark": Skips slow query checks and runs an AI-generated health check.
    - "file": Analyzes queries from a user-provided file.
    """
    try:
        inspector = db_inspector.get_inspector(request.db_uri)
        schema_details = inspector.get_schema_details()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Database connection or inspection failed: {e}")

    source = "empty"
    problems = []

    if request.mode == "auto":
        slow_queries = inspector.get_slow_queries()
        if slow_queries and slow_queries.get("queries"):
            source = slow_queries["source"]
            logger.info(f"Found {len(slow_queries['queries'])} queries in {source}.")
            for stat in slow_queries["queries"]:
                query, calls, mean_time = stat['query'], stat['calls'], stat['mean_time_ms']
                plan = inspector.get_query_plan(query)
                problems.append(Problem(query=query, execution_time_ms=mean_time, query_plan_before=plan, calls=calls))

    elif request.mode == "benchmark":
        logger.info("Running Automated Health Check.")
        source = "automated_benchmark"
        benchmark_queries = ai_analyzer.generate_benchmark_queries(inspector.dialect, schema_details)
        for query in benchmark_queries:
            try:
                plan, exec_time = inspector.get_query_plan_and_execution_time(query)
                problems.append(Problem(query=query, execution_time_ms=exec_time, query_plan_before=plan, calls=1))
            except Exception as e:
                problems.append(Problem(query=query, error=f"Execution failed: {e}"))

    elif request.mode == "file" and request.file_content:
        logger.info("Analyzing queries from user-provided file.")
        source = "user_file"
        user_queries = parse_queries_from_file(request.file_content)
        for query in user_queries:
            try:
                plan, exec_time = inspector.get_query_plan_and_execution_time(query)
                problems.append(Problem(query=query, execution_time_ms=exec_time, query_plan_before=plan, calls=1))
            except Exception as e:
                problems.append(Problem(query=query, error=f"Execution failed: {e}"))

    return AnalysisSessionResult(source=source, problems=problems)

@app.post("/get-optimization-suggestion", response_model=OptimizationResult)
async def get_optimization_suggestion(request: OptimizationRequest):
    try:
        inspector = db_inspector.get_inspector(request.db_uri)
        schema_details = inspector.get_schema_details()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Database connection or inspection failed: {e}")
    
    # --- THIS IS THE FIX ---
    # The keyword argument must be `query_plan`, not `query_plan_before`.
    ai_suggestion = ai_analyzer.get_ai_suggestion(
        dialect=inspector.dialect,
        schema_details=schema_details,
        query=request.query,
        query_plan=request.query_plan_before or {}, # CORRECTED ARGUMENT NAME
        execution_time_ms=request.execution_time_ms
    )

    est_exec_time_after = ai_suggestion.estimated_execution_time_after_ms
    ddl_statement = ai_suggestion.new_index_suggestion
    rewritten = ai_suggestion.rewritten_query

    logger.info("Original query: %s", request.query)
    logger.info("Original plan: %s", json.dumps(request.query_plan_before or {}, ensure_ascii=False))
    logger.info("AI suggestion: rewritten=%s index=%s explanation=%s",
                rewritten, ddl_statement, ai_suggestion.explanation)

    if ddl_statement:
        try:
            sim_result = inspector.simulate_ddl(ddl_statement, request.query)
            if sim_result:
                logger.info("Simulation result after DDL: %s", json.dumps(sim_result))
            else:
                logger.info("DDL simulation not supported or returned no result for this database type.")
        except Exception as e:
            logger.warning("simulate_ddl failed: %s", e)

    cost_info = ai_analyzer.calculate_cost_slayer(
        scanned_before_mb=ai_suggestion.estimated_data_scanned_before_mb,
        scanned_after_mb=ai_suggestion.estimated_data_scanned_after_mb
    )
    
    logger.info("Summary: exec_before=%.2f exec_after_est=%.2f",
                request.execution_time_ms, est_exec_time_after if est_exec_time_after else 0)
    logger.info("Cost Info: %s", cost_info.model_dump_json())

    return OptimizationResult(
        ai_suggestion=ai_suggestion,
        cost_slayer=cost_info,
        estimated_execution_time_after_ms=est_exec_time_after
    )

@app.post("/apply", response_model=ApplyResult)
async def apply_fix(request: ApplyConfirmationRequest):
    try:
        # The apply_ddl function is kept generic; it will create its own engine.
        success, message = db_inspector.apply_ddl(request.write_db_uri, request.ddl_statement)
        if not success:
            raise HTTPException(status_code=500, detail=message)
        return ApplyResult(success=True, message=message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

@app.post("/chat-on-query", response_model=ChatResponse)
async def chat_on_query(request: ChatOnQueryRequest):
    """
    Handles a RAG-based chat conversation about a specific query.
    """
    try:
        inspector = db_inspector.get_inspector(request.db_uri)
        schema_details = inspector.get_schema_details()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Database connection or inspection failed: {e}")

    try:
        ai_response = ai_analyzer.get_rag_chat_response(
            dialect=inspector.dialect,
            schema_details=schema_details,
            query_context=request.query_context,
            optimization_context=request.optimization_context,
            chat_history=request.chat_history,
            user_question=request.user_question
        )
        return ChatResponse(response=ai_response)
    except Exception as e:
        logger.error(f"Error during RAG chat: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing chat request: {e}")

@app.post("/verify-queries", response_model=SandboxResult)
async def verify_queries(request: SandboxRequest):
    """
    Executes the original and optimized queries with a LIMIT clause
    and compares their results to verify correctness.
    """
    try:
        inspector = db_inspector.get_inspector(request.db_uri)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Database connection failed: {e}")

    try:
        match, original_results, optimized_results, error_message = inspector.verify_queries_match(
            original_query=request.original_query,
            optimized_query=request.optimized_query
        )
        
        return SandboxResult(
            match=match,
            original_query_results=original_results,
            optimized_query_results=optimized_results,
            error=error_message
        )
    except Exception as e:
        logger.error(f"Error during query verification: {e}", exc_info=True)
        return SandboxResult(
            match=False,
            original_query_results=[],
            optimized_query_results=[],
            error=f"An unexpected server error occurred: {e}"
        )