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
    ChatMessage, ChatOnQueryRequest, ChatResponse
)



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="PostgreSQL AI Optimizer API",
    description="API for analyzing and optimizing PostgreSQL databases.",
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
    - "auto": Checks pg_stat_statements. The primary, initial mode.
    - "benchmark": Skips pg_stat_statements and runs an AI-generated health check.
    - "file": Analyzes queries from a user-provided file.
    """
    try:
        engine = db_inspector.get_engine(request.db_uri)
        schema_details = db_inspector.get_schema_details(engine)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Database connection failed: {e}")

    source = "empty"
    problems = []
    
    if request.mode == "auto":
        pg_stats = db_inspector.get_pg_stat_statements(engine)
        if pg_stats:
            logger.info(f"Found {len(pg_stats)} queries in pg_stat_statements.")
            source = "pg_stat_statements"
            for stat in pg_stats:
                query, calls, _, mean_time = stat
                plan = db_inspector.get_query_plan(engine, query)
                problems.append(Problem(query=query, execution_time_ms=mean_time, query_plan_before=plan, calls=calls))
    
    elif request.mode == "benchmark":
        logger.info("Running Automated Health Check.")
        source = "automated_benchmark"
        benchmark_queries = ai_analyzer.generate_benchmark_queries(schema_details)
        for query in benchmark_queries:
            try:
                plan, exec_time = db_inspector.get_query_plan_and_execution_time(engine, query)
                problems.append(Problem(query=query, execution_time_ms=exec_time, query_plan_before=plan, calls=1))
            except Exception as e:
                problems.append(Problem(query=query, error=f"Execution failed: {e}"))
    
    elif request.mode == "file" and request.file_content:
        logger.info("Analyzing queries from user-provided file.")
        source = "user_file"
        user_queries = parse_queries_from_file(request.file_content)
        for query in user_queries:
            try:
                plan, exec_time = db_inspector.get_query_plan_and_execution_time(engine, query)
                problems.append(Problem(query=query, execution_time_ms=exec_time, query_plan_before=plan, calls=1))
            except Exception as e:
                problems.append(Problem(query=query, error=f"Execution failed: {e}"))
    
    return AnalysisSessionResult(source=source, problems=problems)

@app.post("/get-optimization-suggestion", response_model=OptimizationResult)
async def get_optimization_suggestion(request: OptimizationRequest):
    try:
        engine = db_inspector.get_engine(request.db_uri)
        schema_details = db_inspector.get_schema_details(engine)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Database connection failed: {e}")

    ai_suggestion = ai_analyzer.get_ai_suggestion(
        schema_details,
        request.query,
        request.query_plan_before or {},
        request.execution_time_ms
    )
    
    est_exec_time_after = ai_suggestion.estimated_execution_time_after_ms
    ddl_statement = ai_suggestion.new_index_suggestion
    rewritten = ai_suggestion.rewritten_query

    print("\n=== OPTIMIZER: ANALYSIS START ===")
    print("Original query:")
    print(request.query)
    logger.info("Original query: %s", request.query)

    print("\nOriginal plan (if available):")
    print(json.dumps(request.query_plan_before or {}, indent=2))
    logger.info("Original plan: %s", json.dumps(request.query_plan_before or {}, ensure_ascii=False))

    print("\nAI suggestion:")
    print("Rewritten query:", rewritten)
    print("Index suggestion:", ddl_statement)
    print("Explanation:", ai_suggestion.explanation)
    logger.info("AI suggestion: rewritten=%s index=%s explanation=%s",
                rewritten, ddl_statement, ai_suggestion.explanation)

   
    if ddl_statement:
        try:
            sim = db_inspector.simulate_ddl(engine, ddl_statement, request.query)
            if sim:
                logger.info("Simulation result after DDL: %s", json.dumps(sim))
        except Exception as e:
            logger.warning("simulate_ddl failed: %s", e)
    elif rewritten:
        try:
            with engine.connect() as conn:
                explain_q = text(f"EXPLAIN (ANALYZE, FORMAT JSON) {rewritten}")
                res = conn.execute(explain_q).scalar_one()
                logger.info("Analyze rewritten query result: %s", res)
        except Exception as e:
            logger.warning("Could not ANALYZE rewritten query: %s", e)
            
    cost_info = ai_analyzer.calculate_cost_slayer(
        scanned_before_mb=ai_suggestion.estimated_data_scanned_before_mb,
        scanned_after_mb=ai_suggestion.estimated_data_scanned_after_mb
    )

    print("\n=== SUMMARY ===")
    print(f"Execution Time (before ms): {request.execution_time_ms:.2f}")
    print(f"Execution Time (after ms) [AI ESTIMATE]: {est_exec_time_after:.2f}" if est_exec_time_after is not None else "Execution Time (after ms) [AI ESTIMATE]: N/A")
    print(f"Data Scanned (before MB) [AI ESTIMATE]: {ai_suggestion.estimated_data_scanned_before_mb}")
    print(f"Data Scanned (after MB) [AI ESTIMATE]: {ai_suggestion.estimated_data_scanned_after_mb}")

    if cost_info.potential_savings:
        print("\nPotential Daily Savings (INR):")
       
    else:
        print("\nNo potential cost savings identified.")
    
    print("=== OPTIMIZER: ANALYSIS END ===\n")
    
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
        
        write_db_uri = request.write_db_uri
        success, message = db_inspector.apply_ddl(write_db_uri, request.ddl_statement)
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
        engine = db_inspector.get_engine(request.db_uri)
        schema_details = db_inspector.get_schema_details(engine)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Database connection failed: {e}")

    try:
        ai_response = ai_analyzer.get_rag_chat_response(
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
