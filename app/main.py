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

    ai_suggestion = ai_analyzer.get_ai_suggestion(schema_details, request.query, request.query_plan_before or {})
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

    cost_before = 0
    if isinstance(request.query_plan_before, dict):
        cost_before = request.query_plan_before.get("Total Cost", 0)
    cost_after = cost_before
    execution_time_after = None

    if ddl_statement:
        is_safe = getattr(ai_analyzer, "is_safe_create_index", lambda s: False)
        if not is_safe(ddl_statement):
            ai_suggestion.explanation = (ai_suggestion.explanation or "") + " (index suggestion rejected: unsafe syntax)"
            print("\nIndex suggestion rejected due to unsafe syntax.")
            logger.warning("Index suggestion rejected: %s", ddl_statement)
        else:
            try:
                print("\nApplying simulated DDL and running EXPLAIN ANALYZE (in transaction)...")
                sim = db_inspector.simulate_ddl(engine, ddl_statement, request.query)
                if sim:
                    cost_after = sim.get("total_cost") or cost_before
                    execution_time_after = sim.get("execution_time_ms")
                    print("\n--- SIMULATION RESULT (AFTER DDL) ---")
                    print("Total Cost (after):", cost_after)
                    print("Execution Time (after ms):", execution_time_after)
                    print("Plan (after):")
                    print(json.dumps(sim.get("plan") or {}, indent=2))
                    logger.info("Simulation result after DDL: cost=%s time=%s plan=%s",
                                cost_after, execution_time_after, json.dumps(sim.get("plan") or {}, ensure_ascii=False))
                else:
                    print("\nSimulation returned no result.")
                    logger.warning("simulate_ddl returned no result for DDL: %s", ddl_statement)
            except Exception as e:
                logger.warning("simulate_ddl failed: %s", e)
                print(f"\nSimulation failed: {e}")

    elif rewritten:
        try:
            print("\nRunning EXPLAIN (ANALYZE, FORMAT JSON) on rewritten query...")
            with engine.connect() as conn:
                explain_q = text(f"EXPLAIN (ANALYZE, FORMAT JSON) {rewritten}")
                res = conn.execute(explain_q).scalar_one()
                if isinstance(res, (list, tuple)) and len(res) > 0:
                    plan_after = res[0].get("Plan")
                    cost_after = plan_after.get("Total Cost", cost_before) if plan_after else cost_before
                    execution_time_after = res[0].get("Execution Time")
                    print("\n--- ANALYZE RESULT (REWRITTEN) ---")
                    print("Total Cost (after):", cost_after)
                    print("Execution Time (after ms):", execution_time_after)
                    print("Plan (after):")
                    print(json.dumps(plan_after or {}, indent=2))
                    logger.info("Analyze rewritten query result: cost=%s time=%s plan=%s",
                                cost_after, execution_time_after, json.dumps(plan_after or {}, ensure_ascii=False))
                else:
                    print("\nNo plan returned for rewritten query.")
                    logger.warning("EXPLAIN returned empty for rewritten query.")
        except Exception as e:
            logger.warning("Could not ANALYZE rewritten query: %s", e)
            print(f"\nCould not ANALYZE rewritten query: {e}")

    cost_info = ai_analyzer.calculate_cost_slayer(cost_before, cost_after, calls=1)
    est_exec_time_after = None
    if execution_time_after is not None:
        est_exec_time_after = execution_time_after
    elif request.execution_time_ms and cost_before > 0 and cost_after > 0:
        improvement_factor = cost_before / cost_after if cost_after > 0 else 1.0
        est_exec_time_after = request.execution_time_ms / improvement_factor

    print("\n=== SUMMARY ===")
    print("Total Cost (before):", cost_before)
    print("Total Cost (after):", cost_after)
    print("Execution Time (before ms):", request.execution_time_ms)
    print("Execution Time (after ms) [measured or estimated]:", est_exec_time_after)
    print("Estimated daily cost:", cost_info.estimated_daily_cost)
    print("Potential savings %:", cost_info.potential_savings_percentage)
    print("=== OPTIMIZER: ANALYSIS END ===\n")

    logger.info("Summary: cost_before=%s cost_after=%s exec_before=%s exec_after=%s savings=%s%%",
                cost_before, cost_after, request.execution_time_ms, est_exec_time_after, cost_info.potential_savings_percentage)

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
