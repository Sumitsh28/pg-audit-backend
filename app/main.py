import logging
import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from . import db_inspector, ai_analyzer
from .models import (
    AnalyzeRequest, AnalysisResult, PerformanceIssue,
    DDLRequest, SimulationResult,
    ApplyConfirmationRequest, ApplyResult
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="PostgreSQL AI Optimizer API",
    description="API for analyzing and optimizing PostgreSQL databases.",
    version="1.1.0" # Version bump
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def parse_queries_from_file(content: str) -> list[str]:
    """Splits a string of SQL text into individual queries, ignoring comments."""
    # Remove multi-line comments
    content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
    # Remove single-line comments
    content = re.sub(r'--.*?\n', '', content)
    # Split by semicolon and filter out empty statements
    queries = [query.strip() for query in content.split(';') if query.strip()]
    return queries

@app.post("/analyze", response_model=AnalysisResult)
async def analyze_database(request: AnalyzeRequest):
    try:
        engine = db_inspector.get_engine(request.db_uri)
        schema_details = db_inspector.get_schema_details(engine)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Database connection failed: {e}")

    source_queries = []
    analysis_mode = None

    # Step 1: Determine the source of queries to analyze
    pg_stats = db_inspector.get_pg_stat_statements(engine)
    if pg_stats:
        logger.info(f"Analyzing {len(pg_stats)} queries from pg_stat_statements.")
        source_queries = pg_stats
        analysis_mode = 'pg_stat'
    elif request.run_automated_check:
        logger.info("Running Automated Health Check.")
        source_queries = ai_analyzer.generate_benchmark_queries(schema_details)
        analysis_mode = 'benchmark'
    elif request.query_file_content:
        logger.info("Analyzing queries from user-provided file.")
        source_queries = parse_queries_from_file(request.query_file_content)
        analysis_mode = 'user_file'
    else:
        # No queries found or provided, return empty response for frontend to handle
        return AnalysisResult(performance_issues=[])

    # Step 2: Process each query from the determined source
    issues = []
    for i, item in enumerate(source_queries):
        try:
            query, calls, plan_before, exec_time_before = None, 1, None, None

            if analysis_mode == 'pg_stat':
                query, calls, _, avg_exec_time = item
                plan_before = db_inspector.get_query_plan(engine, query)
                exec_time_before = avg_exec_time
            else: # benchmark or user_file
                query = item
                plan_before, exec_time_before = db_inspector.get_query_plan_and_execution_time(engine, query)
            
            if "error" in plan_before:
                logger.warning(f"Skipping query due to EXPLAIN error: {plan_before['error']}")
                continue

            cost_before = plan_before.get("Total Cost", 0)

            # Step 3: Get AI suggestion and simulate the fix
            ai_suggestion = ai_analyzer.get_ai_suggestion(schema_details, query, plan_before)
            ddl_statement = ai_suggestion.new_index_suggestion or ai_suggestion.rewritten_query
            
            plan_after = None
            cost_after = cost_before # Default to same cost if no fix is suggested
            if ddl_statement:
                plan_after = db_inspector.simulate_ddl(engine, ddl_statement, query)
                cost_after = plan_after.get("Plan", {}).get("Total Cost", cost_before)

            # Step 4: Calculate costs and estimate time savings
            cost_info = ai_analyzer.calculate_cost_slayer(cost_before, cost_after, calls)
            
            est_exec_time_after = None
            if exec_time_before and cost_before > 0 and cost_after > 0:
                improvement_factor = cost_before / cost_after
                est_exec_time_after = exec_time_before / improvement_factor

            # Step 5: Assemble the final performance issue object
            issue = PerformanceIssue(
                id=i,
                query=query,
                avg_execution_time_ms=exec_time_before if analysis_mode == 'pg_stat' else None,
                actual_execution_time_ms=exec_time_before if analysis_mode != 'pg_stat' else None,
                estimated_execution_time_after_ms=est_exec_time_after,
                calls=calls,
                cost_slayer=cost_info,
                ai_suggestion=ai_suggestion,
                query_plan_before=plan_before
            )
            issues.append(issue)

        except Exception as e:
            logger.error(f"Failed to fully analyze query '{item}': {e}", exc_info=True)
            continue
            
    return AnalysisResult(performance_issues=issues)


# The /simulate and /apply endpoints remain useful for manual testing or re-verification
@app.post("/simulate", response_model=SimulationResult)
async def simulate_fix(request: DDLRequest):
    try:
        engine = db_inspector.get_engine(request.db_uri)
        plan_before = db_inspector.get_query_plan(engine, request.original_query)
        cost_before = plan_before.get("Plan", {}).get("Total Cost", 0)
        plan_after = db_inspector.simulate_ddl(engine, request.ddl_statement, request.original_query)
        cost_after = plan_after.get("Plan", {}).get("Total Cost", 1)

        improvement_factor = round(cost_before / cost_after, 1) if cost_after > 0 else 0
        return SimulationResult(
            query_plan_before=plan_before,
            query_plan_after=plan_after,
            estimated_improvement_factor=improvement_factor
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simulation failed: {e}")


@app.post("/apply", response_model=ApplyResult)
async def apply_fix(request: ApplyConfirmationRequest):
    if not request.write_db_uri:
        raise HTTPException(status_code=400, detail="Write-access database URI is required.")
    try:
        success, message = db_inspector.apply_ddl(request.write_db_uri, request.ddl_statement)
        if not success:
            raise HTTPException(status_code=500, detail=message)
        return ApplyResult(success=True, message=message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")